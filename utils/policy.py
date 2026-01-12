import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tianshou.policy import PPOPolicy
from tianshou.data import Batch, ReplayBuffer, to_torch_as
from tianshou.data.types import RolloutBatchProtocol
from typing import Any


class SharedAttentionModule(nn.Module):
    """
    共享注意力模块，作为策略网络和价值网络的共同特征提取器。
    处理视觉token和文本查询的拼接序列。
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Multi-head self-attention
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, mask=None):
        """
        Args:
            X: [batch_size, seq_len, d_model] - 拼接后的视觉token和查询
            mask: [batch_size, seq_len] - 有效token掩码 (1=有效, 0=无效)
        Returns:
            Z: [batch_size, seq_len, d_model] - 注意力输出
        """
        batch_size, seq_len, _ = X.shape

        # Linear projections
        Q = self.W_Q(X)  # [B, L, d]
        K = self.W_K(X)  # [B, L, d]
        V = self.W_V(X)  # [B, L, d]

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Apply mask if provided
        if mask is not None:
            # mask: [B, L] -> [B, 1, 1, L] for broadcasting
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        attn_output = self.W_O(attn_output)

        # Residual connection and layer norm
        Z = self.layer_norm(X + self.dropout(attn_output))

        return Z


class PolicyValueNet(nn.Module):
    """
    策略和价值网络，使用共享注意力模块。

    策略网络：输出每个token的保留概率 pt,i = σ(w⊤gi + b)
    价值网络：全局平均池化后输出标量价值
    """
    def __init__(self, vision_dim, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.vision_dim = vision_dim
        self.hidden_dim = hidden_dim

        # 查询嵌入投影层（将查询对齐到视觉特征维度）
        self.query_proj = nn.Linear(vision_dim, hidden_dim)

        # 视觉特征投影层
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)

        # 共享注意力模块
        self.shared_attention = SharedAttentionModule(hidden_dim, num_heads, dropout)

        # === 策略网络头部 ===
        # fi = GELU(W1*zi + b1)
        self.policy_W1 = nn.Linear(hidden_dim, hidden_dim)
        # gi = LayerNorm(zi + W2*fi + b2)
        self.policy_W2 = nn.Linear(hidden_dim, hidden_dim)
        self.policy_ln = nn.LayerNorm(hidden_dim)
        # pt,i = σ(w⊤gi + b)
        self.policy_head = nn.Linear(hidden_dim, 1)

        # === 价值网络头部 ===
        # v = GELU(U1*z̄ + c1)
        self.value_U1 = nn.Linear(hidden_dim, hidden_dim)
        # V(st) = u⊤LayerNorm(z̄ + U2*v + c2) + bv
        self.value_U2 = nn.Linear(hidden_dim, hidden_dim)
        self.value_ln = nn.LayerNorm(hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs, state=None, info={}):
        """
        Args:
            obs: 观察字典，包含:
                - visual_features: [B, N, d] 视觉token特征
                - query_embeddings: [B, 1, d] 查询嵌入
                - valid_token_mask: [B, N] 有效token掩码
        Returns:
            logits: [B, N] 每个token的保留logits
            value: [B] 状态价值
        """
        device = next(self.parameters()).device

        # 处理输入
        if isinstance(obs, Batch):
            visual_features = torch.as_tensor(obs.visual_features, dtype=torch.float32, device=device)
            query_embeddings = torch.as_tensor(obs.query_embeddings, dtype=torch.float32, device=device)
            valid_token_mask = torch.as_tensor(obs.valid_token_mask, dtype=torch.float32, device=device)
        else:
            visual_features = torch.as_tensor(obs["visual_features"], dtype=torch.float32, device=device)
            query_embeddings = torch.as_tensor(obs["query_embeddings"], dtype=torch.float32, device=device)
            valid_token_mask = torch.as_tensor(obs["valid_token_mask"], dtype=torch.float32, device=device)

        batch_size, num_patches, _ = visual_features.shape

        # 投影到隐藏维度
        visual_proj = self.vision_proj(visual_features)  # [B, N, hidden_dim]
        query_proj = self.query_proj(query_embeddings.squeeze(1)).unsqueeze(1)  # [B, 1, hidden_dim]

        # 拼接视觉token和查询: X = [Ht; q̃]
        X = torch.cat([visual_proj, query_proj], dim=1)  # [B, N+1, hidden_dim]

        # 构建注意力掩码（包含查询token）
        query_mask = torch.ones(batch_size, 1, device=device)
        attn_mask = torch.cat([valid_token_mask, query_mask], dim=1)  # [B, N+1]

        # 共享注意力模块
        Z = self.shared_attention(X, attn_mask)  # [B, N+1, hidden_dim]

        # 分离视觉token和查询token的输出
        Z_visual = Z[:, :num_patches, :]  # [B, N, hidden_dim]

        # === 策略网络 ===
        # fi = GELU(W1*zi + b1)
        fi = F.gelu(self.policy_W1(Z_visual))
        # gi = LayerNorm(zi + W2*fi + b2) - 残差连接
        gi = self.policy_ln(Z_visual + self.policy_W2(fi))
        # pt,i的logits（sigmoid在外部应用）
        logits = self.policy_head(gi).squeeze(-1)  # [B, N]

        # === 价值网络 ===
        # 全局平均池化（只对有效token）
        valid_mask_expanded = valid_token_mask.unsqueeze(-1)  # [B, N, 1]
        num_valid = valid_token_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
        z_bar = (Z_visual * valid_mask_expanded).sum(dim=1) / num_valid  # [B, hidden_dim]

        # v = GELU(U1*z̄ + c1)
        v = F.gelu(self.value_U1(z_bar))
        # V(st) = u⊤LayerNorm(z̄ + U2*v + c2) + bv - 残差连接
        value_input = self.value_ln(z_bar + self.value_U2(v))
        value = self.value_head(value_input).squeeze(-1)  # [B]

        return logits, value

    def get_probs(self, obs):
        """获取每个token的保留概率"""
        logits, _ = self.forward(obs)
        return torch.sigmoid(logits)

    def load_pretrain_weights(self, path):
        """加载预训练权重"""
        state_dict = torch.load(path, map_location='cpu')
        self.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrain weights from {path}")


class BernoulliActionDistribution:
    """
    伯努利分布，用于处理每个token的二元决策。
    """
    def __init__(self, logits, valid_mask):
        """
        Args:
            logits: [B, N] 每个token的保留logits
            valid_mask: [B, N] 有效token掩码
        """
        self.logits = logits
        self.valid_mask = valid_mask
        self.probs = torch.sigmoid(logits)
        self._dist = torch.distributions.Bernoulli(logits=logits)

    def sample(self):
        """采样动作: 每个token的保留/剪枝决策"""
        actions = self._dist.sample()  # [B, N]
        # 无效token强制设为0（剪枝）
        actions = actions * self.valid_mask
        return actions

    def log_prob(self, actions):
        """计算动作的对数概率"""
        # 只计算有效token的log_prob
        log_probs = self._dist.log_prob(actions)  # [B, N]
        # 对有效token的log_prob求和
        log_probs = (log_probs * self.valid_mask).sum(dim=-1)  # [B]
        return log_probs

    def entropy(self):
        """计算熵（用于探索）"""
        ent = self._dist.entropy()  # [B, N]
        # 只计算有效token的熵
        ent = (ent * self.valid_mask).sum(dim=-1)  # [B]
        return ent


class TokenPruningPPO(PPOPolicy):
    """
    用于Token剪枝的PPO策略。
    动作空间：对所有token同时做二元决策。
    """
    def __init__(
        self,
        *,
        actor: PolicyValueNet,
        optim: torch.optim.Optimizer,
        config,
        **kwargs
    ):
        # 创建一个虚拟的critic（实际使用actor内部的value head）
        self.config = config

        # 调用父类初始化，但我们会覆盖大部分行为
        super().__init__(
            actor=actor,
            critic=CriticWrapper(actor),
            optim=optim,
            dist_fn=lambda *args: None,  # 我们自己处理分布
            action_space=None,
            action_scaling=False,
            **kwargs
        )

    def forward(self, batch, state=None, **kwargs):
        """前向传播，返回动作和分布"""
        logits, value = self.actor(batch.obs)

        # 获取有效token掩码
        if isinstance(batch.obs, Batch):
            valid_mask = torch.as_tensor(
                batch.obs.valid_token_mask,
                dtype=torch.float32,
                device=logits.device
            )
        else:
            valid_mask = torch.as_tensor(
                batch.obs["valid_token_mask"],
                dtype=torch.float32,
                device=logits.device
            )

        # 创建分布
        dist = BernoulliActionDistribution(logits, valid_mask)

        # 采样动作
        act = dist.sample()

        return Batch(act=act, state=state, dist=dist, logits=logits, value=value.flatten())

    def learn(self, batch: RolloutBatchProtocol, batch_size: int | None, repeat: int, *args, **kwargs):
        """PPO学习步骤"""
        losses, clip_losses, vf_losses, ent_losses = [], [], [], []

        for _ in range(repeat):
            for minibatch in batch.split(batch_size or len(batch), merge_last=True):
                # 前向传播
                logits, value = self.actor(minibatch.obs)

                # 获取有效token掩码
                if isinstance(minibatch.obs, Batch):
                    valid_mask = torch.as_tensor(
                        minibatch.obs.valid_token_mask,
                        dtype=torch.float32,
                        device=logits.device
                    )
                else:
                    valid_mask = torch.as_tensor(
                        minibatch.obs["valid_token_mask"],
                        dtype=torch.float32,
                        device=logits.device
                    )

                # 创建分布
                dist = BernoulliActionDistribution(logits, valid_mask)

                # 计算新的log_prob
                act = to_torch_as(minibatch.act, logits)
                log_prob = dist.log_prob(act)

                # 计算ratio
                old_log_prob = to_torch_as(minibatch.logp_old, log_prob)
                ratio = (log_prob - old_log_prob).exp()

                # 优势归一化
                adv = to_torch_as(minibatch.adv, ratio)
                if self.norm_adv:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # PPO裁剪损失
                surr1 = ratio * adv
                surr2 = ratio.clamp(1 - self.eps_clip, 1 + self.eps_clip) * adv
                clip_loss = -torch.min(surr1, surr2).mean()

                # 价值损失
                returns = to_torch_as(minibatch.returns, value)
                vf_loss = F.mse_loss(value.flatten(), returns)

                # 熵损失
                ent_loss = dist.entropy().mean()

                # 总损失
                loss = clip_loss + self.vf_coef * vf_loss - self.ent_coef * ent_loss

                # 反向传播
                self.optim.zero_grad()
                loss.backward()
                if self.max_grad_norm:
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.optim.step()

                losses.append(loss.item())
                clip_losses.append(clip_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(ent_loss.item())

        return {
            "loss": np.mean(losses),
            "clip_loss": np.mean(clip_losses),
            "vf_loss": np.mean(vf_losses),
            "ent_loss": np.mean(ent_losses),
        }

    def process_fn(self, batch: RolloutBatchProtocol, buffer: ReplayBuffer, indices: np.ndarray):
        """处理收集的数据，计算GAE"""
        # 计算returns和advantages
        batch = self._compute_returns(batch, buffer, indices)

        # 计算旧的log_prob
        batch.act = to_torch_as(batch.act, batch.v_s)
        with torch.no_grad():
            logp_old = []
            for minibatch in batch.split(256, shuffle=False, merge_last=True):
                result = self(minibatch)
                logp_old.append(result.dist.log_prob(minibatch.act))
            batch.logp_old = torch.cat(logp_old, dim=0)

        return batch


class CriticWrapper(nn.Module):
    """价值网络包装器"""
    def __init__(self, actor_critic_net):
        super().__init__()
        self.net = actor_critic_net

    def forward(self, obs, *args, **kwargs):
        _, value = self.net(obs, *args, **kwargs)
        return value
