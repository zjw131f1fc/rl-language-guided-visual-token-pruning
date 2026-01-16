"""
GRPO (Group Relative Policy Optimization) Implementation

GRPO通过组内样本的相对比较来估计优势函数，而不是依赖价值函数估计。
这简化了训练架构，因为不需要单独的价值网络。

优势计算：Â_GRPO = (rt - μ_group) / (σ_group + ε)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tianshou.data import Batch, ReplayBuffer, to_torch_as
from tianshou.policy import BasePolicy
from tianshou.data.types import RolloutBatchProtocol
from typing import Any, Dict, Optional
from gymnasium import spaces

from tianshou.policy.modelfree.ppo import PPOTrainingStats
from utils.policy import PolicyValueNet, BernoulliActionDistribution


class GRPOPolicy(BasePolicy):
    """
    Group Relative Policy Optimization (GRPO) for Token Pruning.

    GRPO不需要价值网络，通过组内相对比较来估计优势函数。
    """

    def __init__(
        self,
        *,
        actor: PolicyValueNet,
        optim: torch.optim.Optimizer,
        config,
        action_space: spaces.Space,
        eps_clip: float = 0.2,
        group_size: int = 4,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        deterministic_eval: bool = True,
        **kwargs
    ):
        super().__init__(action_space=action_space, **kwargs)
        self.actor = actor
        self.optim = optim
        self.config = config
        self.eps_clip = eps_clip
        self.group_size = group_size
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self._deterministic_eval = deterministic_eval

    def forward(self, batch: Batch, state: Optional[Dict] = None, **kwargs) -> Batch:
        """前向传播，返回动作和分布"""
        # 只使用actor的策略输出，忽略value
        logits, _ = self.actor(batch.obs)

        # 获取有效token掩码（需要flatten以匹配logits的形状）
        if isinstance(batch.obs, Batch):
            valid_mask = torch.as_tensor(
                batch.obs.valid_token_mask,
                dtype=torch.float32,
                device=logits.device
            ).flatten()
        else:
            valid_mask = torch.as_tensor(
                batch.obs["valid_token_mask"],
                dtype=torch.float32,
                device=logits.device
            ).flatten()

        # 创建分布
        dist = BernoulliActionDistribution(logits, valid_mask)

        # 采样或确定性选择
        if self._deterministic_eval and not self.training:
            # 确定性：使用阈值
            act = (torch.sigmoid(logits) > self.config.THRESHOLD).float() * valid_mask
        else:
            act = dist.sample()

        # 保持 [num_envs, action_dim] 的形状给 Tianshou
        act = act.unsqueeze(0)

        # 不返回 dist 对象，因为 Tianshou 的 Batch 无法处理它
        # 保存 logits 和 valid_mask 用于后续计算 log_prob
        return Batch(act=act, state=state, logits=logits, valid_mask=valid_mask)

    def process_fn(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray
    ) -> RolloutBatchProtocol:
        """
        处理收集的数据，计算GRPO优势。

        GRPO优势：Â = (r - μ_group) / (σ_group + ε)
        """
        # 计算旧的log_prob
        with torch.no_grad():
            logp_old = []
            for minibatch in batch.split(256, shuffle=False, merge_last=True):
                result = self(minibatch)
                # 使用logits和valid_mask重建分布
                dist = BernoulliActionDistribution(result.logits, result.valid_mask)
                # 确保action在正确的设备上
                act = to_torch_as(minibatch.act, result.logits)
                logp_old.append(dist.log_prob(act).unsqueeze(0))
            batch.logp_old = torch.cat(logp_old, dim=0)

        # 计算GRPO优势
        rewards = to_torch_as(batch.rew, batch.logp_old)
        batch.adv = self._compute_grpo_advantage(rewards)

        # GRPO不需要returns，但为了兼容性设置一个
        batch.returns = rewards

        return batch

    def _compute_grpo_advantage(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        计算GRPO组相对优势。

        将样本分组，在组内计算相对优势：
        Â = (r - μ_group) / (σ_group + ε)
        """
        n = len(rewards)
        advantages = torch.zeros_like(rewards)

        # 将样本分成组
        num_groups = max(1, n // self.group_size)

        for i in range(num_groups):
            start_idx = i * self.group_size
            end_idx = min((i + 1) * self.group_size, n)

            group_rewards = rewards[start_idx:end_idx]

            # 组内归一化
            mu = group_rewards.mean()
            sigma = group_rewards.std()

            advantages[start_idx:end_idx] = (group_rewards - mu) / (sigma + 1e-8)

        # 处理剩余样本（如果有）
        remaining_start = num_groups * self.group_size
        if remaining_start < n:
            remaining_rewards = rewards[remaining_start:]
            mu = remaining_rewards.mean()
            sigma = remaining_rewards.std()
            advantages[remaining_start:] = (remaining_rewards - mu) / (sigma + 1e-8)

        return advantages

    def learn(
        self,
        batch: RolloutBatchProtocol,
        batch_size: int | None,
        repeat: int,
        *args,
        **kwargs
    ) -> Dict[str, float]:
        """GRPO学习步骤"""
        losses, clip_losses, ent_losses = [], [], []

        for _ in range(repeat):
            for minibatch in batch.split(batch_size or len(batch), merge_last=True):
                # 前向传播
                logits, _ = self.actor(minibatch.obs)

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

                # GRPO优势（已在process_fn中计算）
                adv = to_torch_as(minibatch.adv, ratio)

                # GRPO裁剪损失（与PPO相同）
                surr1 = ratio * adv
                surr2 = ratio.clamp(1 - self.eps_clip, 1 + self.eps_clip) * adv
                clip_loss = -torch.min(surr1, surr2).mean()

                # 熵损失（鼓励探索）
                ent_loss = dist.entropy().mean()

                # 总损失（GRPO不需要价值损失）
                loss = clip_loss - self.ent_coef * ent_loss

                # 反向传播
                self.optim.zero_grad()
                loss.backward()
                if self.max_grad_norm:
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.optim.step()

                losses.append(loss.item())
                clip_losses.append(clip_loss.item())
                ent_losses.append(ent_loss.item())

        return PPOTrainingStats.from_sequences(
            losses=losses,
            clip_losses=clip_losses,
            vf_losses=[0.0] * len(losses),  # GRPO不使用value loss
            ent_losses=ent_losses,
            gradient_steps=len(losses),
        )

    def update(
        self,
        sample_size: int,
        buffer: ReplayBuffer,
        **kwargs
    ) -> Dict[str, Any]:
        """更新策略"""
        batch, indices = buffer.sample(sample_size)
        batch = self.process_fn(batch, buffer, indices)
        result = self.learn(batch, kwargs.get("batch_size"), kwargs.get("repeat", 1))
        return result


def create_grpo_policy(config, mllm):
    """
    创建GRPO策略。

    Args:
        config: 配置对象
        mllm: MLLM包装器

    Returns:
        GRPOPolicy实例
    """
    # 创建策略网络（GRPO也使用相同的网络，但不使用value head）
    net = PolicyValueNet(
        vision_dim=mllm.feature_dim,
        hidden_dim=config.HIDDEN_DIM,
        num_heads=config.NUM_ATTENTION_HEADS,
        dropout=0.1
    ).to(config.DEVICE)

    # 加载预训练权重（如果有）
    if config.PRETRAIN_WEIGHTS_PATH is not None:
        net.load_pretrain_weights(config.PRETRAIN_WEIGHTS_PATH)

    # 创建优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=config.LR)

    # 创建动作空间（与PPO一致）
    action_dim = config.BATCH_SIZE * config.MAX_PATCHES
    action_space = spaces.MultiBinary(action_dim)

    # 创建GRPO策略
    policy = GRPOPolicy(
        actor=net,
        optim=optimizer,
        config=config,
        action_space=action_space,
        eps_clip=config.EPS_CLIP,
        group_size=config.GRPO_GROUP_SIZE,
        ent_coef=config.ENT_COEF,
        max_grad_norm=0.5,
    )

    return policy
