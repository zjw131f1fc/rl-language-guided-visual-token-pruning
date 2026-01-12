import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
from PIL import Image


class MLLMTokenPruningEnv(gym.Env):
    """
    RL Environment for MLLM token pruning with multi-round pruning.

    Key features:
    - Multi-round pruning: Each episode has T_max rounds
    - Action space: Binary decision for ALL tokens simultaneously
    - State: Unprocessed visual tokens Ht and query q
    - Reward: Incremental batch-averaged reward (task + efficiency)
    """

    def __init__(self, mllm_wrapper, vqa_samples, config, seed=None, is_training=True):
        super().__init__()
        self.mllm = mllm_wrapper
        self.vqa_samples = vqa_samples
        self.config = config
        self.device = config.DEVICE
        self.is_training = is_training
        self._env_seed = seed
        self._set_seed(seed)

        if not self.vqa_samples:
            raise ValueError("VQA samples list cannot be empty.")

        self.feature_dim = self.mllm.feature_dim

        # 训练时使用固定数量的patches，推理时使用最大值
        if is_training:
            self.num_patches = config.TRAIN_NUM_PATCHES
        else:
            self.num_patches = config.MAX_PATCHES

        # Multi-round pruning settings
        self.t_max = config.T_MAX
        self.train_threshold = config.TRAIN_THRESHOLD
        self.enable_random_mask = config.ENABLE_RANDOM_MASK
        self.random_mask_ratio = config.RANDOM_MASK_RATIO

        # Episode state
        self.current_round = 0
        self.current_sample = None
        self.current_question = ""
        self.gt_answer = ""

        # Visual features state (changes each round as tokens are pruned)
        self.visual_features = None  # Current unprocessed tokens Ht
        self.original_visual_features = None  # Original features for answer generation
        self.query_embeddings = None
        self.text_embeds_part1 = None
        self.text_embeds_part2 = None

        # Token tracking
        self.active_indices = None  # Indices of currently active (unprocessed) tokens
        self.current_num_tokens = 0  # Kt: number of unprocessed tokens

        # Previous task score for incremental reward
        self.prev_task_score = None

        # Define action and observation spaces
        # Action: Binary decision for each token (0=prune, 1=keep)
        self.action_space = spaces.MultiBinary(self.num_patches)

        # Observation: visual features + query embeddings + valid mask
        self.observation_space = spaces.Dict({
            "visual_features": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.num_patches, self.feature_dim),
                dtype=np.float32
            ),
            "query_embeddings": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(1, self.feature_dim),
                dtype=np.float32
            ),
            "valid_token_mask": spaces.Box(
                low=0, high=1,
                shape=(self.num_patches,),
                dtype=np.float32
            )
        })

    def _set_seed(self, seed):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    def _get_vqa_sample(self):
        return random.choice(self.vqa_samples)

    def _compute_task_score(self):
        """计算当前状态的任务得分（准确率）"""
        if self.current_num_tokens == 0:
            return 0.0

        # 使用当前保留的token生成答案
        with torch.no_grad():
            # 获取当前活跃token的特征
            active_features = self.original_visual_features[:, self.active_indices, :]

            # 构建最终嵌入
            final_embeddings = torch.cat([
                self.text_embeds_part1,
                active_features,
                self.text_embeds_part2
            ], dim=1)

            attention_mask = torch.ones(
                final_embeddings.shape[:2],
                dtype=torch.long,
                device=self.device
            )

            generated_text = self.mllm.generate_answer(final_embeddings, attention_mask)
            is_correct = 1.0 if self.gt_answer.lower() in generated_text.lower() else 0.0

        return is_correct

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._set_seed(seed)
        super().reset(seed=seed)

        # 获取有效样本
        while True:
            sample = self._get_vqa_sample()
            image = sample.get('image')
            question = sample.get('question')
            answer = sample.get('answer')

            if not isinstance(image, Image.Image) or not question or not answer:
                continue

            # 训练时resize图像到固定分辨率
            if self.is_training:
                image = image.convert("RGB").resize(
                    (self.config.TRAIN_IMAGE_SIZE, self.config.TRAIN_IMAGE_SIZE),
                    Image.BILINEAR
                )

            components = self.mllm.get_components_for_env(image, question)
            if components:
                break

        self.current_sample = sample
        self.current_question = question
        self.gt_answer = answer
        self.current_round = 0

        # 存储原始特征
        self.original_visual_features = components["original_visual_features"]
        actual_num_patches = self.original_visual_features.shape[1]

        # 处理patch数量
        if actual_num_patches > self.num_patches:
            self.original_visual_features = self.original_visual_features[:, :self.num_patches, :]
            actual_num_patches = self.num_patches

        self.text_embeds_part1 = components["text_embeds_part1"]
        self.text_embeds_part2 = components["text_embeds_part2"]
        self.query_embeddings = components["query_embeddings"]

        # 初始化活跃token索引
        self.active_indices = list(range(actual_num_patches))
        self.current_num_tokens = actual_num_patches

        # 应用随机掩码（如果启用且是第一轮）
        if self.is_training and self.enable_random_mask:
            num_to_mask = int(self.current_num_tokens * self.random_mask_ratio)
            if num_to_mask > 0:
                mask_indices = random.sample(self.active_indices, num_to_mask)
                self.active_indices = [i for i in self.active_indices if i not in mask_indices]
                self.current_num_tokens = len(self.active_indices)

        # 设置当前视觉特征（只包含活跃token）
        self._update_visual_features()

        # 计算初始任务得分
        self.prev_task_score = self._compute_task_score()

        return self._get_obs(), {"round": self.current_round}

    def _update_visual_features(self):
        """更新当前视觉特征（只包含活跃token）"""
        if len(self.active_indices) > 0:
            active_indices_tensor = torch.tensor(
                self.active_indices,
                device=self.device,
                dtype=torch.long
            )
            self.visual_features = self.original_visual_features[:, active_indices_tensor, :]
        else:
            self.visual_features = torch.zeros(
                1, 0, self.feature_dim,
                device=self.device
            )

    def _get_obs(self):
        """获取当前观察"""
        # Pad visual features to fixed size
        padded_features = np.zeros((self.num_patches, self.feature_dim), dtype=np.float32)

        if self.visual_features is not None and self.visual_features.shape[1] > 0:
            actual_features = self.visual_features.squeeze(0).cpu().numpy()
            num_active = min(actual_features.shape[0], self.num_patches)
            padded_features[:num_active, :] = actual_features[:num_active, :]

        # Valid token mask
        valid_token_mask = np.zeros(self.num_patches, dtype=np.float32)
        valid_token_mask[:self.current_num_tokens] = 1.0

        # Query embeddings
        query_emb = self.query_embeddings.cpu().numpy() if self.query_embeddings is not None \
            else np.zeros((1, self.feature_dim), dtype=np.float32)

        return {
            "visual_features": padded_features,
            "query_embeddings": query_emb,
            "valid_token_mask": valid_token_mask
        }

    def step(self, action):
        """
        执行一轮剪枝。

        Args:
            action: [num_patches] 二元数组，1=保留，0=剪枝

        Returns:
            obs, reward, terminated, truncated, info
        """
        action = np.asarray(action).flatten()

        # 只考虑当前活跃token的决策
        active_decisions = action[:self.current_num_tokens]

        # 记录剪枝前的token数量
        prev_num_tokens = self.current_num_tokens

        # 根据阈值或采样决定保留哪些token
        # 在训练时，使用训练阈值
        if self.is_training:
            # action已经是采样后的二元决策
            keep_mask = active_decisions == 1
        else:
            keep_mask = active_decisions == 1

        # 更新活跃token索引
        new_active_indices = []
        for i, idx in enumerate(self.active_indices):
            if i < len(keep_mask) and keep_mask[i]:
                new_active_indices.append(idx)

        self.active_indices = new_active_indices
        self.current_num_tokens = len(self.active_indices)

        # 更新视觉特征
        self._update_visual_features()

        # 计算奖励
        reward = self._compute_reward(prev_num_tokens)

        # 更新轮次
        self.current_round += 1

        # 检查是否结束
        # 结束条件：达到最大轮次 或 没有token了
        terminated = (self.current_round >= self.t_max) or (self.current_num_tokens == 0)
        truncated = False

        # 构建info
        info = {
            "round": self.current_round,
            "num_tokens_before": prev_num_tokens,
            "num_tokens_after": self.current_num_tokens,
            "compression_ratio": self.current_num_tokens / self.num_patches if self.num_patches > 0 else 0,
        }

        if terminated:
            # 最终评估
            final_score = self._compute_task_score()
            info["final_accuracy"] = final_score
            info["final_num_tokens"] = self.current_num_tokens

        return self._get_obs(), reward, terminated, truncated, info

    def _compute_reward(self, prev_num_tokens):
        """
        计算增量奖励。

        Reward = Rtask + Reff
        Rtask = α * (TaskScore_t+1 - TaskScore_t)
        Reff = β * (1 - K_t+1 / K_t)
        """
        # 计算当前任务得分
        current_task_score = self._compute_task_score()

        # 任务奖励：得分变化
        r_task = self.config.ALPHA * (current_task_score - self.prev_task_score)

        # 效率奖励：剪枝比例
        if prev_num_tokens > 0:
            r_eff = self.config.BETA * (1 - self.current_num_tokens / prev_num_tokens)
        else:
            r_eff = 0.0

        # 更新前一轮得分
        self.prev_task_score = current_task_score

        return r_task + r_eff


class BatchMLLMTokenPruningEnv(gym.Env):
    """
    批量版本的Token剪枝环境。
    同时处理B个样本，奖励基于批次平均计算。
    """

    def __init__(self, mllm_wrapper, vqa_samples, config, batch_size, seed=None, is_training=True):
        super().__init__()
        self.mllm = mllm_wrapper
        self.vqa_samples = vqa_samples
        self.config = config
        self.batch_size = batch_size
        self.device = config.DEVICE
        self.is_training = is_training

        self._set_seed(seed)

        if not self.vqa_samples:
            raise ValueError("VQA samples list cannot be empty.")

        self.feature_dim = self.mllm.feature_dim

        # 训练时使用固定数量的patches
        if is_training:
            self.num_patches = config.TRAIN_NUM_PATCHES
        else:
            self.num_patches = config.MAX_PATCHES

        # Multi-round pruning settings
        self.t_max = config.T_MAX
        self.train_threshold = config.TRAIN_THRESHOLD
        self.enable_random_mask = config.ENABLE_RANDOM_MASK
        self.random_mask_ratio = config.RANDOM_MASK_RATIO

        # Batch state
        self.current_round = 0
        self.batch_samples = []
        self.batch_questions = []
        self.batch_answers = []

        # Batch visual features [B, N, d]
        self.batch_visual_features = None
        self.batch_original_features = None
        self.batch_query_embeddings = None
        self.batch_text_embeds_part1 = []
        self.batch_text_embeds_part2 = []

        # Token tracking for each sample in batch
        self.batch_active_masks = None  # [B, N] boolean mask
        self.batch_num_tokens = None  # [B] number of active tokens per sample

        # Previous task scores [B]
        self.batch_prev_scores = None

        # Define spaces
        self.action_space = spaces.MultiBinary(self.num_patches)
        self.observation_space = spaces.Dict({
            "visual_features": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.num_patches, self.feature_dim),
                dtype=np.float32
            ),
            "query_embeddings": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(1, self.feature_dim),
                dtype=np.float32
            ),
            "valid_token_mask": spaces.Box(
                low=0, high=1,
                shape=(self.num_patches,),
                dtype=np.float32
            )
        })

    def _set_seed(self, seed):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    def _get_batch_samples(self):
        """获取一批有效样本"""
        samples = []
        while len(samples) < self.batch_size:
            sample = random.choice(self.vqa_samples)
            image = sample.get('image')
            question = sample.get('question')
            answer = sample.get('answer')

            if isinstance(image, Image.Image) and question and answer:
                samples.append(sample)
        return samples

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._set_seed(seed)
        super().reset(seed=seed)

        self.current_round = 0
        self.batch_samples = self._get_batch_samples()
        self.batch_questions = []
        self.batch_answers = []
        self.batch_text_embeds_part1 = []
        self.batch_text_embeds_part2 = []

        batch_features = []
        batch_queries = []

        for sample in self.batch_samples:
            image = sample['image']
            question = sample['question']
            answer = sample['answer']

            self.batch_questions.append(question)
            self.batch_answers.append(answer)

            # Resize for training
            if self.is_training:
                image = image.convert("RGB").resize(
                    (self.config.TRAIN_IMAGE_SIZE, self.config.TRAIN_IMAGE_SIZE),
                    Image.BILINEAR
                )

            components = self.mllm.get_components_for_env(image, question)

            # Handle features
            features = components["original_visual_features"]
            if features.shape[1] > self.num_patches:
                features = features[:, :self.num_patches, :]
            elif features.shape[1] < self.num_patches:
                # Pad
                pad_size = self.num_patches - features.shape[1]
                padding = torch.zeros(1, pad_size, self.feature_dim, device=features.device)
                features = torch.cat([features, padding], dim=1)

            batch_features.append(features)
            batch_queries.append(components["query_embeddings"])
            self.batch_text_embeds_part1.append(components["text_embeds_part1"])
            self.batch_text_embeds_part2.append(components["text_embeds_part2"])

        # Stack batch
        self.batch_original_features = torch.cat(batch_features, dim=0)  # [B, N, d]
        self.batch_visual_features = self.batch_original_features.clone()
        self.batch_query_embeddings = torch.cat(batch_queries, dim=0)  # [B, 1, d]

        # Initialize active masks
        self.batch_active_masks = torch.ones(
            self.batch_size, self.num_patches,
            dtype=torch.bool, device=self.device
        )
        self.batch_num_tokens = torch.full(
            (self.batch_size,), self.num_patches,
            dtype=torch.long, device=self.device
        )

        # Apply random masking
        if self.is_training and self.enable_random_mask:
            num_to_mask = int(self.num_patches * self.random_mask_ratio)
            if num_to_mask > 0:
                for b in range(self.batch_size):
                    mask_indices = random.sample(range(self.num_patches), num_to_mask)
                    self.batch_active_masks[b, mask_indices] = False
                self.batch_num_tokens = self.batch_active_masks.sum(dim=1)

        # Compute initial task scores
        self.batch_prev_scores = self._compute_batch_task_scores()

        # Return first sample's observation (for compatibility with standard gym interface)
        return self._get_obs(0), {"round": self.current_round}

    def _compute_batch_task_scores(self):
        """计算批次中每个样本的任务得分"""
        scores = []

        with torch.no_grad():
            for b in range(self.batch_size):
                active_mask = self.batch_active_masks[b]
                active_indices = torch.where(active_mask)[0]

                if len(active_indices) == 0:
                    scores.append(0.0)
                    continue

                # Get active features
                active_features = self.batch_original_features[b:b+1, active_indices, :]

                # Build final embeddings
                final_embeddings = torch.cat([
                    self.batch_text_embeds_part1[b],
                    active_features,
                    self.batch_text_embeds_part2[b]
                ], dim=1)

                attention_mask = torch.ones(
                    final_embeddings.shape[:2],
                    dtype=torch.long,
                    device=self.device
                )

                generated_text = self.mllm.generate_answer(final_embeddings, attention_mask)
                is_correct = 1.0 if self.batch_answers[b].lower() in generated_text.lower() else 0.0
                scores.append(is_correct)

        return torch.tensor(scores, device=self.device)

    def _get_obs(self, batch_idx=0):
        """获取指定样本的观察"""
        # Get features for this sample
        features = self.batch_visual_features[batch_idx].cpu().numpy()
        query = self.batch_query_embeddings[batch_idx:batch_idx+1].cpu().numpy()
        valid_mask = self.batch_active_masks[batch_idx].float().cpu().numpy()

        return {
            "visual_features": features,
            "query_embeddings": query,
            "valid_token_mask": valid_mask
        }

    def get_batch_obs(self):
        """获取整个批次的观察"""
        return {
            "visual_features": self.batch_visual_features.cpu().numpy(),
            "query_embeddings": self.batch_query_embeddings.cpu().numpy(),
            "valid_token_mask": self.batch_active_masks.float().cpu().numpy()
        }

    def step(self, action):
        """
        执行一轮剪枝（单样本接口，用于兼容标准gym）。
        """
        # 这里假设action是针对第一个样本的
        return self.batch_step(action.reshape(1, -1))

    def batch_step(self, actions):
        """
        批量执行一轮剪枝。

        Args:
            actions: [B, N] 二元数组

        Returns:
            obs, reward, terminated, truncated, info
        """
        actions = torch.as_tensor(actions, dtype=torch.bool, device=self.device)

        # Record previous token counts
        prev_num_tokens = self.batch_num_tokens.clone()

        # Update active masks: keep only tokens that are both active AND kept by action
        self.batch_active_masks = self.batch_active_masks & actions
        self.batch_num_tokens = self.batch_active_masks.sum(dim=1)

        # Compute batch reward
        reward = self._compute_batch_reward(prev_num_tokens)

        # Update round
        self.current_round += 1

        # Check termination
        terminated = (self.current_round >= self.t_max) or (self.batch_num_tokens.sum() == 0)
        truncated = False

        # Build info
        info = {
            "round": self.current_round,
            "batch_num_tokens_before": prev_num_tokens.cpu().numpy(),
            "batch_num_tokens_after": self.batch_num_tokens.cpu().numpy(),
            "batch_compression_ratio": (self.batch_num_tokens.float() / self.num_patches).cpu().numpy(),
        }

        if terminated:
            final_scores = self._compute_batch_task_scores()
            info["batch_final_accuracy"] = final_scores.cpu().numpy()
            info["mean_final_accuracy"] = final_scores.mean().item()

        return self._get_obs(0), reward, terminated, truncated, info

    def _compute_batch_reward(self, prev_num_tokens):
        """
        计算批次平均增量奖励。

        Rtask = α * (1/B) * Σ(TaskScore_t+1 - TaskScore_t)
        Reff = β * (1/B) * Σ(1 - K_t+1 / K_t)
        """
        # Compute current task scores
        current_scores = self._compute_batch_task_scores()

        # Task reward: batch average of score changes
        score_changes = current_scores - self.batch_prev_scores
        r_task = self.config.ALPHA * score_changes.mean().item()

        # Efficiency reward: batch average of pruning ratios
        # Avoid division by zero
        safe_prev = prev_num_tokens.float().clamp(min=1)
        pruning_ratios = 1 - self.batch_num_tokens.float() / safe_prev
        r_eff = self.config.BETA * pruning_ratios.mean().item()

        # Update previous scores
        self.batch_prev_scores = current_scores

        return r_task + r_eff
