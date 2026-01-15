import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
from PIL import Image


def get_num_patches_from_sample(mllm, sample, config):
    """从样本获取视觉token数量"""
    image = sample.get('image')
    question = sample.get('question')

    if not isinstance(image, Image.Image) or not question:
        return None

    # resize到训练分辨率
    image = image.convert("RGB").resize(
        (config.TRAIN_IMAGE_SIZE, config.TRAIN_IMAGE_SIZE),
        Image.BILINEAR
    )

    components = mllm.get_components_for_env(image, question)
    if components is None:
        return None

    return components['current_num_patches']


class BatchMLLMTokenPruningEnv(gym.Env):
    """
    批量Token剪枝环境。

    关键特性：
    - 同时处理B个样本，每步对所有样本的所有token做决策
    - 动作空间: [B * N] 二元决策
    - 奖励: 批量平均的增量奖励
    - 支持多步决策（NUM_DECISION_STEPS）
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

        # 动态获取num_patches（从第一个有效样本）
        if is_training:
            if config.TRAIN_NUM_PATCHES is None:
                self.num_patches = self._get_num_patches_from_first_sample()
            else:
                self.num_patches = config.TRAIN_NUM_PATCHES
        else:
            self.num_patches = config.MAX_PATCHES

        # 多步决策设置
        self.num_decision_steps = config.NUM_DECISION_STEPS
        self.enable_random_mask = config.ENABLE_RANDOM_MASK
        self.random_mask_ratio = config.RANDOM_MASK_RATIO

        # Batch state
        self.current_step = 0
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

        # 动作空间: B * N 个二元决策（展平）
        self.action_space = spaces.MultiBinary(self.batch_size * self.num_patches)

        # 观察空间
        self.observation_space = spaces.Dict({
            "visual_features": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.batch_size, self.num_patches, self.feature_dim),
                dtype=np.float32
            ),
            "query_embeddings": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.batch_size, 1, self.feature_dim),
                dtype=np.float32
            ),
            "valid_token_mask": spaces.Box(
                low=0, high=1,
                shape=(self.batch_size, self.num_patches),
                dtype=np.float32
            )
        })

    def _get_num_patches_from_first_sample(self):
        """从第一个有效样本获取视觉token数量"""
        for sample in self.vqa_samples:
            num_patches = get_num_patches_from_sample(self.mllm, sample, self.config)
            if num_patches is not None:
                print(f"Detected num_patches from first sample: {num_patches}")
                return num_patches
        raise ValueError("Could not determine num_patches from any sample")

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

        self.current_step = 0
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

        # 随机掩码（可选）
        if self.is_training and self.enable_random_mask:
            num_to_mask = int(self.num_patches * self.random_mask_ratio)
            if num_to_mask > 0:
                for b in range(self.batch_size):
                    mask_indices = random.sample(range(self.num_patches), num_to_mask)
                    self.batch_active_masks[b, mask_indices] = False
                self.batch_num_tokens = self.batch_active_masks.sum(dim=1)

        # Compute initial task scores
        self.batch_prev_scores = self._compute_batch_task_scores()

        return self._get_obs(), {"step": self.current_step}

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

    def _get_obs(self):
        """获取批次观察（展平格式）"""
        return {
            "visual_features": self.batch_visual_features.cpu().numpy(),
            "query_embeddings": self.batch_query_embeddings.cpu().numpy(),
            "valid_token_mask": self.batch_active_masks.float().cpu().numpy()
        }

    def step(self, action):
        """
        执行一步决策。

        Args:
            action: [B * N] 展平的二元数组，或 [B, N] 二维数组

        Returns:
            obs, reward, terminated, truncated, info
        """
        action = np.asarray(action)
        if action.ndim == 1:
            action = action.reshape(self.batch_size, self.num_patches)

        actions = torch.as_tensor(action, dtype=torch.bool, device=self.device)

        # Record previous token counts
        prev_num_tokens = self.batch_num_tokens.clone()

        # Update active masks: keep only tokens that are both active AND kept by action
        self.batch_active_masks = self.batch_active_masks & actions
        self.batch_num_tokens = self.batch_active_masks.sum(dim=1)

        # Compute batch reward
        reward, r_task, r_eff = self._compute_batch_reward(prev_num_tokens)

        # Update step
        self.current_step += 1

        # Check termination
        terminated = (self.current_step >= self.num_decision_steps) or (self.batch_num_tokens.sum().item() == 0)
        truncated = False

        # 计算保留比例
        keep_ratio = self.batch_num_tokens.float().mean().item() / self.num_patches

        # Build info
        info = {
            "step": self.current_step,
            "batch_num_tokens_before": prev_num_tokens.tolist(),
            "batch_num_tokens_after": self.batch_num_tokens.tolist(),
            "batch_compression_ratio": (self.batch_num_tokens.float() / self.num_patches).tolist(),
            "r_task": r_task,
            "r_eff": r_eff,
        }

        # 打印每步信息
        print(f"  Step {self.current_step}/{self.num_decision_steps} | "
              f"keep={keep_ratio:.1%} | "
              f"r_task={r_task:+.3f} | r_eff={r_eff:+.3f} | total={reward:+.3f}")

        if terminated:
            final_scores = self._compute_batch_task_scores()
            info["batch_final_accuracy"] = final_scores.tolist()
            info["mean_final_accuracy"] = final_scores.mean().item()
            final_keep = self.batch_num_tokens.float().mean().item() / self.num_patches
            print(f"  Episode done | acc={final_scores.mean().item():.1%} | "
                  f"tokens={self.batch_num_tokens.tolist()} | keep={final_keep:.1%}")

        return self._get_obs(), reward, terminated, truncated, info

    def _compute_batch_reward(self, prev_num_tokens):
        """
        计算批次平均增量奖励。

        Rtask = α * (1/B) * Σ(TaskScore_t+1 - TaskScore_t)
        Reff = β * (1/B) * Σ(1 - K_t+1 / K_t)

        Returns:
            total_reward, r_task, r_eff
        """
        # Compute current task scores
        current_scores = self._compute_batch_task_scores()

        # Task reward: batch average of score changes
        score_changes = current_scores.mean() - self.batch_prev_scores.mean()
        r_task = self.config.ALPHA * score_changes.item()

        # Efficiency reward: batch average of pruning ratios
        safe_prev = prev_num_tokens.float().clamp(min=1)
        pruning_ratios = 1 - self.batch_num_tokens.float() / safe_prev
        r_eff = self.config.BETA * pruning_ratios.mean().item()

        # Update previous scores
        self.batch_prev_scores = current_scores

        return r_task + r_eff, r_task, r_eff