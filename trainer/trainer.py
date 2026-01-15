import numpy as np
import torch
from tianshou.data import Collector, ReplayBuffer
from tianshou.trainer import OnpolicyTrainer

from utils.env import BatchMLLMTokenPruningEnv
from utils.policy import PolicyValueNet, TokenPruningPPO, CriticWrapper


def setup_environments(config, mllm, data_loader):
    """
    Create training environment only.

    Returns:
        train_env, num_patches (实际的视觉token数量)
    """
    train_samples = data_loader.get_train_samples()

    # 创建训练环境
    train_env = BatchMLLMTokenPruningEnv(
        mllm, train_samples, config,
        batch_size=config.BATCH_SIZE,
        seed=42,
        is_training=True
    )
    num_patches = train_env.num_patches
    print(f"Actual num_patches for training: {num_patches}")

    return train_env, num_patches


def setup_policy(config, mllm, num_patches):
    """
    Initialize the PPO policy and optimizer.

    Args:
        config: 配置
        mllm: MLLM模型
        num_patches: 实际的视觉token数量（从环境获取）
    """
    # 创建策略网络
    net = PolicyValueNet(
        vision_dim=mllm.feature_dim,
        hidden_dim=config.HIDDEN_DIM,
        num_heads=config.NUM_ATTENTION_HEADS,
        dropout=0.1
    ).to(config.DEVICE)

    # 加载预训练权重（如果有）
    if config.PRETRAIN_WEIGHTS_PATH is not None:
        net.load_pretrain_weights(config.PRETRAIN_WEIGHTS_PATH)
        print(f"Loaded pretrain weights from {config.PRETRAIN_WEIGHTS_PATH}")

    # 创建优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=config.LR)

    # 创建PPO策略，传入实际的 num_patches
    policy = TokenPruningPPO(
        actor=net,
        optim=optimizer,
        config=config,
        num_patches=num_patches,
        eps_clip=config.EPS_CLIP,
        discount_factor=config.GAMMA_PPO,
        vf_coef=config.VF_COEF,
        ent_coef=config.ENT_COEF,
        gae_lambda=config.GAE_LAMBDA,
        reward_normalization=config.REWARD_NORMALIZATION,
        advantage_normalization=True,
        max_grad_norm=0.5,
    )

    return policy


def train_agent(config, policy, train_env, mllm=None, data_loader=None):
    """
    Run the Tianshou on-policy trainer with periodic evaluation.

    Args:
        config: 配置
        policy: PPO策略
        train_env: 训练环境
        mllm: MLLM模型（用于评估，可选）
        data_loader: 数据加载器（用于评估，可选）
    """
    train_collector = Collector(
        policy, train_env,
        ReplayBuffer(config.BUFFER_SIZE)
    )

    print("\n" + "="*60)
    print("Starting PPO Training")
    print("="*60)
    print(f"  Batch size: {config.BATCH_SIZE} samples")
    print(f"  Steps per episode: {config.NUM_DECISION_STEPS}")
    print(f"  Steps per collect: {config.STEP_PER_COLLECT}")
    print(f"  Epochs: {config.EPOCHS}")
    if config.EVAL_INTERVAL > 0:
        print(f"  Eval interval: every {config.EVAL_INTERVAL} epochs")
    print("="*60 + "\n")

    # 用于跟踪上次评估的 epoch
    last_eval_epoch = [0]

    def epoch_hook(epoch, env_step):
        """每个 epoch 开始时的回调"""
        # 检查是否需要评估
        if config.EVAL_INTERVAL > 0 and mllm is not None and data_loader is not None:
            if epoch > 0 and epoch % config.EVAL_INTERVAL == 0 and epoch != last_eval_epoch[0]:
                last_eval_epoch[0] = epoch
                print(f"\n{'='*60}")
                print(f"Evaluation at Epoch {epoch}")
                print("="*60)
                from evaluator.evaluator import evaluate_policy
                evaluate_policy(config, policy, mllm, data_loader)
                print("="*60 + "\n")

    result = OnpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=None,
        max_epoch=config.EPOCHS,
        step_per_epoch=config.STEP_PER_EPOCH,
        repeat_per_collect=config.REPEAT_PER_COLLECT,
        episode_per_test=0,
        batch_size=config.BATCH_SIZE,
        step_per_collect=config.STEP_PER_COLLECT,
        verbose=True,
        train_fn=epoch_hook,
    ).run()

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)

    return policy


def save_policy(policy, path):
    """
    Save the policy network weights.
    """
    torch.save(policy.actor.state_dict(), path)
    print(f"Policy saved to {path}")


def load_policy(policy, path):
    """
    Load the policy network weights.
    """
    policy.actor.load_state_dict(torch.load(path, map_location='cpu'))
    print(f"Policy loaded from {path}")
    return policy
