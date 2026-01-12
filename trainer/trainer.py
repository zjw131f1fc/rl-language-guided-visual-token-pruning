import numpy as np
import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.trainer import OnpolicyTrainer

from utils.env import MLLMTokenPruningEnv
from utils.policy import PolicyValueNet, TokenPruningPPO, CriticWrapper


def setup_environments(config, mllm, data_loader):
    """
    Create training and testing environments.
    """
    train_samples = data_loader.get_train_samples()
    test_samples = data_loader.get_test_samples()

    def make_train_env():
        return MLLMTokenPruningEnv(
            mllm, train_samples, config,
            seed=np.random.randint(0, 1e6),
            is_training=True
        )

    def make_test_env():
        return MLLMTokenPruningEnv(
            mllm, test_samples, config,
            seed=np.random.randint(0, 1e6),
            is_training=False
        )

    train_envs = DummyVectorEnv([make_train_env for _ in range(config.NUM_TRAIN_ENVS)])
    test_envs = DummyVectorEnv([make_test_env for _ in range(config.NUM_TEST_ENVS)])

    return train_envs, test_envs


def setup_policy(config, mllm, train_envs):
    """
    Initialize the PPO policy and optimizer.
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

    # 创建PPO策略
    policy = TokenPruningPPO(
        actor=net,
        optim=optimizer,
        config=config,
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


def train_agent(config, policy, train_envs, test_envs):
    """
    Run the Tianshou on-policy trainer.
    """
    train_collector = Collector(
        policy, train_envs,
        VectorReplayBuffer(config.BUFFER_SIZE, len(train_envs))
    )
    test_collector = Collector(policy, test_envs, exploration_noise=False)

    print("\nStarting Tianshou On-policy Trainer...")
    print(f"  - Training environments: {config.NUM_TRAIN_ENVS}")
    print(f"  - Test environments: {config.NUM_TEST_ENVS}")
    print(f"  - Max pruning rounds per episode: {config.T_MAX}")
    print(f"  - Training threshold: {config.TRAIN_THRESHOLD}")
    print(f"  - Random masking: {config.ENABLE_RANDOM_MASK} (ratio: {config.RANDOM_MASK_RATIO})")

    result = OnpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=config.EPOCHS,
        step_per_epoch=config.STEP_PER_EPOCH,
        repeat_per_collect=config.REPEAT_PER_COLLECT,
        episode_per_test=config.EPISODE_PER_TEST,
        batch_size=config.BATCH_SIZE,
        step_per_collect=config.STEP_PER_COLLECT,
        verbose=True,
    ).run()

    print(f"\nFinished training! Result:\n{result}")
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
