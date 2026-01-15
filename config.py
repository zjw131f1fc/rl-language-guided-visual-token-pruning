import torch
import os
from datetime import datetime

# --- Global Settings ---
os.environ["HF_HOME"] = "/data/users/zjw/huggingface_cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,4"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Logging ---
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# --- MLLM and Dataset Configuration ---
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DATASET_NAME = "lmms-lab/MME"
DATASET_SPLIT = "test[:1000]"
TRAIN_TEST_SPLIT_RATIO = 0.8

# --- Environment Hyperparameters ---
# Reward coefficients (incremental batch-averaged reward)
ALPHA = 1.0  # Task reward weight
BETA = 0.5   # Efficiency reward weight

# --- Training Image Settings ---
# 训练时使用固定分辨率，确保每个样本的视觉token数量相同
TRAIN_IMAGE_SIZE = 224  # 训练时图像分辨率 224x224
TRAIN_NUM_PATCHES = None  # 训练时的视觉token数量，None表示从第一个样本动态获取

# --- Multi-round Pruning Settings ---
NUM_DECISION_STEPS = 5  # 决策步数（训练和推理一致）
TRAIN_THRESHOLD = 0.7  # 训练时的剪枝阈值 τ_train

# --- Random Masking Settings ---
ENABLE_RANDOM_MASK = False  # 禁用随机掩码（随机初始化的policy已经会剪枝约一半token）
RANDOM_MASK_RATIO = 0.2  # 随机掩码的比例（当前已禁用）

# --- Policy Network Architecture ---
HIDDEN_DIM = 512  # 隐藏层维度
NUM_ATTENTION_HEADS = 8  # 共享注意力模块的头数

# --- PPO Policy & Trainer Hyperparameters ---
MAX_PATCHES = 1800  # 推理时最大patch数
LR = 1e-5
GAMMA_PPO = 0.99  # Discount factor for PPO
EPS_CLIP = 0.2
VF_COEF = 0.5
ENT_COEF = 0.01
GAE_LAMBDA = 0.95
REWARD_NORMALIZATION = True

# --- GRPO Settings ---
USE_GRPO = False  # 是否使用GRPO替代PPO（GRPO不需要价值网络）
GRPO_GROUP_SIZE = 4  # GRPO组内样本数量

# --- Training Loop Settings ---
EPOCHS = 10
BATCH_SIZE = 4  # 环境批量大小（同时处理的样本数）
BUFFER_SIZE = 500
STEP_PER_COLLECT = 10
STEP_PER_EPOCH = 100
REPEAT_PER_COLLECT = 2
NUM_TRAIN_ENVS = 1
NUM_TEST_ENVS = 1
EPISODE_PER_TEST = 10

# --- Inference Settings ---
THRESHOLD = 0.5  # 推理时的Token保留阈值（可调节以平衡效率和精度）

# --- Evaluation Settings ---
EVAL_INTERVAL = 5  # 每隔多少个epoch进行一次评估，0表示不评估
EVAL_MODE = "full"  # 可选值："full", "budget", "none"
EVAL_BUDGET_RATIO = 0.5  # 仅在 EVAL_MODE=="budget" 时生效

# --- Pretrain Settings ---
PRETRAIN_WEIGHTS_PATH = None  # 预训练权重路径，None表示从头训练
