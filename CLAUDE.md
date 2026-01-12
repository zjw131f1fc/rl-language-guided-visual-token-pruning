# CLAUDE.md

## 项目概述

这是一个基于强化学习的视觉Token剪枝项目，用于多模态大语言模型(MLLM)。系统使用PPO算法训练智能Token剪枝策略，在保持VQA任务性能的同时降低计算成本。

## 快速开始

```bash
# 激活环境
conda activate rl-tianshou

# 运行训练和评估
python main.py
```

## 项目结构

```
├── main.py              # 主入口：训练+评估流程
├── config.py            # 所有超参数配置
├── data/
│   ├── base_loader.py   # 数据加载器基类
│   └── mme_loader.py    # MME VQA数据集加载
├── models/
│   ├── base_mllm.py     # MLLM基类
│   └── qwen_mllm.py     # Qwen2.5-VL实现
├── trainer/
│   └── trainer.py       # PPO训练器(基于Tianshou)
├── evaluator/
│   └── evaluator.py     # 评估器(三种模式)
└── utils/
    ├── env.py           # RL环境(Gymnasium)
    └── policy.py        # 策略网络(Actor-Critic)
```

## 核心组件

### RL环境 (utils/env.py)
- `MLLMTokenPruningEnv`: 基于Gymnasium的环境
- 动作空间: MultiDiscrete [max_patches, 2] - 选择token索引和保留/剪枝决策
- 观察空间: 视觉特征、查询嵌入、剪枝掩码、有效token掩码
- 奖励函数: 任务准确率(α=1.0) + 效率(β=0.5) + 语义相似度(γ=0.1)

### 策略网络 (utils/policy.py)
- `PolicyValueNet`: Actor-Critic架构
- `CompositeActionPPO`: 复合动作空间的PPO实现
- `CompositeDistribution`: 处理token选择和保留/剪枝的联合分布

### MLLM包装器 (models/qwen_mllm.py)
- 模型: Qwen/Qwen2.5-VL-3B-Instruct
- 功能: 提取视觉特征、生成答案、处理剪枝后的嵌入

## 关键配置 (config.py)

| 参数 | 值 | 说明 |
|------|-----|------|
| MODEL_NAME | Qwen/Qwen2.5-VL-3B-Instruct | 基础MLLM |
| DATASET_NAME | lmms-lab/MME | VQA数据集 |
| MAX_PATCHES | 1800 | 最大视觉token数 |
| ALPHA | 1.0 | 任务准确率权重 |
| BETA | 0.5 | 效率权重 |
| GAMMA | 0.1 | 语义相似度权重 |
| LR | 1e-5 | 学习率 |
| HIDDEN_DIM | 512 | 隐藏层维度 |
| THRESHOLD | 0.5 | 保留/剪枝阈值 |
| BUDGET_RATIO | 0.5 | budget模式保留比例 |

## 评估模式

1. `none`: 无剪枝基线
2. `full`: 基于阈值剪枝所有token
3. `budget`: 只保留top-k个token(按选择概率)

## 扩展指南

- 添加新数据集: 继承 `BaseDataLoader`
- 添加新MLLM: 继承 `BaseMLLM`

## 依赖

主要依赖: tianshou, gymnasium, transformers, torch, datasets
