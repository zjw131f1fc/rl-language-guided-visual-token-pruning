"""
Evaluator for Token Pruning Policy

推理时采用一次性剪枝策略：
- 单次前向传播获取每个token的保留概率
- 使用阈值τ做二元决策：pi > τ 则保留，否则剪枝
- 阈值可调节以平衡效率和精度
"""

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image


def evaluate_performance(policy, config, mllm, data_loader, logger):
    """
    使用训练好的网络进行评估。

    推理协议：一次性剪枝
    - 策略网络同时处理所有视觉token
    - 输出每个token的保留概率 {pi}
    - 使用阈值τ做二元决策
    """
    logger.info(f"\nStarting evaluation... Mode: {config.EVAL_MODE}")
    logger.info(f"Inference threshold: {config.THRESHOLD}")

    test_samples = data_loader.get_test_samples()
    results = []

    policy.eval()

    for sample in tqdm(test_samples, desc="Evaluating"):
        image = sample['image']
        question = sample['question']
        gt_answer = sample['answer']

        if not isinstance(image, Image.Image):
            continue

        # 使用 mllm 提取特征
        components = mllm.get_components_for_env(image, question)
        if components is None:
            continue

        visual_features = components['original_visual_features']
        query_embeddings = components['query_embeddings']
        current_num_patches = components['current_num_patches']
        text_embeds_part1 = components['text_embeds_part1']
        text_embeds_part2 = components['text_embeds_part2']

        # 处理patch数量
        if current_num_patches > config.MAX_PATCHES:
            visual_features = visual_features[:, :config.MAX_PATCHES, :]
            current_num_patches = config.MAX_PATCHES

        # 准备输入数据
        padded_features = np.zeros(
            (1, config.MAX_PATCHES, visual_features.size(-1)),
            dtype=np.float32
        )
        actual_features = visual_features.squeeze(0).cpu().numpy()
        padded_features[:, :current_num_patches, :] = actual_features[:current_num_patches, :]

        valid_token_mask = np.zeros(config.MAX_PATCHES, dtype=np.float32)
        valid_token_mask[:current_num_patches] = 1.0

        obs = {
            "visual_features": padded_features,
            "query_embeddings": query_embeddings.cpu().numpy(),
            "valid_token_mask": valid_token_mask.reshape(1, -1)
        }

        # 一次性前向传播获取保留概率
        with torch.no_grad():
            logits, _ = policy.actor(obs)
            probs = torch.sigmoid(logits)  # [1, MAX_PATCHES]

        # 根据评估模式决定剪枝策略
        if config.EVAL_MODE == "none":
            # 不进行剪枝，保留所有原始token
            decisions = np.ones(config.MAX_PATCHES, dtype=bool)

        elif config.EVAL_MODE == "full":
            # 使用阈值进行一次性剪枝
            decisions = (probs > config.THRESHOLD).cpu().numpy()[0]

        elif config.EVAL_MODE == "budget":
            # 预算模式：只保留top-k个token
            budget = int(config.EVAL_BUDGET_RATIO * current_num_patches)
            budget = max(1, budget)  # 至少保留1个token

            # 获取概率最高的budget个token
            probs_np = probs.cpu().numpy()[0]
            # 只考虑有效token
            probs_np[current_num_patches:] = -1

            top_indices = np.argsort(probs_np)[-budget:]
            decisions = np.zeros(config.MAX_PATCHES, dtype=bool)
            decisions[top_indices] = True

        else:
            raise ValueError(f"Unknown EVAL_MODE: {config.EVAL_MODE}")

        # 只保留有效且被决定保留的token
        final_mask = decisions & (valid_token_mask.astype(bool))
        kept_indices = np.where(final_mask)[0]
        num_kept = len(kept_indices)

        if num_kept == 0:
            # 如果没有保留任何token，强制保留概率最高的一个
            probs_np = probs.cpu().numpy()[0]
            probs_np[current_num_patches:] = -1
            best_idx = np.argmax(probs_np)
            kept_indices = np.array([best_idx])
            num_kept = 1

        # 使用保留的token生成答案
        kept_features = visual_features[:, kept_indices, :]
        final_embeddings = torch.cat([
            text_embeds_part1,
            kept_features,
            text_embeds_part2
        ], dim=1)

        attention_mask = torch.ones(
            final_embeddings.shape[:2],
            dtype=torch.long,
            device=mllm.device
        )

        generated_text = mllm.generate_answer(final_embeddings, attention_mask)
        accuracy = 1.0 if gt_answer.lower() in generated_text.lower() else 0.0

        # 计算压缩率
        compression_ratio = num_kept / current_num_patches

        results.append({
            "accuracy": accuracy,
            "compression_ratio": compression_ratio,
            "num_kept": num_kept,
            "num_original": current_num_patches,
            "gt_answer": gt_answer,
            "generated": generated_text
        })

        # 打印部分结果
        if len(results) <= 5 or len(results) % 50 == 0:
            logger.info(f"Sample {len(results)}: GT='{gt_answer}' | Pred='{generated_text}' | "
                       f"Acc={accuracy} | Kept={num_kept}/{current_num_patches}")

    # 统计平均结果
    avg_accuracy = np.mean([r["accuracy"] for r in results])
    avg_compression = np.mean([r["compression_ratio"] for r in results])
    avg_kept = np.mean([r["num_kept"] for r in results])

    logger.info(f"\n{'='*50}")
    logger.info(f"Evaluation Results - Mode: {config.EVAL_MODE}")
    logger.info(f"{'='*50}")
    logger.info(f"Total samples: {len(results)}")
    logger.info(f"Average Accuracy: {avg_accuracy:.4f}")
    logger.info(f"Average Compression Ratio: {avg_compression:.4f}")
    logger.info(f"Average Tokens Kept: {avg_kept:.1f}")
    logger.info(f"{'='*50}\n")

    return results


def evaluate_with_different_thresholds(policy, config, mllm, data_loader, logger, thresholds=None):
    """
    使用不同阈值评估，生成效率-精度曲线数据。

    Args:
        thresholds: 要测试的阈值列表，默认为 [0.1, 0.3, 0.5, 0.7, 0.9]
    """
    if thresholds is None:
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]

    logger.info(f"\nEvaluating with multiple thresholds: {thresholds}")

    all_results = {}
    original_threshold = config.THRESHOLD
    original_mode = config.EVAL_MODE

    # 强制使用full模式
    config.EVAL_MODE = "full"

    for threshold in thresholds:
        config.THRESHOLD = threshold
        logger.info(f"\n--- Threshold: {threshold} ---")

        results = evaluate_performance(policy, config, mllm, data_loader, logger)

        avg_accuracy = np.mean([r["accuracy"] for r in results])
        avg_compression = np.mean([r["compression_ratio"] for r in results])

        all_results[threshold] = {
            "accuracy": avg_accuracy,
            "compression_ratio": avg_compression,
            "detailed_results": results
        }

    # 恢复原始配置
    config.THRESHOLD = original_threshold
    config.EVAL_MODE = original_mode

    # 打印汇总
    logger.info(f"\n{'='*60}")
    logger.info("Summary: Accuracy vs Compression at Different Thresholds")
    logger.info(f"{'='*60}")
    logger.info(f"{'Threshold':<12} {'Accuracy':<12} {'Compression':<12}")
    logger.info(f"{'-'*36}")

    for threshold in thresholds:
        r = all_results[threshold]
        logger.info(f"{threshold:<12.2f} {r['accuracy']:<12.4f} {r['compression_ratio']:<12.4f}")

    logger.info(f"{'='*60}\n")

    return all_results
