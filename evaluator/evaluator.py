"""
Evaluator for Token Pruning Policy

评估模式：
- threshold: 使用阈值进行多步剪枝
- none: 不剪枝（基线）
"""

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image


def evaluate_policy(config, policy, mllm, data_loader, num_samples=50):
    """
    快速评估策略性能（用于训练期间）。
    同时评估不剪枝（基线）和剪枝后的性能。
    """
    test_samples = data_loader.get_test_samples()[:num_samples]

    policy.eval()

    # 保存原始模式
    original_mode = config.EVAL_MODE

    results = {"none": {"correct": 0, "kept": 0, "total": 0},
               "threshold": {"correct": 0, "kept": 0, "total": 0}}

    sample_idx = 0
    for sample in test_samples:
        image = sample['image']
        question = sample['question']
        gt_answer = sample['answer']

        if not isinstance(image, Image.Image):
            continue

        # 统一resize到训练时的尺寸
        image = image.convert("RGB").resize(
            (config.TRAIN_IMAGE_SIZE, config.TRAIN_IMAGE_SIZE),
            Image.BILINEAR
        )

        components = mllm.get_components_for_env(image, question)
        if components is None:
            continue

        sample_idx += 1

        # 评估不剪枝
        config.EVAL_MODE = "none"
        result_none = _evaluate_single_sample(policy, config, mllm, components, gt_answer)
        results["none"]["correct"] += result_none['accuracy']
        results["none"]["kept"] += result_none['num_kept']
        results["none"]["total"] += result_none['num_original']

        # 评估剪枝
        config.EVAL_MODE = "threshold"
        result_threshold = _evaluate_single_sample(policy, config, mllm, components, gt_answer)
        results["threshold"]["correct"] += result_threshold['accuracy']
        results["threshold"]["kept"] += result_threshold['num_kept']
        results["threshold"]["total"] += result_threshold['num_original']

        # 打印前10个样本的详细信息
        if sample_idx <= 10:
            print(f"  [{sample_idx}] Q: {question[:50]}...")
            print(f"       GT: '{gt_answer}' | baseline: '{result_none['generated']}' | pruned: '{result_threshold['generated']}'")
            print(f"       baseline_acc={result_none['accuracy']:.0f} | pruned_acc={result_threshold['accuracy']:.0f} | keep={result_threshold['num_kept']}/{result_threshold['num_original']}")

    # 恢复原始模式
    config.EVAL_MODE = original_mode

    n = len(test_samples)
    acc_none = results["none"]["correct"] / n if n > 0 else 0
    acc_threshold = results["threshold"]["correct"] / n if n > 0 else 0
    keep_ratio = results["threshold"]["kept"] / results["threshold"]["total"] if results["threshold"]["total"] > 0 else 0

    print(f"  Eval ({n} samples): baseline={acc_none:.1%} | pruned={acc_threshold:.1%} | keep={keep_ratio:.1%}")

    policy.train()
    return {"acc_baseline": acc_none, "acc_pruned": acc_threshold, "keep_ratio": keep_ratio}


def evaluate_performance(policy, config, mllm, data_loader, logger):
    """
    完整评估策略性能。
    """
    logger.info(f"\nEvaluation Mode: {config.EVAL_MODE}")
    if config.EVAL_MODE == "threshold":
        logger.info(f"Threshold: {config.THRESHOLD}")
    logger.info(f"Decision steps: {config.NUM_DECISION_STEPS}")

    test_samples = data_loader.get_test_samples()
    results = []

    policy.eval()

    for sample in tqdm(test_samples, desc="Evaluating"):
        image = sample['image']
        question = sample['question']
        gt_answer = sample['answer']

        if not isinstance(image, Image.Image):
            continue

        # 统一resize到训练时的尺寸
        image = image.convert("RGB").resize(
            (config.TRAIN_IMAGE_SIZE, config.TRAIN_IMAGE_SIZE),
            Image.BILINEAR
        )

        components = mllm.get_components_for_env(image, question)
        if components is None:
            continue

        result = _evaluate_single_sample(
            policy, config, mllm, components, gt_answer
        )
        results.append(result)

        if len(results) <= 5 or len(results) % 50 == 0:
            logger.info(f"Sample {len(results)}: GT='{gt_answer}' | "
                       f"Pred='{result['generated']}' | "
                       f"Acc={result['accuracy']} | "
                       f"Kept={result['num_kept']}/{result['num_original']}")

    # 统计结果
    avg_accuracy = np.mean([r["accuracy"] for r in results])
    avg_compression = np.mean([r["compression_ratio"] for r in results])
    avg_kept = np.mean([r["num_kept"] for r in results])

    logger.info(f"\n{'='*50}")
    logger.info(f"Results - Mode: {config.EVAL_MODE}")
    logger.info(f"{'='*50}")
    logger.info(f"Samples: {len(results)}")
    logger.info(f"Accuracy: {avg_accuracy:.4f}")
    logger.info(f"Keep Ratio: {avg_compression:.4f}")
    logger.info(f"Avg Tokens: {avg_kept:.1f}")
    logger.info(f"{'='*50}\n")

    return results


def _evaluate_single_sample(policy, config, mllm, components, gt_answer):
    """评估单个样本"""
    visual_features = components['original_visual_features']
    query_embeddings = components['query_embeddings']
    current_num_patches = components['current_num_patches']
    text_embeds_part1 = components['text_embeds_part1']
    text_embeds_part2 = components['text_embeds_part2']

    # 处理patch数量
    if current_num_patches > config.MAX_PATCHES:
        visual_features = visual_features[:, :config.MAX_PATCHES, :]
        current_num_patches = config.MAX_PATCHES

    # 初始化活跃token掩码
    active_mask = np.ones(current_num_patches, dtype=bool)

    if config.EVAL_MODE == "threshold":
        # 多步阈值剪枝
        for step in range(config.NUM_DECISION_STEPS):
            if active_mask.sum() == 0:
                break

            active_indices = np.where(active_mask)[0]
            num_active = len(active_indices)

            # 准备输入
            active_features = visual_features[:, active_indices, :]
            padded_features = np.zeros(
                (1, config.MAX_PATCHES, visual_features.size(-1)),
                dtype=np.float32
            )
            padded_features[:, :num_active, :] = active_features.cpu().numpy()

            valid_token_mask = np.zeros(config.MAX_PATCHES, dtype=np.float32)
            valid_token_mask[:num_active] = 1.0

            obs = {
                "visual_features": padded_features,
                "query_embeddings": query_embeddings.cpu().numpy(),
                "valid_token_mask": valid_token_mask.reshape(1, -1)
            }

            # 前向传播
            with torch.no_grad():
                logits, _ = policy.actor(obs)
                probs = torch.sigmoid(logits)

            # 阈值决策
            decisions = (probs[:num_active] > config.THRESHOLD).cpu().numpy()

            # 更新掩码
            new_active_mask = np.zeros_like(active_mask)
            for i, idx in enumerate(active_indices):
                if decisions[i]:
                    new_active_mask[idx] = True
            active_mask = new_active_mask

    # config.EVAL_MODE == "none" 时不做任何剪枝

    # 获取保留的token
    kept_indices = np.where(active_mask)[0]
    num_kept = len(kept_indices)

    # 如果没有保留任何token，保留概率最高的一个
    if num_kept == 0:
        padded_features = np.zeros(
            (1, config.MAX_PATCHES, visual_features.size(-1)),
            dtype=np.float32
        )
        padded_features[:, :current_num_patches, :] = visual_features.cpu().numpy()[:, :current_num_patches, :]

        valid_token_mask = np.zeros(config.MAX_PATCHES, dtype=np.float32)
        valid_token_mask[:current_num_patches] = 1.0

        obs = {
            "visual_features": padded_features,
            "query_embeddings": query_embeddings.cpu().numpy(),
            "valid_token_mask": valid_token_mask.reshape(1, -1)
        }

        with torch.no_grad():
            logits, _ = policy.actor(obs)
            probs = torch.sigmoid(logits)

        best_idx = probs[:current_num_patches].argmax().item()
        kept_indices = np.array([best_idx])
        num_kept = 1

    # 生成答案
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

    return {
        "accuracy": accuracy,
        "compression_ratio": num_kept / current_num_patches,
        "num_kept": num_kept,
        "num_original": current_num_patches,
        "gt_answer": gt_answer,
        "generated": generated_text
    }


def evaluate_with_different_thresholds(policy, config, mllm, data_loader, logger, thresholds=None):
    """使用不同阈值评估，生成效率-精度曲线。"""
    if thresholds is None:
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]

    logger.info(f"\nMulti-threshold evaluation: {thresholds}")

    all_results = {}
    original_threshold = config.THRESHOLD
    original_mode = config.EVAL_MODE

    config.EVAL_MODE = "threshold"

    for threshold in thresholds:
        config.THRESHOLD = threshold
        logger.info(f"\n--- Threshold: {threshold} ---")

        results = evaluate_performance(policy, config, mllm, data_loader, logger)

        avg_accuracy = np.mean([r["accuracy"] for r in results])
        avg_compression = np.mean([r["compression_ratio"] for r in results])

        all_results[threshold] = {
            "accuracy": avg_accuracy,
            "compression_ratio": avg_compression,
        }

    # 恢复配置
    config.THRESHOLD = original_threshold
    config.EVAL_MODE = original_mode

    # 打印汇总
    logger.info(f"\n{'='*50}")
    logger.info("Threshold vs Accuracy vs Keep Ratio")
    logger.info(f"{'='*50}")
    logger.info(f"{'Threshold':<12} {'Accuracy':<12} {'Keep Ratio':<12}")
    logger.info(f"{'-'*36}")

    for threshold in thresholds:
        r = all_results[threshold]
        logger.info(f"{threshold:<12.2f} {r['accuracy']:<12.4f} {r['compression_ratio']:<12.4f}")

    logger.info(f"{'='*50}\n")

    return all_results
