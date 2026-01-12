import torch
import numpy as np
from tqdm import tqdm


# def deterministic_pruning(policy, env, num_episodes, config, # logger):

#     # logger.info(f"\nStarting deterministic pruning for evaluation... Mode: {config.EVAL_MODE}")
#     results = []

#     for _ in tqdm(range(num_episodes), desc="Evaluating"):
#         obs, info = env.reset()
#         done = False
#         episode_reward = 0

#         if config.EVAL_MODE == "full":
#             # 遍历所有位置，每次决策一个token（index, decision）
#             while not done:
#                 obs_batch = {
#                     'obs': {
#                         'visual_features': torch.tensor(obs['visual_features'], dtype=torch.float32).unsqueeze(0).to(policy.device),
#                         'query_embeddings': torch.tensor(obs['query_embeddings'], dtype=torch.float32).unsqueeze(0).to(policy.device),
#                         'pruning_mask': torch.tensor(obs['pruning_mask'], dtype=torch.int8).unsqueeze(0).to(policy.device),
#                         'valid_token_mask': torch.tensor(obs['valid_token_mask'], dtype=torch.int8).unsqueeze(0).to(policy.device)
#                     },
#                     'info': {}
#                 }
#                 with torch.no_grad():
#                     logits, _ = policy.actor(obs_batch['obs'])
#                 # logits: [index_logits, decision_logits]
#                 index = torch.argmax(logits[0], dim=1).item()
#                 decision = torch.argmax(logits[1], dim=1).item()
#                 action = [index, decision]
#                 obs, reward, terminated, truncated, info = env.step(action)
#                 done = terminated or truncated or info.get("error")
#                 episode_reward += reward

#         elif config.EVAL_MODE == "budget":
#             # 只对概率最高的预算个位置做决策
#             max_num = obs['valid_token_mask'].sum()
#             budget = int(max_num * config.EVAL_BUDGET_RATIO)
#             if budget < 1:
#                 budget = 1
#             obs_batch = {
#                 'obs': {
#                     'visual_features': torch.tensor(obs['visual_features'], dtype=torch.float32).unsqueeze(0).to(policy.device),
#                     'query_embeddings': torch.tensor(obs['query_embeddings'], dtype=torch.float32).unsqueeze(0).to(policy.device),
#                     'pruning_mask': torch.tensor(obs['pruning_mask'], dtype=torch.int8).unsqueeze(0).to(policy.device),
#                     'valid_token_mask': torch.tensor(obs['valid_token_mask'], dtype=torch.int8).unsqueeze(0).to(policy.device)
#                 },
#                 'info': {}
#             }
#             with torch.no_grad():
#                 index_logits, decision_logits = policy.actor(obs_batch['obs'])
#             # index_logits: [1, max_patches], decision_logits: [1, 2]
#             index_logits = index_logits[0, 0].cpu().numpy()
#             # 选出概率最高的 budget 个 index
#             top_indices = np.argsort(index_logits)[-budget:][::-1]
#             for idx in top_indices:
#                 obs_batch = {
#                     'obs': {
#                         'visual_features': torch.tensor(obs['visual_features'], dtype=torch.float32).unsqueeze(0).to(policy.device),
#                         'query_embeddings': torch.tensor(obs['query_embeddings'], dtype=torch.float32).unsqueeze(0).to(policy.device),
#                         'pruning_mask': torch.tensor(obs['pruning_mask'], dtype=torch.int8).unsqueeze(0).to(policy.device),
#                         'valid_token_mask': torch.tensor(obs['valid_token_mask'], dtype=torch.int8).unsqueeze(0).to(policy.device)
#                     },
#                     'info': {}
#                 }
#                 with torch.no_grad():
#                     _, decision_logits = policy.actor(obs_batch['obs'])
#                 decision = torch.argmax(decision_logits, dim=1).item()
#                 action = [idx, decision]
#                 obs, reward, terminated, truncated, info = env.step(action)
#                 done = terminated or truncated or info.get("error")
#                 episode_reward += reward
#                 if done:
#                     break
#             # 剩余未决策的token不再处理

#         else:
#             raise ValueError(f"Unknown EVAL_MODE: {config.EVAL_MODE}")

#         results.append({
#             'reward': episode_reward,
#             'final_info': info
#         })

#     # 结果统计与打印
#     avg_reward = np.mean([r['reward'] for r in results])
#     avg_acc = np.mean([r['final_info'].get('accuracy', 0) for r in results])
#     avg_compression = np.mean([r['final_info'].get('compression_ratio', 0) for r in results])
#     # logger.info("\n--- Evaluation Results ---")
#     # logger.info(f"Average Reward: {avg_reward:.4f}")
#     # logger.info(f"Average Accuracy: {avg_acc:.4f}")
#     # logger.info(f"Average Compression Ratio: {avg_compression:.4f}")
#     # logger.info("--------------------------\n")
#     return results


def evaluate_performance(policy, config, mllm, data_loader, logger):
    """
    使用训练好的网络进行评估。
    """
    logger.info(f"\nStarting evaluation... Mode: {config.EVAL_MODE}")

    test_samples = data_loader.get_test_samples()
    results = []

    for sample in tqdm(test_samples, desc="Evaluating"):
        # 提取样本信息
        image, question, gt_answer = sample['image'], sample['question'], sample['answer']

        # 使用 mllm 提取特征
        components = mllm.get_components_for_env(image, question)
        visual_features = components['original_visual_features']
        query_embeddings = components['query_embeddings']
        if components['current_num_patches'] > config.MAX_PATCHES:
            print(f"Warning: Sample has {components['current_num_patches']} patches, exceeding MAX_PATCHES {config.MAX_PATCHES}. Truncating.")
        current_num_patches = min(components['current_num_patches'], config.MAX_PATCHES)
        text_embeds_part1 = components['text_embeds_part1']
        text_embeds_part2 = components['text_embeds_part2']

        # 准备输入数据
        padded_features = np.zeros((1, config.MAX_PATCHES, visual_features.size(-1)), dtype=np.float32)
        visual_features = visual_features[:, :config.MAX_PATCHES, :]
        actual_features = visual_features.squeeze(0).cpu().numpy()
        padded_features[:, :current_num_patches, :] = actual_features

        valid_token_mask = np.zeros(config.MAX_PATCHES, dtype=np.int8)
        valid_token_mask[:current_num_patches] = 1
        pruning_mask = np.full(config.MAX_PATCHES, -1, dtype=np.int8)
        pruning_mask = pruning_mask.reshape(1, -1)
        obs = {
            "visual_features": padded_features,
            "query_embeddings": query_embeddings.unsqueeze(1).cpu().numpy(),
            "pruning_mask": pruning_mask,
            "valid_token_mask": valid_token_mask
        }

        with torch.no_grad():
            select_logits, keep_logits = policy.actor(obs)[0]
            keep_logits = torch.sigmoid(keep_logits)
            select_logits[:, current_num_patches:] = -float('inf')

        if config.EVAL_MODE == "full":
            # 遍历所有位置，按阈值决策
            decisions = (keep_logits > config.THRESHOLD).cpu().numpy()[0]
        elif config.EVAL_MODE == "budget":
            # 仅对概率最高的预算个位置决策
            budget = int(config.EVAL_BUDGET_RATIO * current_num_patches)
            top_indices = torch.topk(select_logits, budget, dim=1).indices.cpu().numpy()
            decisions = np.ones(config.MAX_PATCHES, dtype=int)
            decisions[top_indices] = (keep_logits > config.THRESHOLD)[:, top_indices].cpu().numpy()[0]
        elif config.EVAL_MODE == "none":
            # 不进行剪枝，保留所有原始token
            decisions = np.ones(config.MAX_PATCHES, dtype=int)
        else:
            raise ValueError(f"Unknown EVAL_MODE: {config.EVAL_MODE}")

        # 使用 mllm 生成答案
        pruned_visual_features = torch.tensor(padded_features[:, (decisions == 1) & (valid_token_mask == 1), :], device=text_embeds_part1.device)
        # pruned_visual_features = self.original_visual_features[:, kept_indices, :]
        final_embeddings = torch.cat([text_embeds_part1, pruned_visual_features, text_embeds_part2], dim=1)
        attention_mask = torch.ones(final_embeddings.shape[:2], dtype=torch.long, device=mllm.device)
        generated_text = mllm.generate_answer(final_embeddings, attention_mask)
        accuracy = 1.0 if gt_answer.lower() in generated_text.lower() else 0.0
        print(f"GT: {gt_answer} | Pred: {generated_text} | Acc: {accuracy}")

        # 计算压缩率
        num_kept = np.sum((decisions == 1) & (valid_token_mask == 1))
        compression_ratio = num_kept / current_num_patches

        results.append({"accuracy": accuracy, "compression_ratio": compression_ratio})

    # 统计平均结果
    avg_accuracy = np.mean([r["accuracy"] for r in results])
    avg_compression = np.mean([r["compression_ratio"] for r in results])

    logger.info(f"\n--- Evaluation Results for {config.EVAL_MODE} ---")
    logger.info(f"Average Accuracy: {avg_accuracy:.4f}")
    logger.info(f"Average Compression Ratio: {avg_compression:.4f}")
    logger.info("--------------------------\n")

    return results
