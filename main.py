import os
import torch
import logging

import config
from data.base_loader import get_data_loader
from models.base_mllm import get_mllm
from trainer.trainer import setup_environments, setup_policy, train_agent, save_policy
from evaluator.evaluator import evaluate_performance, evaluate_with_different_thresholds


def setup_logger():
    """Sets up the logger to write to a file and the console."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create handlers
    file_handler = logging.FileHandler(config.LOG_FILE)
    console_handler = logging.StreamHandler()

    # Create formatters and add it to handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def main():
    """
    Main function to run the complete RL-based token pruning pipeline.

    Supports both PPO and GRPO algorithms based on config.USE_GRPO.
    """
    # 0. Setup Logger
    logger = setup_logger()
    logger.info("=" * 60)
    logger.info("TPRL: Token Pruning with Reinforcement Learning")
    logger.info("=" * 60)

    # Log configuration
    logger.info(f"Algorithm: {'GRPO' if config.USE_GRPO else 'PPO'}")
    logger.info(f"Training patches: {config.TRAIN_NUM_PATCHES} (image size: {config.TRAIN_IMAGE_SIZE})")
    logger.info(f"Decision steps: {config.NUM_DECISION_STEPS}")
    logger.info(f"Training threshold: {config.TRAIN_THRESHOLD}")
    logger.info(f"Random masking: {config.ENABLE_RANDOM_MASK} (ratio: {config.RANDOM_MASK_RATIO})")

    # 1. Load Data
    logger.info("\n--- 1. Initializing Data Loader ---")
    data_loader = get_data_loader(config)
    logger.info(f"Data loader for '{config.DATASET_NAME}' initialized.")

    # 2. Load MLLM
    logger.info("\n--- 2. Initializing MLLM ---")
    mllm = get_mllm(config)
    logger.info(f"MLLM '{config.MODEL_ID}' initialized.")
    logger.info(f"Feature dimension: {mllm.feature_dim}")

    # 3. Setup RL Environment and Policy
    logger.info("\n--- 3. Setting up RL Environment and Policy ---")
    train_env, num_patches = setup_environments(config, mllm, data_loader)
    logger.info(f"Actual num_patches: {num_patches}")

    if config.USE_GRPO:
        # Use GRPO algorithm
        from utils.grpo import create_grpo_policy
        policy = create_grpo_policy(config, mllm)
        logger.info("GRPO policy initialized (no value network).")
    else:
        # Use PPO algorithm
        policy = setup_policy(config, mllm, num_patches)
        logger.info("PPO policy initialized.")

    # 4. Train the Agent
    logger.info("\n--- 4. Starting Agent Training ---")
    trained_policy = train_agent(config, policy, train_env, mllm=mllm, data_loader=data_loader)
    logger.info("Agent training finished.")

    # Save the trained policy
    save_path = os.path.join(config.LOG_DIR, "policy_weights.pth")
    save_policy(trained_policy, save_path)
    logger.info(f"Policy saved to {save_path}")

    # 5. Evaluate the Agent
    logger.info("\n--- 5. Starting Agent Evaluation ---")

    # Evaluate with no pruning (baseline)
    logger.info("\n[Baseline: No Pruning]")
    config.EVAL_MODE = "none"
    evaluate_performance(trained_policy, config, mllm, data_loader, logger)

    # Evaluate with threshold-based pruning
    logger.info("\n[Full Pruning with Threshold]")
    config.EVAL_MODE = "full"
    evaluate_performance(trained_policy, config, mllm, data_loader, logger)

    # Evaluate with budget-based pruning
    logger.info("\n[Budget-based Pruning]")
    config.EVAL_MODE = "budget"
    evaluate_performance(trained_policy, config, mllm, data_loader, logger)

    # Evaluate with multiple thresholds to generate efficiency-accuracy curve
    logger.info("\n[Multi-threshold Evaluation]")
    evaluate_with_different_thresholds(
        trained_policy, config, mllm, data_loader, logger,
        thresholds=[0.1, 0.3, 0.5, 0.7, 0.9]
    )

    logger.info("\n" + "=" * 60)
    logger.info("Pipeline completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
