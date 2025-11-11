import torch
from tqdm import tqdm
import logging
import numpy as np
from trainer.trainer import setup_pruner

def evaluate_performance(pruner, config, mllm, data_loader, logger):
    """
    Evaluates the performance of the pruner.
    Assumes samples from data_loader contain 'answer' key for accuracy calculation.
    """
    test_samples = data_loader.get_test_samples()
    eval_batch_size = config.EVAL_BATCH_SIZE
    target_ratio = config.PRUNING_TARGET_RATIO

    logger.info(f"Starting evaluation in mode: {config.EVAL_MODE}")

    # Determine number of samples based on eval mode
    if config.EVAL_MODE == "full":
        num_samples_to_eval = len(test_samples)
    elif config.EVAL_MODE == "budget":
        # Example: Evaluate on a smaller, fixed budget
        num_samples_to_eval = min(100, len(test_samples))
    elif config.EVAL_MODE == "none": # Add condition for no pruning
        num_samples_to_eval = len(test_samples)
    else: # "none" (or any other unrecognized mode handled here)
        logger.info(f"Unrecognized EVAL_MODE: {config.EVAL_MODE}. Skipping evaluation.")
        return

    eval_samples = test_samples[:num_samples_to_eval]
    logger.info(f"Evaluating on {len(eval_samples)} samples.")

    # --- 用于计算准确率和压缩率的列表 ---
    accuracies = []
    # Compression ratio is 1.0 (no compression) when EVAL_MODE is "none"
    compression_ratios = []

    # --- 计数器，用于调试 ---
    total_processed = 0
    match_found = 0
    no_match_found = 0

    for i in tqdm(range(0, len(eval_samples), eval_batch_size), desc="Evaluating"):
        batch_samples = eval_samples[i:i+eval_batch_size]

        for sample in batch_samples:
            try:
                # Assuming sample has 'image', 'question', and 'answer' keys
                image = sample['image']
                question = sample['question']
                gt_answer = sample['answer'] # 获取 Ground Truth 答案

                # Get components from the MLLM
                components = mllm.get_components_for_env(image, question)
                if components is None:
                    logger.warning(f"Skipping sample due to processing error.")
                    continue # 跳过此样本，不计入统计

                original_visual_features = components["original_visual_features"]
                text_embeds_part1 = components["text_embeds_part1"]
                text_embeds_part2 = components["text_embeds_part2"]
                query_embeddings = components["query_embeddings"]
                current_num_patches = components["current_num_patches"]

                # --- Prune the visual features (or not) based on EVAL_MODE ---
                if config.EVAL_MODE == "none":
                    # No pruning: use original visual features
                    final_visual_features = original_visual_features
                    compression_ratio = 1.0 # No compression
                    logger.debug(f"No pruning applied. Using all {current_num_patches} patches.")
                else:
                    # Apply pruning using the provided pruner
                    pruned_visual_features = pruner.forward(original_visual_features, query_embeddings, target_ratio)
                    logger.debug(f"Pruned from {original_visual_features.shape[1]} to {pruned_visual_features.shape[1]} patches.")
                    final_visual_features = pruned_visual_features
                    compression_ratio = pruned_visual_features.shape[1] / current_num_patches

                # --- Combine visual features (original or pruned) with text embeddings ---
                combined_embeddings = torch.cat([
                    text_embeds_part1,
                    final_visual_features, # Use final_visual_features here
                    text_embeds_part2
                ], dim=1)
                attention_mask = torch.ones((1, combined_embeddings.shape[1]), dtype=torch.long, device=mllm.device)

                # --- Generate answer using the MLLM with (potentially pruned) embeddings ---
                generated_answer = mllm.generate_answer(combined_embeddings, attention_mask)

                # --- Evaluation Logic (Accuracy Calculation) ---
                # --- 添加调试日志 ---
                logger.debug(f"--- Sample {total_processed} ---")
                logger.debug(f"Question: {question}")
                logger.debug(f"Generated Answer Raw: '{generated_answer}'")
                logger.debug(f"Ground Truth Answer Raw: '{gt_answer}'")
                logger.debug(f"Generated Lower: '{generated_answer.lower()}'")
                logger.debug(f"GT Lower: '{gt_answer.lower()}'")
                # ---

                # 计算准确率
                is_match = gt_answer.lower() in generated_answer.lower()
                logger.debug(f"Is GT in Generated? {is_match}")
                accuracy = 1.0 if is_match else 0.0
                logger.debug(f"Calculated Accuracy for this sample: {accuracy}\n")
                # ---

                accuracies.append(accuracy) # 将本次样本的准确率加入列表
                compression_ratios.append(compression_ratio) # 将本次样本的压缩率加入列表

                # 计数调试 ---
                total_processed += 1
                if is_match:
                    match_found += 1
                else:
                    no_match_found += 1
                # ---

            except Exception as e:
                logger.error(f"Error processing sample: {e}")
                # 可以选择跳过错误样本或计入失败，这里选择跳过
                continue

    # --- 计算并打印最终的平均结果 ---
    if accuracies: # 检查是否有成功评估的样本
        avg_accuracy = np.mean(accuracies)
        avg_compression_ratio = np.mean(compression_ratios)

        # --- 打印调试计数 ---
        logger.info(f"--- Debug Counts for {config.EVAL_MODE} mode ---")
        logger.info(f"Total Samples Processed: {total_processed}")
        logger.info(f"Matches Found (GT in Generated): {match_found}")
        logger.info(f"No Matches Found: {no_match_found}")
        logger.info(f"---------------------")

        logger.info(f"Evaluation Mode: {config.EVAL_MODE}")
        logger.info(f"Total Samples Attempted: {len(eval_samples)}")
        logger.info(f"Successfully Evaluated Samples: {len(accuracies)}")
        logger.info(f"Average Accuracy: {avg_accuracy:.4f}")
        logger.info(f"Average Compression Ratio: {avg_compression_ratio:.4f}")
        logger.info("--------------------------")
    else:
        logger.warning(f"No samples were successfully evaluated for {config.EVAL_MODE} mode.")
        logger.info("--------------------------")
