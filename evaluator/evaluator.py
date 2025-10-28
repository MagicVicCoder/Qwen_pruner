import torch
from tqdm import tqdm
import logging
from ..trainer.trainer import setup_pruner

def evaluate_performance(policy_or_pruner, config, mllm, data_loader, logger):
    """
    Evaluates the performance of the pruner.
    The 'policy_or_pruner' argument is expected to be the pruner object now.
    """
    pruner = policy_or_pruner # Rename for clarity
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
    else: # "none"
        logger.info("Evaluation mode is 'none', skipping evaluation.")
        return

    eval_samples = test_samples[:num_samples_to_eval]
    logger.info(f"Evaluating on {len(eval_samples)} samples.")

    correct_count = 0
    total_count = 0

    # Example evaluation: Compare answers from full model vs. pruned model
    # This requires ground truth answers in the dataset, which MME might not have directly.
    # We'll assume a simple comparison or use a proxy metric if ground truth isn't available.
    # For demonstration, let's assume we are just checking if the model runs without error
    # and perhaps measure the length of the generated answer as a very rough proxy.
    # A more robust evaluation would require a dataset with answers or a more complex metric.

    for i in tqdm(range(0, len(eval_samples), eval_batch_size), desc="Evaluating"):
        batch_samples = eval_samples[i:i+eval_batch_size]

        for sample in batch_samples:
            try:
                # Assuming sample has 'image' and 'question' keys
                image = sample['image']
                question = sample['question']

                # Get components from the MLLM
                components = mllm.get_components_for_env(image, question)
                if components is None:
                    logger.warning(f"Skipping sample {total_count} due to processing error.")
                    continue

                original_visual_features = components["original_visual_features"]
                text_embeds_part1 = components["text_embeds_part1"]
                text_embeds_part2 = components["text_embeds_part2"]
                query_embeddings = components["query_embeddings"]
                # current_num_patches = components["current_num_patches"] # Not directly used here

                # --- Prune the visual features ---
                pruned_visual_features = pruner.forward(original_visual_features, query_embeddings, target_ratio)
                logger.debug(f"Pruned from {original_visual_features.shape[1]} to {pruned_visual_features.shape[1]} patches.")

                # --- Combine pruned visual features with text embeddings ---
                # Concatenate: [text_embeds_part1, pruned_visual_features, text_embeds_part2]
                combined_embeddings = torch.cat([
                    text_embeds_part1,
                    pruned_visual_features,
                    text_embeds_part2
                ], dim=1) # Shape: [1, new_seq_len, hidden_dim]

                # Create attention mask for the combined embeddings
                attention_mask = torch.ones((1, combined_embeddings.shape[1]), dtype=torch.long, device=mllm.device)

                # --- Generate answer using the MLLM with pruned embeddings ---
                generated_answer = mllm.generate_answer(combined_embeddings, attention_mask)

                # --- Evaluation Logic ---
                # Since MME doesn't have ground truth answers in its standard form,
                # we'll just log the generated answer or perform a simple check.
                # Replace this section with your specific evaluation metric.
                logger.debug(f"Question: {question}")
                logger.debug(f"Generated Answer: {generated_answer}")
                # Example placeholder evaluation: check if answer is not empty
                if generated_answer.strip():
                    correct_count += 1 # Count as 'successful' generation
                total_count += 1

            except Exception as e:
                logger.error(f"Error processing sample {total_count}: {e}")
                total_count += 1 # Count as failed but continue
                continue

    if total_count > 0:
        success_rate = correct_count / total_count
        logger.info(f"Evaluation Mode: {config.EVAL_MODE}")
        logger.info(f"Total Samples Evaluated: {total_count}")
        logger.info(f"Successful Generations (Non-empty): {correct_count}")
        logger.info(f"Success Rate (Proxy): {success_rate:.4f}")
    else:
        logger.warning("No samples were successfully evaluated.")