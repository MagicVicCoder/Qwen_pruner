import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_pruner import BasePruner
import numpy as np

class DivPruner(BasePruner):
    """
    Pruner based on diversity maximization (DivPrune concept).
    It aims to select a subset of visual tokens that are both important and diverse.
    This implementation uses a simplified approach: select top-k important tokens,
    then iteratively add diverse tokens based on feature distance.
    """
    def _build_model(self):
        # DivPruner doesn't require a learnable neural network model.
        # We might need parameters like alpha (importance vs diversity trade-off),
        # which are typically in config.
        print("DivPruner initialized (no model to build).")

    def calculate_pruning_scores(self, visual_features, query_embeddings):
        """
        Calculate initial importance scores for visual features based on their relevance to the query.
        This is often done using a simple metric like cosine similarity or a learned projection.
        Here, we use the cosine similarity between visual features and the query embedding.

        Args:
            visual_features (torch.Tensor): Shape [B, N, hidden_dim]
            query_embeddings (torch.Tensor): Shape [B, 1, hidden_dim]

        Returns:
            torch.Tensor: Importance scores for each patch. Higher score means more important.
                          Shape: [B, N]
        """
        # Normalize features for cosine similarity
        visual_norm = F.normalize(visual_features, p=2, dim=-1) # [B, N, hidden_dim]
        query_norm = F.normalize(query_embeddings, p=2, dim=-1) # [B, 1, hidden_dim]

        # Calculate cosine similarity scores [B, N, 1] -> [B, N]
        importance_scores = torch.sum(visual_norm * query_norm, dim=-1) # [B, N]
        return importance_scores

    def prune_tokens(self, visual_features, query_embeddings, target_ratio):
        """
        Prune visual tokens based on importance and diversity.

        Args:
            visual_features (torch.Tensor): The original visual features.
                                            Shape: [batch_size, num_patches, hidden_dim]
            query_embeddings (torch.Tensor): The query embeddings.
                                           Shape: [batch_size, 1, hidden_dim]
            target_ratio (float): The ratio of tokens to keep (e.g., 0.5 for 50%).

        Returns:
            torch.Tensor: The pruned visual features.
                          Shape: [batch_size, num_patches_kept, hidden_dim]
        """
        batch_size, num_patches, hidden_dim = visual_features.shape
        num_to_keep = int(num_patches * target_ratio)

        if num_to_keep >= num_patches or num_to_keep <= 0:
            if num_to_keep >= num_patches:
                print("Warning: target_ratio >= 1.0, returning all features.")
                return visual_features
            else:
                print("Warning: target_ratio <= 0.0, returning empty features. This is likely an error.")
                return visual_features[:, :0, :] # Return empty sequence

        # Step 1: Calculate initial importance scores
        importance_scores = self.calculate_pruning_scores(visual_features, query_embeddings) # [B, N]

        # Step 2: Select top-k most important tokens
        _, top_k_indices = torch.topk(importance_scores, num_to_keep, dim=1, largest=True, sorted=False) # [B, num_to_keep]

        # --- DivPrune Logic: Iteratively add diverse tokens ---
        # For simplicity, we'll implement a greedy diversity selection *after* selecting top-k important ones.
        # A more sophisticated DivPrune might interleave importance and diversity selection.
        # Here, we start with top-k important and then potentially swap/add diverse ones.

        # Get the features and scores of the top-k important tokens
        batch_indices = torch.arange(batch_size, device=self.device).unsqueeze(1).expand(-1, num_to_keep) # [B, num_to_keep]
        top_k_features = visual_features[batch_indices, top_k_indices] # [B, num_to_keep, hidden_dim]
        top_k_scores = importance_scores[batch_indices, top_k_indices] # [B, num_to_keep]

        # Get the features of *all* tokens for diversity calculation
        all_features_normalized = F.normalize(visual_features, p=2, dim=-1) # [B, N, hidden_dim]

        pruned_features_list = []
        for b in range(batch_size):
            current_top_features = top_k_features[b] # [num_to_keep, hidden_dim]
            current_top_indices = top_k_indices[b] # [num_to_keep]
            current_all_features_norm = all_features_normalized[b] # [N, hidden_dim]
            current_top_features_norm = F.normalize(current_top_features, p=2, dim=-1) # [num_to_keep, hidden_dim]

            # Calculate pairwise cosine similarities within the selected top-k features [num_to_keep, num_to_keep]
            intra_sim = torch.mm(current_top_features_norm, current_top_features_norm.t()) # [num_to_keep, num_to_keep]
            # Diagonal is 1.0, subtract it to ignore self-similarity if needed, but for max calculation it's fine.
            max_intra_sim, _ = torch.max(intra_sim, dim=1) # [num_to_keep] - max similarity of each selected token to *any other* selected token
            # A lower max_intra_sim indicates the token is less similar to others in the set (more diverse)

            # Calculate similarity of each selected token to the query (importance)
            selected_to_query_sim = torch.sum(current_top_features_norm * query_embeddings[b], dim=-1) # [num_to_keep]

            # Define a simple score combining importance and diversity (this is a heuristic)
            # alpha controls the trade-off: score = alpha * importance + (1-alpha) * diversity
            # Here, diversity is approximated as (1 - max_intra_sim)
            alpha = getattr(self.config, 'DIV_PRUNE_IMPORTANCE_WEIGHT', 0.5) # Use config or default
            combined_scores = alpha * selected_to_query_sim + (1 - alpha) * (1 - max_intra_sim) # [num_to_keep]

            # For DivPrune, a more common approach is to iteratively select the most diverse token
            # from the *remaining* pool that is not yet selected.
            # Let's implement a more standard greedy max-diversity selection starting from the most important one.

            # Start with the single most important token
            selected_indices_set = set()
            _, most_important_idx_in_top_k = torch.topk(top_k_scores[b], 1, largest=True, sorted=False)
            initial_selected_global_idx = current_top_indices[most_important_idx_in_top_k.item()].item()
            selected_indices_set.add(initial_selected_global_idx)

            # Iteratively select diverse tokens
            for _ in range(num_to_keep - 1):
                max_min_sim_to_selected = -2.0 # Cosine sim range is [-1, 1]
                best_candidate_idx = -1

                for candidate_idx in range(num_patches):
                    if candidate_idx in selected_indices_set:
                        continue

                    candidate_feature_norm = current_all_features_norm[candidate_idx:candidate_idx+1] # [1, hidden_dim]

                    # Calculate similarity from this candidate to *all currently selected* tokens
                    sim_to_selected = torch.mm(candidate_feature_norm, current_top_features_norm.t()) # [1, len(selected)]
                    min_sim_to_selected = torch.min(sim_to_selected).item() # Find the *minimum* similarity (most diverse)

                    if min_sim_to_selected > max_min_sim_to_selected:
                        max_min_sim_to_selected = min_sim_to_selected
                        best_candidate_idx = candidate_idx

                if best_candidate_idx != -1:
                    selected_indices_set.add(best_candidate_idx)
                else:
                    # This should ideally not happen if num_to_keep <= num_patches
                    print(f"Warning: Could not find enough diverse tokens for batch {b}. Stopping early.")
                    break

            # Convert set to sorted list to maintain consistent order (optional, for reproducibility if needed)
            final_selected_indices = sorted(list(selected_indices_set))
            final_selected_tensor = torch.tensor(final_selected_indices, dtype=torch.long, device=self.device)

            # Gather the final pruned features for this batch item
            pruned_features_list.append(visual_features[b:b+1, final_selected_tensor, :]) # Add batch dim back [1, len(indices), hidden_dim]

        # Concatenate results for all batches
        pruned_features = torch.cat(pruned_features_list, dim=0) # [B, num_to_keep, hidden_dim]
        return pruned_features

    # The 'forward' method from BasePruner can be inherited, as it calls 'prune_tokens'.
