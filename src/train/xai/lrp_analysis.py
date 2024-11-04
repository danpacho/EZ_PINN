from typing import Literal

import torch
from torch.utils.data import DataLoader

from src.train.xai.lrp import LRP


def lrp_analysis(
    lrp: LRP,
    data_loader: DataLoader,
    feature_count: int,
    analysis_type: Literal["epsilon", "z_beta", "combine"] = "combine",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform LRP analysis on the given model and data loader.

    Args:
        lrp (LRP): LRP instance.
        data_loader (DataLoader): Data loader.
        feature_count (int): Number of features.
        analysis_type (Literal[&quot;epsilon&quot;, &quot;z_beta&quot;, &quot;combine&quot;], optional): Backpropagation type. Defaults to "combine".

    Raises:
        ValueError: Invalid analysis type.

    Returns:
        tuple: `[Rank count vector, Rank sum vector]`
    """
    rank_count_vector = torch.zeros(feature_count, dtype=torch.int32)
    rank_sum_vector = torch.zeros(feature_count, dtype=torch.float32)

    for batch_inputs, _ in data_loader:
        batch_inputs = batch_inputs.to(lrp.device)
        lrp.forward(batch_inputs)
        if analysis_type == "epsilon":
            lrp.back_propagation_epsilon()
        elif analysis_type == "z_beta":
            lrp.back_propagation_z_beta()
        elif analysis_type == "combine":
            lrp.back_propagation()
        else:
            raise ValueError("Invalid analysis type.")

        relevance_scores = lrp.relevance_score[0]  # Shape: [batch_size, feature_count]

        # Sum over batches
        # Summing the absolute relevance scores for better interpretability
        batch_ranks = torch.argsort(relevance_scores, dim=1, descending=True)
        batch_rank_counts = torch.zeros_like(relevance_scores[0], dtype=torch.int32)

        # Count the ranks for each feature
        for ranks in batch_ranks:
            for idx, feature_idx in enumerate(ranks):
                batch_rank_counts[feature_idx] += idx

        rank_count_vector += batch_rank_counts.cpu()
        rank_sum_vector += relevance_scores.sum(dim=0).cpu()

    return rank_count_vector, rank_sum_vector
