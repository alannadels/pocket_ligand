"""
Cost-sensitive MSE loss.

Weights each sample inversely by its protein's frequency in the training set,
so data-rich proteins (e.g., HIV protease with 100+ structures) don't dominate
over proteins with only 3-5 structures.
"""

import torch
import torch.nn as nn


class CostSensitiveMSELoss(nn.Module):
    """Weighted MSE loss where each sample's contribution is scaled by its weight.

    loss = sum(weights * per_sample_mse) / sum(weights)

    where per_sample_mse = mean over 18 targets of (pred - target)^2
    """

    def forward(self, predictions, targets, weights):
        """
        Args:
            predictions: (B, 18) predicted values
            targets:     (B, 18) ground truth values
            weights:     (B,) per-sample cost-sensitive weights

        Returns:
            Scalar loss value.
        """
        # Per-sample MSE: average squared error across the 18 targets
        per_sample_mse = ((predictions - targets) ** 2).mean(dim=1)  # (B,)

        # Weighted average across the batch
        loss = (weights * per_sample_mse).sum() / weights.sum()

        return loss
