"""
Target normalization and cost-sensitive sample weight computation.

Normalization stats are computed from UNIQUE pockets in the training set
(not per-sample) to avoid bias from proteins with many crystal structures.
"""

import numpy as np
import pandas as pd


def compute_target_stats(train_csv_path, distributions_csv_path, target_columns):
    """Compute mean and std of each target from unique pockets in the training set.

    Uses unique pocket_ids so that a pocket appearing in 50 crystal structures
    doesn't count 50 times toward the normalization statistics.

    Args:
        train_csv_path: Path to train.csv
        distributions_csv_path: Path to pocket_distributions.csv
        target_columns: List of 18 column names (e.g. ["MW_mean", "MW_std", ...])

    Returns:
        (target_mean, target_std): Each a numpy array of shape (18,).
    """
    train = pd.read_csv(train_csv_path)
    distributions = pd.read_csv(distributions_csv_path)

    # Get unique pocket_ids from training set
    train_pocket_ids = train["pocket_id"].unique()

    # Filter distributions to training pockets only
    train_dists = distributions[distributions["pocket_id"].isin(train_pocket_ids)]

    # Extract target values
    target_values = train_dists[target_columns].values  # (n_pockets, 18)

    target_mean = np.nanmean(target_values, axis=0).astype(np.float32)
    target_std = np.nanstd(target_values, axis=0).astype(np.float32)

    # Prevent division by zero — if a target has zero std, set to 1.0
    target_std[target_std < 1e-8] = 1.0

    return target_mean, target_std


def compute_sample_weights(train_csv_path):
    """Compute cost-sensitive weights inversely proportional to protein frequency.

    Each sample's weight = N_total / (N_proteins * count_of_its_protein).
    This ensures each protein contributes equally to the total loss regardless
    of how many crystal structures it has.

    Weights sum to N_total (preserving loss magnitude).

    Args:
        train_csv_path: Path to train.csv

    Returns:
        weights: numpy array of shape (N_samples,), same order as rows in train.csv.
    """
    train = pd.read_csv(train_csv_path)

    # Count how many samples each protein has
    protein_counts = train["uniprot_id"].value_counts()

    n_total = len(train)
    n_proteins = train["uniprot_id"].nunique()

    # Assign weight to each sample
    weights = train["uniprot_id"].map(
        lambda uid: n_total / (n_proteins * protein_counts[uid])
    ).values.astype(np.float32)

    return weights
