"""
Dataset and DataLoader for pocket point clouds.

Each sample is one point cloud (.npz file) paired with 18 target values
(mean and std of 9 ligand properties for that pocket).

Batching uses concatenation with batch indices — no padding, no graphs.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class PocketDataset(Dataset):
    """Dataset that loads pocket point clouds and their distribution targets.

    Each item returns:
        positions:  (N, 3) float32 — centroid-centered atom coordinates
        features:   (N, 46) float32 — per-atom chemical features
        targets:    (18,) float32 — normalized distribution targets
        weight:     float32 — cost-sensitive sample weight
        pdb_code:   str — identifier for the point cloud file
        pocket_id:  str — pocket identifier
        n_atoms:    int — number of atoms in this point cloud
    """

    def __init__(
        self,
        csv_path,
        distributions_csv_path,
        pointcloud_dir,
        target_columns,
        target_mean=None,
        target_std=None,
        sample_weights=None,
    ):
        """
        Args:
            csv_path: Path to train.csv, val.csv, or test.csv
            distributions_csv_path: Path to pocket_distributions.csv
            pointcloud_dir: Directory containing {pdb_code}.npz files
            target_columns: List of 18 target column names
            target_mean: (18,) array for normalization (None = no normalization)
            target_std: (18,) array for normalization (None = no normalization)
            sample_weights: (N,) array of per-sample weights (None = uniform)
        """
        self.pointcloud_dir = Path(pointcloud_dir)
        self.target_columns = target_columns

        # Load CSV and merge with distributions to get targets
        entries = pd.read_csv(csv_path)
        distributions = pd.read_csv(distributions_csv_path)

        # Merge on pocket_id to attach target columns
        merged = entries.merge(
            distributions[["pocket_id"] + target_columns],
            on="pocket_id",
            how="left",
        )

        self.pdb_codes = merged["pdb_code"].values
        self.pocket_ids = merged["pocket_id"].values

        # Extract raw targets
        raw_targets = merged[target_columns].values.astype(np.float32)  # (N, 18)

        # Handle NaN targets (pockets with <3 ligands may have NaN std)
        # Replace NaN with 0 — these will still be used for training but the
        # cost-sensitive weighting ensures they don't dominate
        nan_mask = np.isnan(raw_targets)
        if nan_mask.any():
            n_nan = nan_mask.sum()
            print(f"Warning: {n_nan} NaN target values found, replacing with 0")
            raw_targets = np.nan_to_num(raw_targets, nan=0.0)

        # Normalize targets
        if target_mean is not None and target_std is not None:
            self.target_mean = target_mean
            self.target_std = target_std
            self.targets = (raw_targets - target_mean) / target_std
        else:
            self.target_mean = None
            self.target_std = None
            self.targets = raw_targets

        self.targets = self.targets.astype(np.float32)

        # Sample weights
        if sample_weights is not None:
            self.weights = sample_weights.astype(np.float32)
        else:
            # Uniform weights
            self.weights = np.ones(len(self.pdb_codes), dtype=np.float32)

    def __len__(self):
        return len(self.pdb_codes)

    def __getitem__(self, idx):
        pdb_code = self.pdb_codes[idx]

        # Load point cloud from disk
        npz_path = self.pointcloud_dir / f"{pdb_code}.npz"
        data = np.load(npz_path)
        positions = data["positions"]   # (N, 3) float32
        features = data["features"]     # (N, 46) float32

        return {
            "positions": torch.from_numpy(positions),
            "features": torch.from_numpy(features),
            "targets": torch.from_numpy(self.targets[idx]),
            "weight": torch.tensor(self.weights[idx], dtype=torch.float32),
            "pdb_code": pdb_code,
            "pocket_id": self.pocket_ids[idx],
            "n_atoms": positions.shape[0],
        }


def collate_fn(batch_list):
    """Concatenation batching: stack all atoms into one tensor with batch indices.

    Instead of padding to the max size, we concatenate all point clouds and
    track which atoms belong to which sample via a batch_idx tensor.

    Args:
        batch_list: List of dicts from PocketDataset.__getitem__

    Returns:
        dict with:
            positions:  (total_atoms, 3) float32
            features:   (total_atoms, 46) float32
            batch_idx:  (total_atoms,) long — sample index for each atom
            targets:    (B, 18) float32
            weights:    (B,) float32
            pdb_codes:  list of str
            pocket_ids: list of str
            n_atoms:    list of int
    """
    all_positions = []
    all_features = []
    all_batch_idx = []
    all_targets = []
    all_weights = []
    pdb_codes = []
    pocket_ids = []
    n_atoms_list = []

    for i, sample in enumerate(batch_list):
        n = sample["n_atoms"]
        all_positions.append(sample["positions"])
        all_features.append(sample["features"])
        all_batch_idx.append(torch.full((n,), i, dtype=torch.long))
        all_targets.append(sample["targets"])
        all_weights.append(sample["weight"])
        pdb_codes.append(sample["pdb_code"])
        pocket_ids.append(sample["pocket_id"])
        n_atoms_list.append(n)

    return {
        "positions": torch.cat(all_positions, dim=0),      # (total_atoms, 3)
        "features": torch.cat(all_features, dim=0),        # (total_atoms, 46)
        "batch_idx": torch.cat(all_batch_idx, dim=0),      # (total_atoms,)
        "targets": torch.stack(all_targets, dim=0),         # (B, 18)
        "weights": torch.stack(all_weights, dim=0),         # (B,)
        "pdb_codes": pdb_codes,
        "pocket_ids": pocket_ids,
        "n_atoms": n_atoms_list,
    }


def create_dataloader(dataset, batch_size, shuffle=True, num_workers=4):
    """Create a DataLoader with our custom collate function."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
