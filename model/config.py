"""
Central configuration for the SE(3)-Transformer model.

All hyperparameters live here — no magic numbers scattered in other files.
"""

import dataclasses
from pathlib import Path


@dataclasses.dataclass
class Config:
    # ---- Paths ----
    data_dir: Path = Path(__file__).resolve().parent.parent / "data"
    pointcloud_dir: Path = None  # set in __post_init__
    train_csv: Path = None
    val_csv: Path = None
    test_csv: Path = None
    pocket_distributions_csv: Path = None
    checkpoint_dir: Path = None

    # ---- Target properties ----
    properties: tuple = (
        "MW", "logP", "HBD", "HBA", "TPSA",
        "rotatable_bonds", "formal_charge", "aromatic_rings", "fsp3",
    )
    stats: tuple = ("mean", "std")
    num_targets: int = 18  # len(properties) * len(stats)

    # ---- Input ----
    input_features: int = 46

    # ---- Model architecture ----
    hidden_scalars: int = 32       # multiplicity for l=0 (scalar) channels
    hidden_vectors: int = 16       # multiplicity for l=1 (vector) channels
    num_layers: int = 4
    num_heads: int = 4
    lmax: int = 1
    radius_cutoff: float = 10.0    # Angstroms — tunable
    num_radial_basis: int = 32
    radial_mlp_hidden: int = 64

    # ---- Pooling + prediction head ----
    # After pooling: hidden_scalars + hidden_vectors (vector norms) = 48
    head_hidden_1: int = 96
    head_hidden_2: int = 48
    head_dropout: float = 0.3

    # ---- Training ----
    batch_size: int = 4
    num_epochs: int = 200
    learning_rate: float = 1e-4
    weight_decay: float = 1e-3
    lr_patience: int = 10
    lr_factor: float = 0.5
    early_stop_patience: int = 25
    grad_clip_norm: float = 10.0
    num_workers: int = 4
    seed: int = 42

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        if self.pointcloud_dir is None:
            self.pointcloud_dir = self.data_dir / "pocket_pointclouds"
        if self.train_csv is None:
            self.train_csv = self.data_dir / "train.csv"
        if self.val_csv is None:
            self.val_csv = self.data_dir / "val.csv"
        if self.test_csv is None:
            self.test_csv = self.data_dir / "test.csv"
        if self.pocket_distributions_csv is None:
            self.pocket_distributions_csv = self.data_dir / "pocket_distributions.csv"
        if self.checkpoint_dir is None:
            self.checkpoint_dir = Path(__file__).resolve().parent / "checkpoints"

    @property
    def target_columns(self):
        """List of 18 target column names in pocket_distributions.csv."""
        return [f"{prop}_{stat}" for prop in self.properties for stat in self.stats]

    @property
    def hidden_irreps_str(self):
        """e3nn irreps string for hidden features."""
        return f"{self.hidden_scalars}x0e + {self.hidden_vectors}x1o"

    @property
    def pooled_dim(self):
        """Dimensionality after invariant pooling."""
        return self.hidden_scalars + self.hidden_vectors
