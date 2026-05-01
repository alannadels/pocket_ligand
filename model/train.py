"""
Step 7: Train the SE(3)-Transformer.

Main entry point for model training. Handles:
    - Data loading with concatenation batching
    - Target normalization (zero mean, unit variance from training set)
    - Cost-sensitive loss (inverse protein frequency weighting)
    - Learning rate scheduling (ReduceLROnPlateau)
    - Early stopping
    - Checkpointing (best model saved)
    - Training metrics logging

Usage:
    python -m model.train

    Or from the project root:
    python model/train.py

Dependencies:
    pip install torch e3nn torch-scatter torch-cluster pandas numpy
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# Add project root to path so imports work when run as script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.config import Config
from model.dataset import PocketDataset, create_dataloader
from model.architecture import PocketSE3Transformer
from model.loss import CostSensitiveMSELoss
from model.normalization import compute_target_stats, compute_sample_weights
from model.utils import seed_everything, save_checkpoint, load_checkpoint, TrainingLogger


def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip_norm, scaler, epoch):
    """Run one training epoch with mixed precision.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    total_samples = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)
    for batch in pbar:
        positions = batch["positions"].to(device)
        features = batch["features"].to(device)
        batch_idx = batch["batch_idx"].to(device)
        targets = batch["targets"].to(device)
        weights = batch["weights"].to(device)

        optimizer.zero_grad()

        with autocast(device_type="cuda", dtype=torch.float16):
            predictions = model(positions, features, batch_idx)
            loss = criterion(predictions, targets, weights)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        batch_size = targets.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / total_samples


@torch.no_grad()
def validate(model, loader, device, epoch):
    """Run validation with uniform (non-cost-sensitive) MSE.

    Validation uses uniform weights so we evaluate all samples equally,
    regardless of protein frequency.

    Returns:
        Average validation MSE loss.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [val]  ", leave=False)
    for batch in pbar:
        positions = batch["positions"].to(device)
        features = batch["features"].to(device)
        batch_idx = batch["batch_idx"].to(device)
        targets = batch["targets"].to(device)

        with autocast(device_type="cuda", dtype=torch.float16):
            predictions = model(positions, features, batch_idx)
            loss = F.mse_loss(predictions, targets)

        batch_size = targets.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / total_samples


def main():
    config = Config()
    print(f"Configuration:")
    print(f"  Data dir:        {config.data_dir}")
    print(f"  Radius cutoff:   {config.radius_cutoff} A")
    print(f"  Hidden irreps:   {config.hidden_irreps_str}")
    print(f"  Layers:          {config.num_layers}")
    print(f"  Heads:           {config.num_heads}")
    print(f"  Batch size:      {config.batch_size}")
    print(f"  Learning rate:   {config.learning_rate}")
    print(f"  Targets:         {config.num_targets}")
    print()

    # ---- Seed ----
    seed_everything(config.seed)
    print(f"Random seed: {config.seed}")

    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # ---- Target normalization ----
    print("Computing target normalization stats from training set...")
    target_mean, target_std = compute_target_stats(
        config.train_csv, config.pocket_distributions_csv, config.target_columns
    )
    print(f"  Target columns: {config.target_columns}")
    print(f"  Mean range: [{target_mean.min():.3f}, {target_mean.max():.3f}]")
    print(f"  Std range:  [{target_std.min():.3f}, {target_std.max():.3f}]")
    print()

    # ---- Sample weights ----
    print("Computing cost-sensitive sample weights...")
    sample_weights = compute_sample_weights(config.train_csv)
    print(f"  Weight range: [{sample_weights.min():.4f}, {sample_weights.max():.4f}]")
    print(f"  Weight sum: {sample_weights.sum():.1f} (should ≈ {len(sample_weights)})")
    print()

    # ---- Datasets ----
    print("Loading datasets...")
    train_dataset = PocketDataset(
        csv_path=config.train_csv,
        distributions_csv_path=config.pocket_distributions_csv,
        pointcloud_dir=config.pointcloud_dir,
        target_columns=config.target_columns,
        target_mean=target_mean,
        target_std=target_std,
        sample_weights=sample_weights,
    )
    val_dataset = PocketDataset(
        csv_path=config.val_csv,
        distributions_csv_path=config.pocket_distributions_csv,
        pointcloud_dir=config.pointcloud_dir,
        target_columns=config.target_columns,
        target_mean=target_mean,  # Use training stats for validation too
        target_std=target_std,
        sample_weights=None,  # Uniform weights for validation
    )
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print()

    # ---- DataLoaders ----
    train_loader = create_dataloader(
        train_dataset, config.batch_size, shuffle=True, num_workers=config.num_workers,
    )
    val_loader = create_dataloader(
        val_dataset, config.batch_size, shuffle=False, num_workers=config.num_workers,
    )

    # ---- Model ----
    print("Building model...")
    model = PocketSE3Transformer(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")
    print()

    # ---- Optimizer, scheduler, loss ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.lr_factor,
        patience=config.lr_patience,
    )
    criterion = CostSensitiveMSELoss()
    scaler = GradScaler()

    # ---- Logger ----
    logger = TrainingLogger(config.checkpoint_dir)
    best_checkpoint_path = config.checkpoint_dir / "se3_pocket_ligand_best.pt"

    # ---- Training loop ----
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, config.num_epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, config.grad_clip_norm,
            scaler, epoch,
        )

        # Validate
        val_loss = validate(model, val_loader, device, epoch)

        epoch_time = time.time() - epoch_start

        # Learning rate scheduling
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        # Logging
        logger.log_epoch(epoch, train_loss, val_loss, current_lr, best_val_loss)
        print(f"  Epoch {epoch}: train={train_loss:.6f} val={val_loss:.6f} "
              f"lr={current_lr:.2e} time={epoch_time:.1f}s")

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss, config,
                target_mean, target_std, best_checkpoint_path,
            )
            logger.log_message(f"New best model saved (val_loss={val_loss:.6f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config.early_stop_patience:
            logger.log_message(
                f"Early stopping at epoch {epoch} "
                f"(no improvement for {config.early_stop_patience} epochs)"
            )
            break

    # ---- Done ----
    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best validation loss: {best_val_loss:.6f}")
    print(f"  Checkpoint saved at: {best_checkpoint_path}")
    print(f"  Training log: {logger.csv_path}")


if __name__ == "__main__":
    main()
