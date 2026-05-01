"""
Evaluation and inference for the trained SE(3)-Transformer.

Computes per-target and per-property metrics in the original
(unnormalized) target space. Also supports single-sample inference.

Usage:
    python -m model.evaluate
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.amp import autocast
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.config import Config
from model.dataset import PocketDataset, create_dataloader
from model.architecture import PocketSE3Transformer
from model.utils import load_checkpoint


@torch.no_grad()
def run_inference(model, loader, device):
    """Run the model on all batches and collect predictions.

    Returns:
        predictions: (N, 18) numpy array (normalized space)
        targets: (N, 18) numpy array (normalized space)
        pdb_codes: list of str
        pocket_ids: list of str
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_pdb_codes = []
    all_pocket_ids = []

    for batch in tqdm(loader, desc="Inference"):
        positions = batch["positions"].to(device)
        features = batch["features"].to(device)
        batch_idx = batch["batch_idx"].to(device)

        with autocast(device_type="cuda", dtype=torch.float16):
            preds = model(positions, features, batch_idx)

        all_preds.append(preds.cpu().numpy())
        all_targets.append(batch["targets"].numpy())
        all_pdb_codes.extend(batch["pdb_codes"])
        all_pocket_ids.extend(batch["pocket_ids"])

    return (
        np.concatenate(all_preds, axis=0),
        np.concatenate(all_targets, axis=0),
        all_pdb_codes,
        all_pocket_ids,
    )


def denormalize(values, target_mean, target_std):
    """Convert from normalized space back to original units."""
    return values * target_std + target_mean


def compute_metrics(predictions, targets, target_columns, properties):
    """Compute per-target and per-property evaluation metrics.

    Args:
        predictions: (N, 18) in original units
        targets: (N, 18) in original units
        target_columns: list of 18 column names
        properties: list of 9 property names

    Returns:
        dict with:
            per_target: {col_name: {mae, rmse, r2}}
            per_property: {prop_name: {mae, rmse}}  (averaged over mean+std)
            overall: {mae, rmse}
    """
    results = {"per_target": {}, "per_property": {}, "overall": {}}

    errors = predictions - targets

    # Per-target metrics
    for i, col in enumerate(target_columns):
        col_errors = errors[:, i]
        col_preds = predictions[:, i]
        col_targets = targets[:, i]

        mae = np.abs(col_errors).mean()
        rmse = np.sqrt((col_errors ** 2).mean())

        # R² score
        ss_res = (col_errors ** 2).sum()
        ss_tot = ((col_targets - col_targets.mean()) ** 2).sum()
        r2 = 1.0 - ss_res / (ss_tot + 1e-8)

        results["per_target"][col] = {"mae": mae, "rmse": rmse, "r2": r2}

    # Per-property metrics (average of mean and std targets)
    for prop in properties:
        mean_col = f"{prop}_mean"
        std_col = f"{prop}_std"
        mean_idx = target_columns.index(mean_col)
        std_idx = target_columns.index(std_col)

        prop_mae = (
            np.abs(errors[:, mean_idx]).mean() +
            np.abs(errors[:, std_idx]).mean()
        ) / 2
        prop_rmse = (
            np.sqrt((errors[:, mean_idx] ** 2).mean()) +
            np.sqrt((errors[:, std_idx] ** 2).mean())
        ) / 2

        results["per_property"][prop] = {"mae": prop_mae, "rmse": prop_rmse}

    # Overall metrics
    results["overall"]["mae"] = np.abs(errors).mean()
    results["overall"]["rmse"] = np.sqrt((errors ** 2).mean())

    return results


def print_metrics(results, target_columns):
    """Pretty-print evaluation metrics."""
    print()
    print("=" * 70)
    print("EVALUATION RESULTS (original units)")
    print("=" * 70)

    # Per-target table
    print(f"\n{'Target':<25} {'MAE':>10} {'RMSE':>10} {'R²':>10}")
    print("-" * 55)
    for col in target_columns:
        m = results["per_target"][col]
        print(f"{col:<25} {m['mae']:>10.4f} {m['rmse']:>10.4f} {m['r2']:>10.4f}")

    # Per-property table
    print(f"\n{'Property':<25} {'MAE':>10} {'RMSE':>10}")
    print("-" * 45)
    for prop, m in results["per_property"].items():
        print(f"{prop:<25} {m['mae']:>10.4f} {m['rmse']:>10.4f}")

    # Overall
    print(f"\n{'Overall':<25} {results['overall']['mae']:>10.4f} {results['overall']['rmse']:>10.4f}")


def main():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    checkpoint_path = config.checkpoint_dir / "best_model.pt"
    if not checkpoint_path.exists():
        print(f"No checkpoint found at {checkpoint_path}")
        print("Train the model first with: python -m model.train")
        return

    print(f"Loading checkpoint: {checkpoint_path}")
    model = PocketSE3Transformer(config).to(device)
    checkpoint = load_checkpoint(checkpoint_path, model)

    target_mean = checkpoint["target_mean"]
    target_std = checkpoint["target_std"]
    print(f"  Trained for {checkpoint['epoch']} epochs")
    print(f"  Best val loss: {checkpoint['val_loss']:.6f}")

    # Load test dataset
    test_dataset = PocketDataset(
        csv_path=config.test_csv,
        distributions_csv_path=config.pocket_distributions_csv,
        pointcloud_dir=config.pointcloud_dir,
        target_columns=config.target_columns,
        target_mean=target_mean,
        target_std=target_std,
        sample_weights=None,
    )
    test_loader = create_dataloader(
        test_dataset, config.batch_size, shuffle=False, num_workers=config.num_workers,
    )
    print(f"Test set: {len(test_dataset)} samples")

    # Run inference
    print("Running inference...")
    preds_norm, targets_norm, pdb_codes, pocket_ids = run_inference(
        model, test_loader, device,
    )

    # Denormalize
    preds_orig = denormalize(preds_norm, target_mean, target_std)
    targets_orig = denormalize(targets_norm, target_mean, target_std)

    # Compute and print metrics
    results = compute_metrics(
        preds_orig, targets_orig, config.target_columns, list(config.properties),
    )
    print_metrics(results, config.target_columns)

    # Save predictions to CSV
    pred_df = pd.DataFrame(preds_orig, columns=[f"pred_{c}" for c in config.target_columns])
    pred_df["pdb_code"] = pdb_codes
    pred_df["pocket_id"] = pocket_ids
    for i, col in enumerate(config.target_columns):
        pred_df[f"true_{col}"] = targets_orig[:, i]

    pred_path = config.checkpoint_dir / "test_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"\nPredictions saved to: {pred_path}")


if __name__ == "__main__":
    main()
