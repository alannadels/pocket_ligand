# How to Run — SE(3)-Transformer (10Å pocket radius)

Train the pocket-to-ligand-property-distribution model on an A100 GPU.

## Hardware

- **Required**: NVIDIA A100 (40 GB or 80 GB).
- **Not sufficient**: A10G / V100 / smaller — the 10Å radius graph generates ~8× more edges than the 5Å baseline and the radial MLP activations OOM at every batch size we tested on a 22 GB A10G.

## Default config (already set in `model/config.py`)

| Knob | Value | Notes |
|---|---|---|
| `radius_cutoff` | 10.0 Å | up from 5.0 Å — captures extended pocket walls |
| `num_radial_basis` | 32 | scaled with the radius (~0.32 Å per Gaussian) |
| `batch_size` | 4 | safe on A100 40 GB. On A100 80 GB you can try 8 |
| `num_layers` | 4 | unchanged |
| `hidden_irreps` | 32x0e + 16x1o | unchanged |
| `num_epochs` | 200 | early stop patience 25 |
| `learning_rate` | 1e-4 | unchanged |

## Environment

Python 3.10. Core packages (versions verified to work):

```
torch==2.5.1+cu121
torch_cluster==1.6.3+pt25cu121
torch_scatter==2.1.2+pt25cu121
e3nn==0.6.0
numpy==2.2.6
pandas==2.3.3
tqdm
```

Install (CUDA 12.1 wheels):

```
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install torch-cluster torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
pip install e3nn==0.6.0 numpy pandas tqdm
```

## Data prerequisites

The following must be present under `data/` before training:

- `data/pocket_pointclouds/` — one `.npz` per pocket-ligand complex (Step 5 output).
- `data/train.csv`, `data/val.csv`, `data/test.csv` — split manifests (Step 6 output).
- `data/pocket_distributions.csv` — 18 target columns per pocket (Step 4 output).

If any are missing, regenerate them with the scripts in `scripts/` (`compute_pocket_pointclouds.py`, `create_splits.py`, `compute_pocket_distributions.py`).

## Train

From the project root, inside a `tmux` session so the run survives disconnects:

```
tmux new -s train10a
cd ~/pocket_ligand
source <your_venv>/bin/activate
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python -m model.train 2>&1 | tee model/checkpoints/train_10A_$(date +%Y%m%d_%H%M%S).log
```

Detach with `Ctrl-b d`. Reattach with `tmux attach -t train10a`.

Outputs:
- Best checkpoint → `model/checkpoints/se3_pocket_ligand_best.pt`
- Per-epoch metrics CSV → `model/checkpoints/training_log_<timestamp>.csv`
- Console log → `model/checkpoints/train_10A_<timestamp>.log`

Expect roughly **30–60 min per epoch on A100 40 GB at bs=4**, faster on 80 GB at bs=8. Early stopping (patience 25) typically kicks in well before 200 epochs.

## Evaluate

After training finishes (or at any point — uses the best checkpoint):

```
python -m model.evaluate
```

Writes `model/checkpoints/test_predictions.csv` and prints per-target MAE / RMSE / R².

## If you hit OOM

Edit `model/config.py:50`:

1. Drop `batch_size` to 2, then 1.
2. If bs=1 still OOMs on A100 40 GB, reduce `num_radial_basis` from 32 → 24, or drop `radius_cutoff` from 10.0 → 9.0.
3. As a last resort, lower model capacity (`hidden_scalars`, `hidden_vectors`, `num_layers`).

The `expandable_segments:True` env var helps with fragmentation — keep it set.

## Knobs worth sweeping

- **Radius**: 8 / 10 / 12 Å. Bigger = more pocket context, more memory.
- **`num_radial_basis`**: keep ~0.3 Å spacing per Gaussian (`num_radial_basis ≈ 3 * radius_cutoff`).
- **`max_num_neighbors`** (currently uncapped in `architecture.py:454`): adding `max_num_neighbors=32` to the `radius_graph` call bounds edges in dense regions and can recover memory at fixed radius.
