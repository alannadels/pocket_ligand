# Train/Validation/Test Split

This document describes how the dataset is split for model training, produced by
`scripts/create_splits.py` (Step 6).

## Output Files

- `data/train.csv` — Training set (6,374 rows)
- `data/val.csv` — Validation set (1,071 rows)
- `data/test.csv` — Test set (2,927 rows)
- `data/splits.csv` — The full dataset (10,372 rows) with two new columns:
  - `split`: one of `train`, `val`, or `test`
  - `split_reason`: why this entry is in its split (see below)
- `data/split_summary.csv` — Per-split statistics (protein count, pocket count, point cloud count)

Each of `train.csv`, `val.csv`, and `test.csv` contains the same columns as `pdbbind_with_pockets.csv` plus `split` and `split_reason`. These can be loaded directly by the training code without any filtering.

## Why Split by Protein

The fundamental rule: **all entries for the same protein (UniProt ID) must go in the same split.**

If even one crystal structure of HIV protease appears in training while another appears in test, the model has effectively seen the test pocket geometry during training. Since many entries for the same protein share identical or near-identical pocket geometry, this would be severe data leakage — the model could memorize pocket shapes rather than learning generalizable pocket-property relationships.

Splitting by protein ensures the test set contains truly unseen pocket geometries.

## Minimum Ligand Count for Training

Pockets with only 1-2 known ligands have unreliable distribution statistics:
- With 1 ligand: std = 0, min = max = median = mean. The "distribution" is a single point.
- With 2 ligands: std is based on just 2 observations. Min and max are the only two values.

Training the model on these labels would teach it that zero variance is common and that distributions collapse to point estimates — neither of which is true for well-characterized pockets.

**Decision:** Only proteins that have at least one pocket with 3+ ligands are eligible for the train and validation splits. With 3 ligands, you get a nonzero std, a median distinct from the mean, and min/max that bound a meaningful range.

Proteins with only 1-2 ligand pockets are placed in the test set. They can still be used for evaluation: check whether the 1-2 known ligands fall within the model's predicted property ranges.

## Split Ratio: 70/15/15

The 730 eligible proteins (those with at least one 3+ ligand pocket) are split 70/15/15:
- **70% train** (~511 proteins) — enough data to learn pocket-property relationships
- **15% validation** (~109 proteins) — used for hyperparameter tuning, learning rate scheduling, and early stopping. 109 proteins provides a stable validation signal.
- **15% test** (~110 proteins) — held out for final evaluation, never used during training or tuning

We chose 70/15/15 over 80/10/10 because 109 validation proteins gives a more reliable signal for hyperparameter decisions than 73 would. The cost is ~70 fewer training proteins, but with 511 training proteins contributing 6,374 point clouds, training data is sufficient.

## Stratification

The split is stratified by protein classification (HYDROLASE, TRANSFERASE, LYASE, etc.) to ensure each split has a representative mix of protein families. Without stratification, random chance could put all kinases in training and all proteases in test, making the evaluation unrepresentative.

Classifications with fewer than 10 proteins are grouped into an "OTHER" category for stratification to avoid splitting failures when a class has too few members.

## Test Set Composition

The test set has two tiers:

### Tier 1: Eligible proteins (3+ ligand pockets)
- 110 proteins, 114 pockets with 3+ ligands
- 1,128 point clouds
- **Evaluation method:** Compare predicted distribution statistics (mean, std, median, min, max) against actual distributions. This is the primary evaluation metric.

### Tier 2: Small-pocket proteins (1-2 ligand pockets)
- 1,407 proteins, 1,471 pockets
- 1,799 point clouds
- **Evaluation method:** Check whether the 1-2 known ligands' properties fall within the model's predicted property ranges. This tests generalization to data-sparse targets — the real-world scenario where a protein has been crystallized with only one drug candidate.

## The `split_reason` Column

| Value | Meaning |
|-------|---------|
| eligible_train | Protein has 3+ ligand pocket, assigned to training set |
| eligible_val | Protein has 3+ ligand pocket, assigned to validation set |
| eligible_test | Protein has 3+ ligand pocket, assigned to test set |
| small_pocket | Protein has only 1-2 ligand pockets, assigned to test set by default |

## Split Statistics

| Split | Proteins | Pockets (3+ ligands) | Pockets (<3 ligands) | Point Clouds |
|-------|----------|---------------------|---------------------|--------------|
| Train | 511 | 531 | 50 | 6,374 |
| Val | 109 | 114 | 15 | 1,071 |
| Test | 1,517 | 114 | 1,471 | 2,927 |
| **Total** | **2,137** | **759** | **1,536** | **10,372** |

Note: Train and val proteins may also have some pockets with <3 ligands (50 and 15 respectively). These small pockets still appear in the split since they belong to eligible proteins — they just have less reliable labels for those specific pockets.

## Reproducibility

The split uses `random_state=42` for both splitting steps, making it fully deterministic. Running `create_splits.py` again will produce identical splits.

## Sanity Checks (verified by script)

- No protein appears in multiple splits (no data leakage)
- All 10,372 entries have a split assignment
- Entry counts across splits sum to total
