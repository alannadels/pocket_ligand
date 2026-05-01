"""
Step 6: Create train/validation/test splits.

Splits the dataset by protein (UniProt ID) to prevent data leakage. All entries
for the same protein always go in the same split.

Strategy:
    - Only proteins with at least one pocket containing 3+ ligands are eligible
      for train/val splits (730 proteins). These have meaningful distribution
      labels (nonzero std, distinct min/max/median).
    - 70% of eligible proteins go to train, 15% to validation, 15% to test.
    - All remaining proteins (1,407 with only 1-2 ligand pockets) are added to
      the test set. These have unreliable distribution statistics but can still
      be used to check if known ligands fall within predicted property ranges.
    - Stratified by protein classification to ensure balanced representation of
      protein families across splits.

Input:
    ../data/pdbbind_with_pockets.csv    (10,372 entries with pocket assignments)
    ../data/pocket_distributions.csv    (2,295 pockets with distribution stats)

Output:
    ../data/train.csv             Training set entries (6,374 rows)
    ../data/val.csv               Validation set entries (1,071 rows)
    ../data/test.csv              Test set entries (2,927 rows)
    ../data/splits.csv            All entries with 'split' and 'split_reason' columns
    ../data/split_summary.csv     Per-split statistics

Dependencies:
    pip install pandas numpy scikit-learn
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ENTRIES_CSV = DATA_DIR / "pdbbind_with_pockets.csv"
POCKETS_CSV = DATA_DIR / "pocket_distributions.csv"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MIN_LIGANDS = 3        # Minimum ligands per pocket to be eligible for train/val
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    entries = pd.read_csv(ENTRIES_CSV)
    pockets = pd.read_csv(POCKETS_CSV)
    print(f"Loaded {len(entries)} entries, {len(pockets)} pockets")

    # --- Identify eligible proteins (have at least one pocket with 3+ ligands) ---
    eligible_pockets = pockets[pockets["n_ligands"] >= MIN_LIGANDS]
    eligible_proteins = set(eligible_pockets["uniprot_id"].unique())
    all_proteins = set(pockets["uniprot_id"].unique())
    small_proteins = all_proteins - eligible_proteins

    print(f"\nProteins with 3+ ligand pocket: {len(eligible_proteins)}")
    print(f"Proteins with only 1-2 ligand pockets: {len(small_proteins)}")

    # --- Build protein-level table for stratified splitting ---
    # For each eligible protein, get its classification for stratification
    protein_info = (
        entries[entries["uniprot_id"].isin(eligible_proteins)]
        .groupby("uniprot_id")["classification"]
        .first()
        .reset_index()
    )

    # Group rare classifications into "OTHER" to avoid stratification failures
    # Need enough members per class to survive two rounds of splitting
    class_counts = protein_info["classification"].value_counts()
    rare_classes = class_counts[class_counts < 10].index
    protein_info["strat_class"] = protein_info["classification"].where(
        ~protein_info["classification"].isin(rare_classes), "OTHER"
    )

    print(f"\nStratification classes: {protein_info['strat_class'].nunique()}")

    # --- Split eligible proteins: 70/15/15 ---
    # First split: 70% train, 30% temp
    train_proteins, temp_proteins = train_test_split(
        protein_info["uniprot_id"],
        test_size=(VAL_FRAC + TEST_FRAC),
        random_state=RANDOM_SEED,
        stratify=protein_info["strat_class"],
    )

    # Get strat_class for temp proteins
    temp_info = protein_info[protein_info["uniprot_id"].isin(temp_proteins)]

    # Second split: 50/50 of the 30% -> 15% val, 15% test
    val_proteins, test_proteins = train_test_split(
        temp_info["uniprot_id"],
        test_size=0.5,
        random_state=RANDOM_SEED,
        stratify=temp_info["strat_class"],
    )

    train_proteins = set(train_proteins)
    val_proteins = set(val_proteins)
    test_proteins = set(test_proteins)

    print(f"\nEligible protein split:")
    print(f"  Train: {len(train_proteins)} proteins")
    print(f"  Val:   {len(val_proteins)} proteins")
    print(f"  Test:  {len(test_proteins)} proteins")

    # --- Assign splits to entries ---
    def assign_split(row):
        uid = row["uniprot_id"]
        if uid in train_proteins:
            return "train", "eligible_train"
        elif uid in val_proteins:
            return "val", "eligible_val"
        elif uid in test_proteins:
            return "test", "eligible_test"
        elif uid in small_proteins:
            return "test", "small_pocket"
        else:
            return "test", "unknown"

    splits = entries.apply(assign_split, axis=1, result_type="expand")
    entries["split"] = splits[0]
    entries["split_reason"] = splits[1]

    # --- Summary statistics ---
    print(f"\n{'='*60}")
    print("SPLIT SUMMARY")
    print(f"{'='*60}")

    summary_rows = []
    for split_name in ["train", "val", "test"]:
        split_entries = entries[entries["split"] == split_name]
        split_pockets = pockets[pockets["pocket_id"].isin(split_entries["pocket_id"].unique())]
        split_proteins = split_entries["uniprot_id"].nunique()

        n_3plus = split_pockets[split_pockets["n_ligands"] >= MIN_LIGANDS].shape[0]
        n_small = split_pockets[split_pockets["n_ligands"] < MIN_LIGANDS].shape[0]

        print(f"\n{split_name.upper()}:")
        print(f"  Proteins:     {split_proteins}")
        print(f"  Pockets:      {len(split_pockets)} ({n_3plus} with 3+ ligands, {n_small} with <3)")
        print(f"  Point clouds: {len(split_entries)}")

        # Classification distribution
        class_dist = split_entries.groupby("uniprot_id")["classification"].first().value_counts()
        top3 = class_dist.head(3)
        print(f"  Top classes:  {', '.join(f'{c}({n})' for c, n in top3.items())}")

        summary_rows.append({
            "split": split_name,
            "n_proteins": split_proteins,
            "n_pockets": len(split_pockets),
            "n_pockets_3plus": n_3plus,
            "n_pockets_small": n_small,
            "n_point_clouds": len(split_entries),
        })

    # Test set breakdown
    test_entries = entries[entries["split"] == "test"]
    n_eligible_test = (test_entries["split_reason"] == "eligible_test").sum()
    n_small_test = (test_entries["split_reason"] == "small_pocket").sum()
    print(f"\nTest set breakdown:")
    print(f"  From eligible proteins (3+ ligands): {n_eligible_test} entries")
    print(f"  From small-pocket proteins (1-2 ligands): {n_small_test} entries")

    # --- Save ---
    # Combined file with split column
    splits_path = DATA_DIR / "splits.csv"
    entries.to_csv(splits_path, index=False)
    print(f"\nSaved: {splits_path} ({len(entries)} rows)")

    # Separate CSV per split
    for split_name in ["train", "val", "test"]:
        split_df = entries[entries["split"] == split_name]
        split_path = DATA_DIR / f"{split_name}.csv"
        split_df.to_csv(split_path, index=False)
        print(f"Saved: {split_path} ({len(split_df)} rows)")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = DATA_DIR / "split_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    # --- Sanity checks ---
    print(f"\nSanity checks:")

    # No protein appears in multiple splits
    for p in all_proteins:
        splits_for_p = entries[entries["uniprot_id"] == p]["split"].unique()
        assert len(splits_for_p) == 1, f"Protein {p} in multiple splits: {splits_for_p}"
    print("  ✓ No protein leakage across splits")

    # All entries assigned
    assert entries["split"].notna().all()
    print("  ✓ All entries have a split assignment")

    # Counts add up
    total = sum(r["n_point_clouds"] for r in summary_rows)
    assert total == len(entries), f"Count mismatch: {total} vs {len(entries)}"
    print("  ✓ Entry counts match")

    print("\nDone.")


if __name__ == "__main__":
    main()
