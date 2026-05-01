"""
Data Exploration: High-quality figures exploring the PDBbind dataset.

Generates 10 publication-quality figures (300 DPI) covering dataset composition,
physicochemical property distributions, pocket clustering results, and more.

Input files (all from ../data/):
    pdbbind_with_pockets.csv   - Per-ligand data with pocket assignments (10,372 rows)
    pocket_distributions.csv   - Per-pocket aggregated statistics (2,295 rows)
    pocket_summary.csv         - Per-protein pocket clustering summary (2,137 rows)

Output:
    figures/*.png              - 10 high-quality figures at 300 DPI

Dependencies:
    pip install pandas matplotlib seaborn numpy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

PALETTE = sns.color_palette("Set2", 10)

# Property metadata for labels and units
PROPERTIES = {
    "MW": ("Molecular Weight", "Daltons"),
    "logP": ("logP", ""),
    "HBD": ("H-Bond Donors", "count"),
    "HBA": ("H-Bond Acceptors", "count"),
    "TPSA": ("Polar Surface Area", "Å²"),
    "rotatable_bonds": ("Rotatable Bonds", "count"),
    "formal_charge": ("Formal Charge", "e"),
    "aromatic_rings": ("Aromatic Rings", "count"),
    "fsp3": ("Fraction sp3", ""),
}


def load_data():
    """Load all datasets."""
    ligands = pd.read_csv(DATA_DIR / "pdbbind_with_pockets.csv")
    pockets = pd.read_csv(DATA_DIR / "pocket_distributions.csv")
    summary = pd.read_csv(DATA_DIR / "pocket_summary.csv")
    return ligands, pockets, summary


# ===== Figure 1: Dataset Overview Dashboard =====
def fig01_dataset_overview(ligands, pockets, summary):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Dataset Overview", fontsize=14, fontweight="bold", y=0.98)

    # (a) Pipeline funnel
    ax = axes[0, 0]
    stages = ["Raw PDBbind", "Quality Filtered", "UniProt Mapped", "Aligned"]
    counts = [19037, 10692, 10542, len(ligands)]
    bars = ax.barh(stages[::-1], counts[::-1], color=[PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[3]])
    for bar, count in zip(bars, counts[::-1]):
        ax.text(bar.get_width() + 200, bar.get_y() + bar.get_height() / 2,
                f"{count:,}", va="center", fontsize=9)
    ax.set_xlabel("Number of Entries")
    ax.set_title("(a) Data Pipeline")
    ax.set_xlim(0, 22000)

    # (b) Key numbers
    ax = axes[0, 1]
    ax.axis("off")
    stats = [
        ("Ligand-Pocket Complexes", f"{len(ligands):,}"),
        ("Unique Proteins", f"{len(summary):,}"),
        ("Unique Binding Pockets", f"{len(pockets):,}"),
        ("Multi-Pocket Proteins", f"{(summary['n_pockets'] > 1).sum():,}"),
        ("Median Ligands per Pocket", f"{pockets['n_ligands'].median():.0f}"),
        ("Resolution Range", f"{ligands['resolution'].min():.2f}–{ligands['resolution'].max():.2f} Å"),
        ("Affinity Range", f"{ligands['affinity_nM'].min():.1f}–{ligands['affinity_nM'].max():.0f} nM"),
    ]
    for i, (label, value) in enumerate(stats):
        y = 0.88 - i * 0.12
        ax.text(0.05, y, label, fontsize=10, transform=ax.transAxes, va="center")
        ax.text(0.95, y, value, fontsize=10, fontweight="bold", transform=ax.transAxes,
                va="center", ha="right")
    ax.set_title("(b) Key Statistics")

    # (c) Affinity type breakdown
    ax = axes[1, 0]
    aff_counts = ligands["affinity_type"].value_counts()
    wedges, texts, autotexts = ax.pie(
        aff_counts.values, labels=aff_counts.index, autopct="%1.1f%%",
        colors=PALETTE[:len(aff_counts)], startangle=90)
    for t in autotexts:
        t.set_fontsize(8)
    ax.set_title("(c) Affinity Type Distribution")

    # (d) Resolution histogram
    ax = axes[1, 1]
    ax.hist(ligands["resolution"].dropna(), bins=40, color=PALETTE[0], edgecolor="white", linewidth=0.5)
    ax.axvline(ligands["resolution"].median(), color="red", linestyle="--", linewidth=1,
               label=f"Median: {ligands['resolution'].median():.2f} Å")
    ax.set_xlabel("Resolution (Å)")
    ax.set_ylabel("Count")
    ax.set_title("(d) Crystal Structure Resolution")
    ax.legend()

    plt.tight_layout()
    fig.savefig(FIG_DIR / "01_dataset_overview.png")
    plt.close(fig)
    print("  Saved 01_dataset_overview.png")


# ===== Figure 2: Physicochemical Property Distributions =====
def fig02_property_distributions(ligands):
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    fig.suptitle("Ligand Physicochemical Property Distributions (n={:,})".format(len(ligands)),
                 fontsize=14, fontweight="bold", y=1.0)

    for idx, (prop, (name, unit)) in enumerate(PROPERTIES.items()):
        ax = axes[idx // 3, idx % 3]
        data = ligands[prop].dropna()

        if prop in ("HBD", "HBA", "rotatable_bonds", "formal_charge", "aromatic_rings"):
            vals = data.astype(int)
            val_range = range(int(vals.min()), int(vals.max()) + 1)
            ax.hist(vals, bins=np.arange(vals.min() - 0.5, vals.max() + 1.5, 1),
                    color=PALETTE[idx], edgecolor="white", linewidth=0.5)
        else:
            ax.hist(data, bins=50, color=PALETTE[idx], edgecolor="white", linewidth=0.5)

        label = f"{name} ({unit})" if unit else name
        ax.set_xlabel(label)
        ax.set_ylabel("Count")

        med = data.median()
        ax.axvline(med, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.text(0.97, 0.95, f"med={med:.1f}\nμ={data.mean():.1f}\nσ={data.std():.1f}",
                transform=ax.transAxes, fontsize=7, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    fig.savefig(FIG_DIR / "02_property_distributions.png")
    plt.close(fig)
    print("  Saved 02_property_distributions.png")


# ===== Figure 3: Property Correlation Heatmap =====
def fig03_correlation_heatmap(ligands):
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.suptitle("Physicochemical Property Correlations", fontsize=14, fontweight="bold")

    prop_cols = list(PROPERTIES.keys())
    labels = [PROPERTIES[p][0] for p in prop_cols]
    corr = ligands[prop_cols].corr()

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.5,
                xticklabels=labels, yticklabels=labels, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "03_correlation_heatmap.png")
    plt.close(fig)
    print("  Saved 03_correlation_heatmap.png")


# ===== Figure 4: Pairwise Scatter Matrix (selected pairs) =====
def fig04_scatter_pairs(ligands):
    pairs = [
        ("MW", "logP"), ("MW", "TPSA"), ("logP", "TPSA"),
        ("HBD", "HBA"), ("aromatic_rings", "fsp3"), ("MW", "rotatable_bonds"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle("Key Property Relationships", fontsize=14, fontweight="bold", y=1.0)

    for idx, (px, py) in enumerate(pairs):
        ax = axes[idx // 3, idx % 3]
        x = ligands[px].dropna()
        y = ligands[py].dropna()
        common = x.index.intersection(y.index)

        ax.scatter(x[common], y[common], alpha=0.08, s=4, color=PALETTE[idx], rasterized=True)

        nx = PROPERTIES[px][0]
        ny = PROPERTIES[py][0]
        ux = f" ({PROPERTIES[px][1]})" if PROPERTIES[px][1] else ""
        uy = f" ({PROPERTIES[py][1]})" if PROPERTIES[py][1] else ""
        ax.set_xlabel(f"{nx}{ux}")
        ax.set_ylabel(f"{ny}{uy}")

        r = ligands.loc[common, [px, py]].corr().iloc[0, 1]
        ax.text(0.03, 0.97, f"r = {r:.2f}", transform=ax.transAxes, fontsize=8,
                va="top", bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    fig.savefig(FIG_DIR / "04_scatter_pairs.png")
    plt.close(fig)
    print("  Saved 04_scatter_pairs.png")


# ===== Figure 5: Temporal Evolution =====
def fig05_temporal(ligands):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("Dataset Temporal Evolution", fontsize=14, fontweight="bold", y=1.02)

    # (a) Entries per year
    ax = axes[0]
    year_counts = ligands["year"].value_counts().sort_index()
    ax.bar(year_counts.index, year_counts.values, color=PALETTE[0], edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Deposition Year")
    ax.set_ylabel("Number of Entries")
    ax.set_title("(a) Entries per Year")

    # (b) Resolution trend
    ax = axes[1]
    yearly_res = ligands.groupby("year")["resolution"].median()
    ax.plot(yearly_res.index, yearly_res.values, "o-", color=PALETTE[1], markersize=3, linewidth=1.2)
    ax.set_xlabel("Deposition Year")
    ax.set_ylabel("Median Resolution (Å)")
    ax.set_title("(b) Resolution Trend")
    ax.invert_yaxis()

    # (c) MW trend
    ax = axes[2]
    yearly_mw = ligands.groupby("year")["MW"].median()
    ax.plot(yearly_mw.index, yearly_mw.values, "o-", color=PALETTE[2], markersize=3, linewidth=1.2)
    ax.set_xlabel("Deposition Year")
    ax.set_ylabel("Median MW (Da)")
    ax.set_title("(c) Ligand Size Trend")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "05_temporal_evolution.png")
    plt.close(fig)
    print("  Saved 05_temporal_evolution.png")


# ===== Figure 6: Organism and Classification Diversity =====
def fig06_diversity(ligands):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("Protein Diversity", fontsize=14, fontweight="bold", y=1.0)

    # (a) Top organisms
    ax = axes[0]
    org_counts = ligands.groupby("uniprot_id")["organism"].first().value_counts()
    top_n = 12
    top_orgs = org_counts.head(top_n)
    other = org_counts.iloc[top_n:].sum()
    if other > 0:
        top_orgs = pd.concat([top_orgs, pd.Series({"Other": other})])
    bars = ax.barh(top_orgs.index[::-1], top_orgs.values[::-1], color=PALETTE[0])
    for bar, val in zip(bars, top_orgs.values[::-1]):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=7)
    ax.set_xlabel("Number of Unique Proteins")
    ax.set_title("(a) Top Organisms")

    # (b) Top classifications
    ax = axes[1]
    cls_counts = ligands.groupby("uniprot_id")["classification"].first().value_counts()
    top_cls = cls_counts.head(top_n)
    other_cls = cls_counts.iloc[top_n:].sum()
    if other_cls > 0:
        top_cls = pd.concat([top_cls, pd.Series({"Other": other_cls})])
    bars = ax.barh(top_cls.index[::-1], top_cls.values[::-1], color=PALETTE[3])
    for bar, val in zip(bars, top_cls.values[::-1]):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=7)
    ax.set_xlabel("Number of Unique Proteins")
    ax.set_title("(b) Protein Classification")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "06_diversity.png")
    plt.close(fig)
    print("  Saved 06_diversity.png")


# ===== Figure 7: Pocket Size Distribution and Clustering =====
def fig07_pocket_clustering(pockets, summary):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("Pocket Clustering Results", fontsize=14, fontweight="bold", y=1.02)

    # (a) Ligands per pocket (log scale)
    ax = axes[0]
    bins = np.logspace(0, np.log10(pockets["n_ligands"].max() + 1), 40)
    ax.hist(pockets["n_ligands"], bins=bins, color=PALETTE[0], edgecolor="white", linewidth=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("Ligands per Pocket")
    ax.set_ylabel("Number of Pockets")
    ax.set_title(f"(a) Pocket Size Distribution (n={len(pockets):,})")
    ax.text(0.97, 0.95,
            f"1 ligand: {(pockets['n_ligands'] == 1).sum()}\n"
            f"2-5: {((pockets['n_ligands'] >= 2) & (pockets['n_ligands'] <= 5)).sum()}\n"
            f"6-20: {((pockets['n_ligands'] >= 6) & (pockets['n_ligands'] <= 20)).sum()}\n"
            f"21-100: {((pockets['n_ligands'] >= 21) & (pockets['n_ligands'] <= 100)).sum()}\n"
            f">100: {(pockets['n_ligands'] > 100).sum()}",
            transform=ax.transAxes, fontsize=7, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # (b) Pockets per protein
    ax = axes[1]
    pocket_counts = summary["n_pockets"].value_counts().sort_index()
    ax.bar(pocket_counts.index, pocket_counts.values, color=PALETTE[1], edgecolor="white")
    ax.set_xlabel("Pockets per Protein")
    ax.set_ylabel("Number of Proteins")
    ax.set_title(f"(b) Pockets per Protein (n={len(summary):,})")
    for x, y in zip(pocket_counts.index, pocket_counts.values):
        ax.text(x, y + 5, str(y), ha="center", fontsize=7)

    # (c) Entries dropped per protein
    ax = axes[2]
    dropped = summary[summary["n_dropped"] > 0]["n_dropped"]
    if len(dropped) > 0:
        ax.hist(dropped, bins=range(1, int(dropped.max()) + 2), color=PALETTE[2],
                edgecolor="white", linewidth=0.5, align="left")
        ax.set_xlabel("Entries Dropped")
        ax.set_ylabel("Number of Proteins")
        ax.set_title(f"(c) Alignment Failures ({len(dropped)} proteins affected)")
    else:
        ax.text(0.5, 0.5, "No dropped entries", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("(c) Alignment Failures")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "07_pocket_clustering.png")
    plt.close(fig)
    print("  Saved 07_pocket_clustering.png")


# ===== Figure 8: Binding Affinity Landscape =====
def fig08_affinity(ligands):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("Binding Affinity Landscape", fontsize=14, fontweight="bold", y=1.02)

    aff = ligands["affinity_nM"].dropna()
    log_aff = np.log10(aff.clip(lower=1e-3))

    # (a) Affinity distribution (log scale)
    ax = axes[0]
    ax.hist(log_aff, bins=60, color=PALETTE[0], edgecolor="white", linewidth=0.5)
    ax.set_xlabel("log₁₀(Affinity / nM)")
    ax.set_ylabel("Count")
    ax.set_title("(a) Affinity Distribution")
    ax.axvline(log_aff.median(), color="red", linestyle="--", linewidth=1,
               label=f"Median: {10**log_aff.median():.0f} nM")
    ax.legend()

    # (b) Affinity vs MW
    ax = axes[1]
    ax.scatter(ligands["MW"], log_aff, alpha=0.08, s=4, color=PALETTE[1], rasterized=True)
    ax.set_xlabel("Molecular Weight (Da)")
    ax.set_ylabel("log₁₀(Affinity / nM)")
    ax.set_title("(b) Affinity vs Molecular Weight")

    # (c) Affinity by type
    ax = axes[2]
    types = ligands["affinity_type"].unique()
    data_by_type = [np.log10(ligands[ligands["affinity_type"] == t]["affinity_nM"].clip(lower=1e-3))
                    for t in types]
    bp = ax.boxplot(data_by_type, labels=types, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xlabel("Affinity Type")
    ax.set_ylabel("log₁₀(Affinity / nM)")
    ax.set_title("(c) Affinity by Measurement Type")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "08_affinity_landscape.png")
    plt.close(fig)
    print("  Saved 08_affinity_landscape.png")


# ===== Figure 9: Within-Pocket Property Variation =====
def fig09_within_pocket_variation(pockets):
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    fig.suptitle("Within-Pocket Property Variation (Coefficient of Variation)",
                 fontsize=14, fontweight="bold", y=1.0)

    # Only use pockets with >= 5 ligands for meaningful CV
    multi = pockets[pockets["n_ligands"] >= 5].copy()

    for idx, (prop, (name, unit)) in enumerate(PROPERTIES.items()):
        ax = axes[idx // 3, idx % 3]
        mean_col = f"{prop}_mean"
        std_col = f"{prop}_std"

        means = multi[mean_col]
        stds = multi[std_col]

        # CV = std / |mean|, skip where mean is near zero
        valid = means.abs() > 1e-6
        cv = (stds[valid] / means[valid].abs()).clip(upper=5)

        if len(cv) > 0:
            ax.hist(cv, bins=40, color=PALETTE[idx], edgecolor="white", linewidth=0.5)
            ax.axvline(cv.median(), color="black", linestyle="--", linewidth=0.8)
            ax.text(0.97, 0.95, f"med CV={cv.median():.2f}\nn={len(cv)} pockets",
                    transform=ax.transAxes, fontsize=7, va="top", ha="right",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.set_xlabel(f"CV ({name})")
        ax.set_ylabel("Count")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "09_within_pocket_variation.png")
    plt.close(fig)
    print("  Saved 09_within_pocket_variation.png")


# ===== Figure 10: Between-Pocket Property Means =====
def fig10_between_pocket_diversity(pockets):
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    fig.suptitle("Property Means Across Pockets (Between-Pocket Diversity)",
                 fontsize=14, fontweight="bold", y=1.0)

    for idx, (prop, (name, unit)) in enumerate(PROPERTIES.items()):
        ax = axes[idx // 3, idx % 3]
        col = f"{prop}_mean"
        data = pockets[col].dropna()

        ax.hist(data, bins=50, color=PALETTE[idx], edgecolor="white", linewidth=0.5)

        label = f"{name} ({unit})" if unit else name
        ax.set_xlabel(f"Pocket Mean {label}")
        ax.set_ylabel("Count")

        ax.text(0.97, 0.95,
                f"μ={data.mean():.1f}\nσ={data.std():.1f}\nn={len(data)}",
                transform=ax.transAxes, fontsize=7, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    fig.savefig(FIG_DIR / "10_between_pocket_diversity.png")
    plt.close(fig)
    print("  Saved 10_between_pocket_diversity.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading data...")
    ligands, pockets, summary = load_data()
    print(f"  Ligands: {len(ligands):,} rows")
    print(f"  Pockets: {len(pockets):,} rows")
    print(f"  Proteins: {len(summary):,} rows")
    print()

    print("Generating figures...")
    fig01_dataset_overview(ligands, pockets, summary)
    fig02_property_distributions(ligands)
    fig03_correlation_heatmap(ligands)
    fig04_scatter_pairs(ligands)
    fig05_temporal(ligands)
    fig06_diversity(ligands)
    fig07_pocket_clustering(pockets, summary)
    fig08_affinity(ligands)
    fig09_within_pocket_variation(pockets)
    fig10_between_pocket_diversity(pockets)

    print(f"\nAll figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
