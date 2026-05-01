"""
Step 4: Group ligands by pocket and compute per-pocket property distributions.

A single protein can have multiple binding pockets (orthosteric, allosteric,
cofactor sites, etc.). Different ligands for the same protein may bind to
different pockets, and mixing their properties would create meaningless
distributions. This script identifies distinct pockets within each protein
and computes property distributions per pocket — the actual training targets
for our model.

Pocket identification strategy:
    We use 3D structural alignment to determine whether two ligands bind
    the same pocket. For each protein with multiple crystal structures:

    1. Extract the ligand's 3D centroid (center of mass) from its SDF file.
    2. Extract protein backbone CA (alpha-carbon) atoms from the full protein
       PDB file, filtered to the pocket chain.
    3. Pick a reference structure and align all other structures to it using
       the Kabsch algorithm (optimal rigid-body superposition of matched CA
       atoms).
    4. Transform each ligand centroid into the reference coordinate frame.
    5. Cluster the aligned centroids by spatial proximity — ligands within
       10 Angstroms of each other bind the same pocket.
    6. Compute property distributions (mean, std, min, max, median, count)
       per pocket for all 9 physicochemical properties.

    For matching CA atoms across structures, we first try matching by
    residue ID (resname + resnum). If too few matches are found (indicating
    inconsistent residue numbering across PDB depositions), we fall back to
    sequence alignment (Needleman-Wunsch) to establish residue correspondences
    purely from the amino acid sequence.

Input:
    ../data/pdbbind_enriched.csv  (produced by enrich_protein_metadata.py)

Output:
    ../data/pdbbind_with_pockets.csv
        The enriched dataset with a new pocket_id column (format: {uniprot}_{n}).

    ../data/pocket_distributions.csv
        One row per pocket, with aggregate statistics for each of the 9
        ligand properties across all ligands assigned to that pocket.

    ../data/pocket_summary.csv
        Summary of pocket clustering results per protein.

Dependencies:
    pip install pandas scipy numpy rdkit
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"
INPUT_CSV = OUTPUT_DIR / "pdbbind_enriched.csv"
PDBBIND_DIR = Path("/Users/alannadels/Desktop/datasets/PDBbind/P-L")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Distance threshold for clustering ligand centroids into the same pocket.
# Two ligands are in the same pocket if their centroids (after structural
# alignment) are within this distance. Binding pockets are typically 10-20
# Angstroms across. Two ligands in the same pocket usually have centroids
# within 5-8 Angstroms. Ligands in different pockets are typically >15
# Angstroms apart. 10 Angstroms provides a good separation boundary.
CENTROID_DISTANCE_THRESHOLD = 10.0  # Angstroms

# Minimum number of matched CA atoms required for a reliable structural
# alignment. With fewer than this, the rotation/translation is under-
# determined and the aligned centroids can't be trusted.
MIN_MATCHED_CA = 10

# Maximum acceptable RMSD (Angstroms) for a structural alignment.
# Alignments above this threshold are considered unreliable — usually
# caused by fusion proteins, chimeras, or multi-domain constructs where
# the pocket chain contains much more than the target protein domain.
# Normal same-protein alignments have RMSD < 2-3 Angstroms.
MAX_ALIGNMENT_RMSD = 5.0

# The 9 physicochemical properties we compute distributions for.
PROPERTY_COLUMNS = [
    "MW", "logP", "HBD", "HBA", "TPSA",
    "rotatable_bonds", "formal_charge", "aromatic_rings", "fsp3"
]

# 3-letter to 1-letter amino acid code mapping, used for sequence alignment
# when residue numbering is inconsistent across PDB structures.
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


# ---------------------------------------------------------------------------
# Structure parsing
# ---------------------------------------------------------------------------

def get_chain_ca_atoms(pdb_code, data_subdir, chain_id):
    """
    Extract CA (alpha-carbon) atoms from a specific chain of the full
    protein PDB file.

    CA atoms define the protein backbone — one per residue. They provide
    a compact representation of the protein's 3D fold that is sufficient
    for structural alignment. We filter to the pocket chain (the chain
    that forms the binding site) to avoid including unrelated chains
    (e.g., a peptide inhibitor or a partner protein in a complex).

    Args:
        pdb_code:    4-character PDB identifier
        data_subdir: PDBbind time-period subdirectory
        chain_id:    Single-letter chain identifier to extract

    Returns:
        List of dicts, each with keys:
            resname: 3-letter amino acid name (e.g., "ALA")
            resnum:  Residue sequence number as string (e.g., "142")
            resid:   Combined identifier "ALA142" for matching
            coords:  numpy array of [x, y, z] coordinates
        Returns empty list if the file can't be read.
    """
    path = PDBBIND_DIR / data_subdir / pdb_code / f"{pdb_code}_protein.pdb"
    ca_atoms = []

    seen_resids = set()  # Track seen residues to skip alternate conformations

    try:
        with open(path, "r") as f:
            for line in f:
                if not line.startswith("ATOM"):
                    continue
                # CA atom check: columns 12-15 contain the atom name
                atom_name = line[12:16].strip()
                if atom_name != "CA":
                    continue
                # Chain check: column 21
                chain = line[21].strip()
                if chain != chain_id:
                    continue
                # Skip alternate conformations: column 16 is the altloc
                # indicator. Keep only the first conformation (' ' or 'A')
                # to avoid duplicate CA atoms for the same residue.
                altloc = line[16]
                if altloc not in (" ", "A", ""):
                    continue

                resname = line[17:20].strip()
                resnum = line[22:26].strip()
                # Include insertion code (column 26) to distinguish
                # residues like 27 vs 27A
                insertion = line[26].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])

                resid = f"{resname}{resnum}{insertion}"

                # Skip if we've already seen this residue (extra safety
                # against duplicates beyond altloc filtering)
                if resid in seen_resids:
                    continue
                seen_resids.add(resid)

                ca_atoms.append({
                    "resname": resname,
                    "resnum": resnum,
                    "resid": resid,
                    "coords": np.array([x, y, z]),
                })
    except FileNotFoundError:
        pass

    return ca_atoms


def get_ligand_centroid(pdb_code, data_subdir):
    """
    Compute the 3D centroid (center of mass) of a ligand from its SDF file.

    The centroid is the average position of all heavy atoms in the ligand.
    This single point summarizes where the ligand sits in 3D space relative
    to the protein. After structural alignment, comparing centroids across
    structures tells us whether ligands bind the same pocket.

    Args:
        pdb_code:    4-character PDB identifier
        data_subdir: PDBbind time-period subdirectory

    Returns:
        numpy array of [x, y, z] centroid coordinates, or None if the
        ligand file can't be read.
    """
    # Try SDF first (preferred format)
    sdf_path = PDBBIND_DIR / data_subdir / pdb_code / f"{pdb_code}_ligand.sdf"
    if sdf_path.exists():
        supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=True)
        for mol in supplier:
            if mol is not None and mol.GetNumConformers() > 0:
                return mol.GetConformer().GetPositions().mean(axis=0)

    # Fallback to MOL2
    mol2_path = PDBBIND_DIR / data_subdir / pdb_code / f"{pdb_code}_ligand.mol2"
    if mol2_path.exists():
        mol = Chem.MolFromMol2File(str(mol2_path), sanitize=False, removeHs=True)
        if mol is not None and mol.GetNumConformers() > 0:
            return mol.GetConformer().GetPositions().mean(axis=0)

    return None


# ---------------------------------------------------------------------------
# Structural alignment
# ---------------------------------------------------------------------------

def kabsch(P, Q):
    """
    Compute the optimal rotation and translation to align point set P onto Q.

    Uses the Kabsch algorithm (SVD-based). Finds rotation matrix R and
    translation vector t that minimize the RMSD between R @ P + t and Q.

    The sign correction (d matrix) handles the reflection case — without it,
    the algorithm might return an improper rotation (reflection + rotation)
    when the optimal alignment involves a reflection. We force a proper
    rotation by flipping the sign of the smallest singular value component.

    Args:
        P: Nx3 numpy array of points to be aligned (moved).
        Q: Nx3 numpy array of target points (fixed).

    Returns:
        R: 3x3 rotation matrix
        t: 3-element translation vector
        Such that: aligned_point = R @ point + t
    """
    centroid_P = P.mean(axis=0)
    centroid_Q = Q.mean(axis=0)

    # Center both point sets at origin
    Pc = P - centroid_P
    Qc = Q - centroid_Q

    # Cross-covariance matrix
    H = Pc.T @ Qc

    # SVD decomposition
    U, S, Vt = np.linalg.svd(H)

    # Ensure proper rotation (det = +1, not -1)
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1.0, 1.0, d])

    # Optimal rotation and translation
    R = Vt.T @ sign_matrix @ U.T
    t = centroid_Q - R @ centroid_P

    return R, t


def match_ca_by_resid(ca_ref, ca_tgt):
    """
    Match CA atoms between two structures by residue identifier.

    This works when both structures use the same residue numbering scheme
    (the common case — ~96% of proteins). Each CA atom is identified by
    its residue name + number (e.g., "ALA142"), and atoms with matching
    identifiers are paired.

    Args:
        ca_ref: List of CA atom dicts from reference structure.
        ca_tgt: List of CA atom dicts from target structure.

    Returns:
        Tuple of (matched_ref, matched_tgt) — both Nx3 numpy arrays of
        corresponding atom coordinates. Empty arrays if no matches found.
    """
    ref_dict = {a["resid"]: a["coords"] for a in ca_ref}

    matched_ref = []
    matched_tgt = []
    for a in ca_tgt:
        if a["resid"] in ref_dict:
            matched_ref.append(ref_dict[a["resid"]])
            matched_tgt.append(a["coords"])

    if not matched_ref:
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)
    return np.array(matched_ref), np.array(matched_tgt)


def match_ca_by_sequence(ca_ref, ca_tgt):
    """
    Match CA atoms by local sequence alignment (Smith-Waterman).

    Used when residue numbering is inconsistent across PDB structures.
    Converts the CA atom residue names to a 1-letter amino acid sequence,
    finds the best local alignment, and pairs atoms at matching positions.

    We use Smith-Waterman (local) rather than Needleman-Wunsch (global)
    because some PDB entries are fusion proteins or multi-domain constructs
    where the pocket chain contains much more than just the target protein.
    Global alignment would force alignment of the entire longer sequence
    against the shorter one, potentially matching the wrong domain. Local
    alignment finds the best matching subsequence — the target protein
    domain within the larger chain.

    Scoring: +2 for matches, -1 for mismatches, -1 for gaps.
    Higher match bonus encourages aligning through the correct domain
    rather than accumulating gap penalties.

    Args:
        ca_ref: List of CA atom dicts from reference structure.
        ca_tgt: List of CA atom dicts from target structure.

    Returns:
        Tuple of (matched_ref, matched_tgt) — both Nx3 numpy arrays.
        Empty arrays if alignment produces fewer than MIN_MATCHED_CA matches.
    """
    seq_ref = [THREE_TO_ONE.get(a["resname"], "X") for a in ca_ref]
    seq_tgt = [THREE_TO_ONE.get(a["resname"], "X") for a in ca_tgt]

    n = len(seq_ref)
    m = len(seq_tgt)

    if n == 0 or m == 0:
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)

    # --- Smith-Waterman local alignment ---
    # Unlike Needleman-Wunsch, scores are clamped to >= 0, so the
    # alignment can start and end anywhere in either sequence.
    MATCH_SCORE = 2
    MISMATCH_PENALTY = -1
    GAP_PENALTY = -1

    score = np.zeros((n + 1, m + 1))
    max_score = 0
    max_i, max_j = 0, 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = score[i - 1, j - 1] + (
                MATCH_SCORE if seq_ref[i - 1] == seq_tgt[j - 1]
                else MISMATCH_PENALTY
            )
            up = score[i - 1, j] + GAP_PENALTY
            left = score[i, j - 1] + GAP_PENALTY
            score[i, j] = max(0, diag, up, left)

            if score[i, j] > max_score:
                max_score = score[i, j]
                max_i, max_j = i, j

    # Traceback from the maximum score cell, stopping at 0
    pairs = []
    i, j = max_i, max_j
    while i > 0 and j > 0 and score[i, j] > 0:
        current = score[i, j]
        diag = score[i - 1, j - 1] + (
            MATCH_SCORE if seq_ref[i - 1] == seq_tgt[j - 1]
            else MISMATCH_PENALTY
        )
        if current == diag:
            if seq_ref[i - 1] == seq_tgt[j - 1]:
                pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif current == score[i - 1, j] + GAP_PENALTY:
            i -= 1
        else:
            j -= 1

    pairs.reverse()

    if len(pairs) < MIN_MATCHED_CA:
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)

    matched_ref = np.array([ca_ref[i]["coords"] for i, j in pairs])
    matched_tgt = np.array([ca_tgt[j]["coords"] for i, j in pairs])
    return matched_ref, matched_tgt


def align_and_transform(ca_ref, ca_tgt, ligand_centroid_tgt):
    """
    Align a target structure to a reference and transform its ligand centroid.

    First attempts to match CA atoms by residue ID (fast, works for ~96%
    of proteins). If too few matches are found (indicating inconsistent
    numbering), falls back to sequence alignment.

    Args:
        ca_ref:              CA atoms from reference structure.
        ca_tgt:              CA atoms from target structure.
        ligand_centroid_tgt: 3D centroid of the target's ligand.

    Returns:
        Tuple of (transformed_centroid, rmsd, n_matched, method) where:
            transformed_centroid: Ligand centroid in reference frame (or None).
            rmsd:    RMSD of matched CA atoms after alignment (quality metric).
            n_matched: Number of CA atoms used for alignment.
            method:  "resid" or "sequence" indicating which matching was used.
    """
    if not ca_ref or not ca_tgt or ligand_centroid_tgt is None:
        return None, None, 0, "failed"

    # Try residue ID matching first
    P_ref, P_tgt = match_ca_by_resid(ca_ref, ca_tgt)

    # The resid match threshold must be proportional to chain length.
    # A fixed threshold of 10 would accept 15/389 matches (3.9%) for
    # a large protein, producing unreliable alignments. We require at
    # least 30% of the shorter chain to match by resid; if fewer match,
    # the numbering is likely inconsistent and we fall back to sequence
    # alignment.
    min_chain_len = min(len(ca_ref), len(ca_tgt))
    resid_threshold = max(MIN_MATCHED_CA, int(0.3 * min_chain_len))

    method = "resid"
    if len(P_ref) < resid_threshold:
        # Fall back to sequence alignment
        P_ref, P_tgt = match_ca_by_sequence(ca_ref, ca_tgt)
        method = "sequence"

    if len(P_ref) < 3:
        return None, None, 0, "failed"

    # Compute optimal superposition
    R, t = kabsch(P_tgt, P_ref)

    # Transform the ligand centroid into the reference frame
    transformed = R @ ligand_centroid_tgt + t

    # Compute RMSD as alignment quality metric
    aligned_tgt = (R @ P_tgt.T).T + t
    rmsd = np.sqrt(np.mean(np.sum((aligned_tgt - P_ref) ** 2, axis=1)))

    return transformed, rmsd, len(P_ref), method


# ---------------------------------------------------------------------------
# Centroid-based pocket clustering
# ---------------------------------------------------------------------------

def cluster_centroids(centroids, pdb_codes):
    """
    Cluster ligand centroids by spatial proximity.

    After structural alignment, all ligand centroids are in the same
    coordinate frame. Ligands binding the same pocket will have centroids
    close together (< CENTROID_DISTANCE_THRESHOLD). Ligands in different
    pockets will be far apart (typically > 15 Angstroms).

    Uses agglomerative clustering with average linkage and a distance
    threshold cutoff, same approach as the previous residue-based method
    but operating on 3D Euclidean distances instead of Jaccard distances.

    Args:
        centroids: List of 3D centroid arrays (some may be None if
                   alignment failed for that entry).
        pdb_codes: Corresponding PDB codes.

    Returns:
        List of integer cluster labels (1-indexed), one per entry.
    """
    n = len(centroids)

    if n == 1:
        return [1]

    # Identify entries with valid centroids
    valid_indices = [i for i, c in enumerate(centroids) if c is not None]

    # If fewer than 2 valid centroids, can't cluster — assign all to pocket 1
    if len(valid_indices) < 2:
        return [1] * n

    # Cluster valid centroids
    valid_coords = np.array([centroids[i] for i in valid_indices])
    distances = pdist(valid_coords)

    Z = linkage(distances, method="average")
    valid_labels = fcluster(Z, t=CENTROID_DISTANCE_THRESHOLD, criterion="distance")

    # Build full label array. Entries with failed alignment get label 0
    # (meaning "unassignable") — these will be dropped from the dataset.
    labels = [0] * n
    for idx, vi in enumerate(valid_indices):
        labels[vi] = int(valid_labels[idx])

    return labels


# ---------------------------------------------------------------------------
# Property distribution computation
# ---------------------------------------------------------------------------

def compute_distribution(group_df):
    """
    Compute property distribution statistics for a group of ligands.

    For each of the 9 physicochemical properties, computes mean, std,
    median, min, max, and count (number of ligands with a valid value).

    Args:
        group_df: DataFrame subset containing ligands for one pocket.

    Returns:
        Dictionary of statistics, keyed like "MW_mean", "MW_std", etc.
    """
    stats = {}
    for prop in PROPERTY_COLUMNS:
        values = group_df[prop].dropna()
        stats[f"{prop}_count"] = len(values)
        if len(values) > 0:
            stats[f"{prop}_mean"] = values.mean()
            stats[f"{prop}_std"] = values.std() if len(values) > 1 else 0.0
            stats[f"{prop}_median"] = values.median()
            stats[f"{prop}_min"] = values.min()
            stats[f"{prop}_max"] = values.max()
        else:
            for suffix in ["mean", "std", "median", "min", "max"]:
                stats[f"{prop}_{suffix}"] = None
    return stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} entries from {INPUT_CSV.name}")

    # --- Step A: Extract ligand centroids and protein CA atoms ---
    print(f"\nExtracting ligand centroids and protein CA atoms...")
    ligand_centroids = {}   # pdb_code -> np.array([x, y, z]) or None
    protein_ca = {}         # pdb_code -> list of CA atom dicts

    for i, (_, row) in enumerate(df.iterrows()):
        code = row["pdb_code"]
        subdir = row["data_subdir"]
        chain = row["pocket_chain"]

        ligand_centroids[code] = get_ligand_centroid(code, subdir)
        protein_ca[code] = get_chain_ca_atoms(code, subdir, chain)

        if (i + 1) % 2000 == 0:
            print(f"  Processed {i + 1}/{len(df)} entries")

    n_centroids = sum(1 for v in ligand_centroids.values() if v is not None)
    n_ca = sum(1 for v in protein_ca.values() if len(v) > 0)
    print(f"  Ligand centroids extracted: {n_centroids}/{len(df)}")
    print(f"  Protein CA atoms extracted: {n_ca}/{len(df)}")

    # --- Step B: Align structures and cluster by centroid distance ---
    print(f"\nAligning structures and clustering per protein...")
    pocket_ids = {}          # pdb_code -> pocket_id string
    pocket_summary = []      # summary stats per protein

    grouped = df.groupby("uniprot_id")
    n_proteins = len(grouped)
    multi_pocket_count = 0
    alignment_stats = {"resid": 0, "sequence": 0, "failed": 0}

    for i, (uniprot_id, group) in enumerate(grouped):
        pdb_codes = group["pdb_code"].tolist()

        # Single entry — trivially one pocket
        if len(pdb_codes) == 1:
            pocket_ids[pdb_codes[0]] = f"{uniprot_id}_1"
            pocket_summary.append({
                "uniprot_id": uniprot_id,
                "protein_name": group.iloc[0]["protein_name"],
                "n_entries": 1,
                "n_pockets": 1,
                "pocket_sizes": "[1]",
            })
            continue

        # Pick reference: entry whose CA count is closest to the median.
        # This avoids fusion proteins and chimeras that have anomalously high
        # CA counts (e.g., a 556-residue fusion when the protein is 99 residues).
        # The median is robust to such outliers.
        ca_counts = [len(protein_ca.get(c, [])) for c in pdb_codes]
        median_ca = np.median(ca_counts)
        ref_code = min(pdb_codes, key=lambda c: abs(len(protein_ca.get(c, [])) - median_ca))
        ref_ca = protein_ca[ref_code]
        ref_centroid = ligand_centroids[ref_code]

        # Align all entries to the reference and collect centroids
        aligned_centroids = []
        for code in pdb_codes:
            if code == ref_code:
                # Reference is already in its own frame
                aligned_centroids.append(ref_centroid)
            else:
                tgt_ca = protein_ca[code]
                tgt_centroid = ligand_centroids[code]
                transformed, rmsd, n_matched, method = align_and_transform(
                    ref_ca, tgt_ca, tgt_centroid
                )

                # Quality gate: discard alignments with high RMSD.
                # High RMSD typically means the target is a fusion protein,
                # chimera, or multi-domain construct where only part of the
                # chain matches the reference. The transformed centroid
                # would be unreliable.
                if rmsd is not None and rmsd > MAX_ALIGNMENT_RMSD:
                    transformed = None
                    method = "high_rmsd"

                aligned_centroids.append(transformed)
                alignment_stats[method] = alignment_stats.get(method, 0) + 1

        # Cluster by centroid distance
        labels = cluster_centroids(aligned_centroids, pdb_codes)

        # Label 0 means the entry couldn't be aligned — mark for dropping
        for code, label in zip(pdb_codes, labels):
            if label == 0:
                pocket_ids[code] = None  # Will be dropped
            else:
                pocket_ids[code] = f"{uniprot_id}_{label}"

        n_pockets = max(labels)
        if n_pockets > 1:
            multi_pocket_count += 1

        # Count only assigned entries (label > 0) for pocket sizes
        assigned_labels = [l for l in labels if l > 0]
        n_assigned = len(assigned_labels)
        n_dropped_here = len(labels) - n_assigned

        pocket_summary.append({
            "uniprot_id": uniprot_id,
            "protein_name": group.iloc[0]["protein_name"],
            "n_entries": len(group),
            "n_assigned": n_assigned,
            "n_dropped": n_dropped_here,
            "n_pockets": n_pockets,
            "pocket_sizes": str(
                [assigned_labels.count(l) for l in sorted(set(assigned_labels))]
            ) if assigned_labels else "[]",
        })

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{n_proteins} proteins")

    print(f"  Processed {n_proteins}/{n_proteins} proteins")
    print(f"\n=== Clustering Results ===")
    print(f"  Total proteins: {n_proteins}")
    print(f"  Proteins with multiple pockets: {multi_pocket_count}")
    print(f"  Proteins with single pocket: {n_proteins - multi_pocket_count}")
    print(f"  Alignment method counts (across all pairwise alignments):")
    print(f"    By residue ID:       {alignment_stats.get('resid', 0)}")
    print(f"    By sequence:         {alignment_stats.get('sequence', 0)}")
    print(f"    Discarded (RMSD>{MAX_ALIGNMENT_RMSD}A): {alignment_stats.get('high_rmsd', 0)}")
    print(f"    Failed:              {alignment_stats.get('failed', 0)}")

    # --- Add pocket_id to the main DataFrame ---
    df["pocket_id"] = df["pdb_code"].map(pocket_ids)

    # Drop entries that couldn't be reliably aligned (label 0 → None).
    # These are fusion proteins, chimeras, or multi-domain constructs
    # where the pocket chain doesn't match the reference well enough
    # for structural alignment.
    n_dropped = df["pocket_id"].isna().sum()
    if n_dropped > 0:
        df = df.dropna(subset=["pocket_id"]).reset_index(drop=True)
        print(f"  Dropped {n_dropped} entries with unreliable alignment "
              f"({len(df)} remaining)")

    total_pockets = df["pocket_id"].nunique()
    print(f"  Total unique pockets: {total_pockets}")

    # --- Step C: Compute per-pocket property distributions ---
    print(f"\nComputing per-pocket property distributions...")
    distribution_rows = []

    for pocket_id, pocket_group in df.groupby("pocket_id"):
        stats = compute_distribution(pocket_group)

        # Add pocket metadata
        first = pocket_group.iloc[0]
        stats["pocket_id"] = pocket_id
        stats["uniprot_id"] = first["uniprot_id"]
        stats["protein_name"] = first["protein_name"]
        stats["organism"] = first["organism"]
        stats["classification"] = first["classification"]
        stats["n_ligands"] = len(pocket_group)
        stats["pdb_codes"] = ",".join(sorted(pocket_group["pdb_code"]))

        distribution_rows.append(stats)

    dist_df = pd.DataFrame(distribution_rows)

    # Reorder columns: metadata first, then property stats
    meta_cols = [
        "pocket_id", "uniprot_id", "protein_name", "organism",
        "classification", "n_ligands", "pdb_codes"
    ]
    prop_cols = []
    for prop in PROPERTY_COLUMNS:
        for suffix in ["mean", "std", "median", "min", "max", "count"]:
            prop_cols.append(f"{prop}_{suffix}")
    dist_df = dist_df[meta_cols + prop_cols]

    # --- Report distribution statistics ---
    print(f"\n=== Distribution Summary ===")
    print(f"  Total pockets: {len(dist_df)}")
    print(f"  Ligands per pocket: "
          f"mean={dist_df['n_ligands'].mean():.1f}, "
          f"median={dist_df['n_ligands'].median():.0f}, "
          f"min={dist_df['n_ligands'].min()}, "
          f"max={dist_df['n_ligands'].max()}")

    # Breakdown by ligand count
    for threshold in [1, 5, 10, 20, 50]:
        n = (dist_df["n_ligands"] >= threshold).sum()
        print(f"  Pockets with >= {threshold:2d} ligands: {n}")

    # Show top pockets
    print(f"\nTop 15 pockets by ligand count:")
    for _, row in dist_df.nlargest(15, "n_ligands").iterrows():
        print(f"  {row['pocket_id']:20s} | {row['n_ligands']:4d} ligands | "
              f"MW={row['MW_mean']:.0f}±{row['MW_std']:.0f} | "
              f"logP={row['logP_mean']:.1f}±{row['logP_std']:.1f} | "
              f"{row['protein_name'][:35]}")

    # --- Save outputs ---
    # 1. Enriched dataset with pocket_id
    pockets_path = OUTPUT_DIR / "pdbbind_with_pockets.csv"
    df.to_csv(pockets_path, index=False)
    print(f"\nSaved: {pockets_path} ({len(df)} rows, {df.shape[1]} columns)")

    # 2. Per-pocket distributions
    dist_path = OUTPUT_DIR / "pocket_distributions.csv"
    dist_df.to_csv(dist_path, index=False)
    print(f"Saved: {dist_path} ({len(dist_df)} rows, {dist_df.shape[1]} columns)")

    # 3. Pocket clustering summary
    summary_df = pd.DataFrame(pocket_summary).sort_values(
        "n_pockets", ascending=False
    )
    summary_path = OUTPUT_DIR / "pocket_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path} ({len(summary_df)} rows)")


if __name__ == "__main__":
    main()
