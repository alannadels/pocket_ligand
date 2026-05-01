"""
Step 1: Parse the PDBbind v.2020R1 protein-ligand index file.

This script reads the raw PDBbind index file (INDEX_general_PL.2020R1.lst),
which contains one line per protein-ligand complex with binding affinity data.
It extracts structured fields from each line and applies quality filters to
produce a clean dataset for downstream analysis.

Input:
    PDBbind index file at:
    ~/Desktop/datasets/PDBbind/index/INDEX_general_PL.2020R1.lst

    Each line in the index file has the format:
    PDB_code  resolution  year  affinity  // reference.pdf (ligand_name) [notes]
    Example: 1hsg  2.00  1998  Ki=0.4nM  // 1hsg.pdf (MK1)

Outputs (saved to ../data/):
    - pdbbind_all.csv:      All 19,037 entries, unfiltered
    - pdbbind_filtered.csv: ~10,692 entries after quality filtering

Quality filters applied (see documentation/quality_filters.md for details):
    1. Resolution <= 2.5 Angstroms (excludes NMR structures)
    2. No covalent complexes
    3. No incomplete ligand structures
    4. Exact affinity measurements only (operator must be '=')
    5. Binding affinity <= 10,000 nM (10 uM)
"""

import re
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INDEX_FILE = Path("/Users/alannadels/Desktop/datasets/PDBbind/index/INDEX_general_PL.2020R1.lst")
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Multiplication factors to convert each unit to nanomolar (nM).
# Example: 1 uM = 1,000 nM, so the factor is 1e3.
UNIT_TO_NM = {
    "fM": 1e-6,   # femtomolar  (10^-15 M)
    "pM": 1e-3,   # picomolar   (10^-12 M)
    "nM": 1.0,    # nanomolar   (10^-9 M)  -- base unit
    "uM": 1e3,    # micromolar  (10^-6 M)
    "mM": 1e6,    # millimolar  (10^-3 M)
}


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_affinity(affinity_str):
    """
    Parse an affinity string into its components.

    The PDBbind index encodes binding affinity as a single token like:
        Ki=0.4nM, Kd<10uM, IC50~500nM, EC50>=1uM

    This function extracts:
        - affinity_type: Kd, Ki, IC50, or EC50
        - operator: '=', '<', '>', '<=', '>=', or '~'
        - value in nanomolar: the numeric value converted to a common unit

    Args:
        affinity_str: Raw affinity token, e.g. "Ki=0.4nM"

    Returns:
        Tuple of (affinity_type, operator, value_in_nM).
        Returns (None, None, None) if the string can't be parsed.
    """
    # Regex captures: (type)(operator)(number)(unit)
    # Examples matched: Ki=0.4nM, Kd<10uM, IC50<=3.5nM, EC50~1uM
    match = re.match(
        r"(Kd|Ki|IC50|EC50)"   # Affinity type
        r"(<=?|>=?|~|=)"       # Comparison operator
        r"([0-9.]+)"           # Numeric value
        r"(fM|pM|nM|uM|mM)",  # Concentration unit
        affinity_str
    )
    if not match:
        return None, None, None

    aff_type, operator, value, unit = match.groups()

    # Convert to nanomolar for consistent comparison across entries
    value_nm = float(value) * UNIT_TO_NM[unit]

    return aff_type, operator, value_nm


def parse_ligand_name(reference_part):
    """
    Extract the ligand name from the reference section of an index line.

    The reference section appears after '//' and contains the ligand name
    in parentheses. Examples:
        "1hsg.pdf (MK1)"              -> "MK1"
        "6apr.pdf (6-mer)"            -> "6-mer"
        "8abp.pdf (GLA/GLB) isomer"   -> "GLA/GLB"

    Args:
        reference_part: Everything after '//' in the index line.

    Returns:
        Ligand name string, or None if no parenthesized name found.
    """
    match = re.search(r"\(([^)]+)\)", reference_part)
    return match.group(1) if match else None


def parse_flags(line):
    """
    Detect quality flags from the full index line.

    PDBbind annotates problematic entries with text notes at the end of
    the line. We scan for three known flags:

        - "covalent complex": ligand is covalently bonded to the protein
          (irreversible binding — different mechanism than what we model)
        - "incomplete ligand": some ligand atoms are missing from the
          crystal structure (can't compute accurate properties)
        - "isomer": ligand has isomeric forms noted (kept for reference,
          not filtered on)

    Args:
        line: Full text line from the index file.

    Returns:
        List of flag strings found (e.g. ["covalent", "incomplete_ligand"]).
    """
    flags = []
    lower = line.lower()
    if "covalent complex" in lower or "covalent" in lower:
        flags.append("covalent")
    if "incomplete ligand" in lower:
        flags.append("incomplete_ligand")
    if "isomer" in lower:
        flags.append("isomer")
    return flags


# ---------------------------------------------------------------------------
# Main parsing logic
# ---------------------------------------------------------------------------

def parse_index(filepath):
    """
    Parse the entire PDBbind index file into a pandas DataFrame.

    Reads the file line by line, skipping comment lines (starting with '#').
    Each data line is split on '//' to separate the structured data fields
    (PDB code, resolution, year, affinity) from the reference/notes section.

    Args:
        filepath: Path to INDEX_general_PL.2020R1.lst

    Returns:
        DataFrame with one row per protein-ligand complex and columns:
        pdb_code, resolution, is_nmr, year, affinity_type, affinity_operator,
        affinity_nM, affinity_raw, ligand_name, is_covalent, is_incomplete,
        is_isomer, reference
    """
    records = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()

            # Skip empty lines and comment headers (first 6 lines of the file)
            if not line or line.startswith("#"):
                continue

            # --- Split line into data vs. reference sections ---
            # Left of '//' : PDB code, resolution, year, affinity
            # Right of '//': PDF reference, ligand name in parens, notes
            parts = line.split("//", 1)
            data_part = parts[0].strip()
            reference_part = parts[1].strip() if len(parts) > 1 else ""

            # --- Parse the whitespace-separated data fields ---
            tokens = data_part.split()
            if len(tokens) < 4:
                continue  # Malformed line, skip

            pdb_code = tokens[0]          # e.g. "1hsg"
            resolution_str = tokens[1]     # e.g. "2.00" or "NMR"
            year = int(tokens[2])          # e.g. 1998
            affinity_str = tokens[3]       # e.g. "Ki=0.4nM"

            # --- Handle resolution ---
            # NMR-solved structures don't have a numeric resolution value.
            # We store None for resolution and flag them separately.
            if resolution_str == "NMR":
                resolution = None
                is_nmr = True
            else:
                resolution = float(resolution_str)
                is_nmr = False

            # --- Parse affinity into structured components ---
            aff_type, aff_operator, aff_value_nm = parse_affinity(affinity_str)

            # --- Extract ligand name and quality flags ---
            ligand_name = parse_ligand_name(reference_part)
            flags = parse_flags(line)

            records.append({
                "pdb_code": pdb_code,
                "resolution": resolution,
                "is_nmr": is_nmr,
                "year": year,
                "affinity_type": aff_type,
                "affinity_operator": aff_operator,
                "affinity_nM": aff_value_nm,
                "affinity_raw": affinity_str,
                "ligand_name": ligand_name,
                "is_covalent": "covalent" in flags,
                "is_incomplete": "incomplete_ligand" in flags,
                "is_isomer": "isomer" in flags,
                "reference": reference_part,
            })

    return pd.DataFrame(records)


def apply_quality_filters(df):
    """
    Apply quality filters to keep only high-confidence entries.

    Starting from the full 19,037-entry dataset, this function applies five
    filters (combined with AND logic) to remove entries that would introduce
    noise or errors into downstream analysis. See documentation/quality_filters.md
    for detailed rationale behind each filter.

    Args:
        df: Raw DataFrame from parse_index().

    Returns:
        Filtered DataFrame with reset index.
    """
    initial = len(df)

    # Filter 1: Resolution <= 2.5 Angstroms
    # Structures with worse resolution have unreliable atom positions.
    # NMR structures (resolution=None) are also excluded here.
    mask = df["resolution"].notna() & (df["resolution"] <= 2.5)

    # Filter 2: No covalent complexes
    # Covalent binders form permanent bonds — different binding mechanism.
    mask &= ~df["is_covalent"]

    # Filter 3: No incomplete ligand structures
    # Missing atoms mean we can't compute accurate physicochemical properties.
    mask &= ~df["is_incomplete"]

    # Filter 4: Affinity must be parseable
    # A small number of entries have unusual affinity formats we can't parse.
    mask &= df["affinity_nM"].notna()

    # Filter 5: Exact affinity <= 10,000 nM (10 uM)
    # Two sub-conditions:
    #   a) Operator must be '=' (not '<', '>', '~', '<=' — these are imprecise)
    #   b) Value must be <= 10,000 nM (weaker binders are likely non-specific)
    mask &= (df["affinity_operator"] == "=") & (df["affinity_nM"] <= 10_000)

    filtered = df[mask].copy().reset_index(drop=True)

    # Print filtering summary
    print(f"Quality filtering: {initial} -> {len(filtered)} entries")
    print(f"  Removed by resolution > 2.5 or NMR: "
          f"{initial - (df['resolution'].notna() & (df['resolution'] <= 2.5)).sum()}")
    print(f"  Covalent complexes: {df['is_covalent'].sum()}")
    print(f"  Incomplete ligands: {df['is_incomplete'].sum()}")
    print(f"  Unparseable affinity: {df['affinity_nM'].isna().sum()}")
    print(f"  Affinity > 10 uM or inexact: "
          f"{((df['affinity_operator'] != '=') | (df['affinity_nM'] > 10_000)).sum()}")

    return filtered


def determine_data_dir(pdb_code, year):
    """
    Determine which PDBbind subdirectory contains this entry's structure files.

    PDBbind organizes structure files into three time-period folders:
        P-L/1981-2000/  (1,111 complexes)
        P-L/2001-2010/  (5,691 complexes)
        P-L/2011-2019/  (12,235 complexes)

    Args:
        pdb_code: 4-character PDB identifier (unused, but kept for API consistency).
        year: Release year of the PDB structure.

    Returns:
        Subdirectory name string, e.g. "2001-2010".
    """
    if year <= 2000:
        return "1981-2000"
    elif year <= 2010:
        return "2001-2010"
    else:
        return "2011-2019"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print(f"Parsing: {INDEX_FILE}")
    df = parse_index(INDEX_FILE)

    # --- Report raw dataset statistics ---
    print(f"\n=== Raw Dataset ===")
    print(f"Total entries: {len(df)}")
    print(f"Year range: {df['year'].min()} - {df['year'].max()}")
    print(f"Affinity types: {df['affinity_type'].value_counts().to_dict()}")
    print(f"NMR structures: {df['is_nmr'].sum()}")
    print(f"Covalent complexes: {df['is_covalent'].sum()}")
    print(f"Incomplete ligands: {df['is_incomplete'].sum()}")
    print(f"Resolution stats (non-NMR):")
    print(f"  {df['resolution'].describe().to_string()}")

    # --- Apply quality filters ---
    print(f"\n=== Applying Quality Filters ===")
    df_filtered = apply_quality_filters(df)

    # --- Report filtered dataset statistics ---
    print(f"\n=== Filtered Dataset ===")
    print(f"Total entries: {len(df_filtered)}")
    print(f"Affinity types: {df_filtered['affinity_type'].value_counts().to_dict()}")
    print(f"Affinity range (nM): {df_filtered['affinity_nM'].min():.4f} - "
          f"{df_filtered['affinity_nM'].max():.1f}")
    print(f"Resolution range: {df_filtered['resolution'].min():.2f} - "
          f"{df_filtered['resolution'].max():.2f}")

    # --- Add data directory path for locating structure files ---
    df["data_subdir"] = df.apply(
        lambda r: determine_data_dir(r["pdb_code"], r["year"]), axis=1
    )
    df_filtered["data_subdir"] = df_filtered.apply(
        lambda r: determine_data_dir(r["pdb_code"], r["year"]), axis=1
    )

    # --- Save outputs ---
    raw_path = OUTPUT_DIR / "pdbbind_all.csv"
    filtered_path = OUTPUT_DIR / "pdbbind_filtered.csv"

    df.to_csv(raw_path, index=False)
    df_filtered.to_csv(filtered_path, index=False)

    print(f"\nSaved raw dataset: {raw_path} ({len(df)} rows)")
    print(f"Saved filtered dataset: {filtered_path} ({len(df_filtered)} rows)")


if __name__ == "__main__":
    main()
