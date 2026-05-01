"""
Step 2: Compute physicochemical properties for all ligands.

This script reads the filtered PDBbind dataset (pdbbind_filtered.csv) and
computes 9 physicochemical descriptors for each ligand using RDKit. These
are the properties our model will eventually learn to predict from pocket
geometry.

Input:
    ../data/pdbbind_filtered.csv  (produced by parse_pdbbind_index.py)

    For each entry, the corresponding ligand structure file is loaded from:
    ~/Desktop/datasets/PDBbind/P-L/{data_subdir}/{pdb_code}/{pdb_code}_ligand.sdf

Output:
    ../data/pdbbind_with_properties.csv
        The filtered dataset with 9 new property columns appended.

    ../data/ligand_parse_failures.txt  (if any failures)
        List of PDB codes whose ligand files could not be parsed by RDKit.

Properties computed:
    MW              - Molecular weight (Daltons). Larger molecules fill larger pockets.
    logP            - Partition coefficient (Crippen method). Measures hydrophobicity:
                      positive = hydrophobic, negative = hydrophilic. Must complement
                      the pocket's polarity.
    HBD             - Hydrogen bond donor count. Number of NH and OH groups that can
                      donate hydrogen bonds to the protein.
    HBA             - Hydrogen bond acceptor count. Number of O and N atoms that can
                      accept hydrogen bonds from the protein.
    TPSA            - Topological polar surface area (Angstroms^2). Total surface area
                      of polar atoms (N, O, S). Related to solubility and permeability.
    rotatable_bonds - Number of freely rotating single bonds. Measures molecular
                      flexibility — rigid pockets favor rigid ligands.
    formal_charge   - Net integer charge on the molecule. Must complement pocket
                      electrostatics (e.g., a negatively charged pocket favors
                      positively charged ligands).
    aromatic_rings  - Count of aromatic ring systems. Related to pi-stacking
                      interactions with aromatic residues (Phe, Tyr, Trp, His).
    fsp3            - Fraction of sp3-hybridized carbon atoms (0 to 1). Measures
                      3D complexity: 0 = fully flat/aromatic, 1 = fully tetrahedral.

Dependencies:
    pip install rdkit pandas
"""

import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PDBBIND_DIR = Path("/Users/alannadels/Desktop/datasets/PDBbind/P-L")
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"
FILTERED_CSV = OUTPUT_DIR / "pdbbind_filtered.csv"


# ---------------------------------------------------------------------------
# Property computation
# ---------------------------------------------------------------------------

def compute_properties(mol):
    """
    Compute all 9 physicochemical descriptors for an RDKit molecule object.

    These descriptors capture complementary aspects of a molecule's drug-likeness
    and binding behavior. Together, they define the "physicochemical profile" that
    our model will learn to predict from pocket geometry.

    Args:
        mol: An RDKit Mol object (with or without explicit hydrogens).

    Returns:
        Dictionary mapping property names to computed values.
    """
    return {
        "MW": Descriptors.MolWt(mol),                     # Molecular weight in Daltons
        "logP": Crippen.MolLogP(mol),                     # Crippen logP (hydrophobicity)
        "HBD": Descriptors.NumHDonors(mol),               # H-bond donors (NH, OH groups)
        "HBA": Descriptors.NumHAcceptors(mol),             # H-bond acceptors (N, O atoms)
        "TPSA": Descriptors.TPSA(mol),                    # Topological polar surface area
        "rotatable_bonds": Descriptors.NumRotatableBonds(mol),  # Rotatable bond count
        "formal_charge": Chem.GetFormalCharge(mol),        # Net formal charge
        "aromatic_rings": Descriptors.NumAromaticRings(mol),    # Aromatic ring count
        "fsp3": Descriptors.FractionCSP3(mol),            # Fraction sp3 carbons
    }


def load_ligand(pdb_code, data_subdir):
    """
    Load a ligand molecule from PDBbind structure files.

    Attempts to read the .sdf file first (preferred format for RDKit), and
    falls back to the .mol2 file if the SDF fails to parse. Both formats
    contain 3D coordinates and atom/bond information.

    The 'sanitize=True' flag tells RDKit to validate the chemical structure
    (check valences, aromaticity, etc.). 'removeHs=False' keeps explicit
    hydrogens, which are needed for accurate H-bond donor/acceptor counting.

    Args:
        pdb_code:    4-character PDB identifier, e.g. "1hsg"
        data_subdir: Time-period folder name, e.g. "1981-2000"

    Returns:
        RDKit Mol object, or None if both formats fail to parse.
    """
    # Try SDF format first (more reliable parsing in RDKit)
    sdf_path = PDBBIND_DIR / data_subdir / pdb_code / f"{pdb_code}_ligand.sdf"
    if sdf_path.exists():
        # SDMolSupplier can read multi-molecule SDF files; we take the first valid one
        supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=True, removeHs=False)
        for mol in supplier:
            if mol is not None:
                return mol

    # Fallback to MOL2 format (Tripos format, includes atom types and charges)
    mol2_path = PDBBIND_DIR / data_subdir / pdb_code / f"{pdb_code}_ligand.mol2"
    if mol2_path.exists():
        mol = Chem.MolFromMol2File(str(mol2_path), sanitize=True, removeHs=False)
        if mol is not None:
            return mol

    return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    df = pd.read_csv(FILTERED_CSV)
    print(f"Loaded {len(df)} entries from filtered dataset")

    # Process each ligand: load structure file -> compute properties
    results = []    # List of property dicts, one per entry
    failed = []     # PDB codes that couldn't be parsed

    for i, row in df.iterrows():
        pdb_code = row["pdb_code"]
        data_subdir = row["data_subdir"]

        mol = load_ligand(pdb_code, data_subdir)

        if mol is None:
            # Record failure — append None values so the row count stays aligned
            failed.append(pdb_code)
            results.append({prop: None for prop in [
                "MW", "logP", "HBD", "HBA", "TPSA",
                "rotatable_bonds", "formal_charge", "aromatic_rings", "fsp3"
            ]})
        else:
            results.append(compute_properties(mol))

        # Progress reporting every 2000 entries
        if (i + 1) % 2000 == 0:
            print(f"  Processed {i + 1}/{len(df)} ligands "
                  f"({len(failed)} failures so far)")

    # --- Merge property columns into the original DataFrame ---
    props_df = pd.DataFrame(results)
    merged = pd.concat([df, props_df], axis=1)

    # --- Report results ---
    n_success = merged["MW"].notna().sum()
    print(f"\nCompleted: {n_success}/{len(merged)} ligands successfully processed")
    print(f"Failed to parse: {len(failed)} ligands")

    if failed:
        print(f"  Failed PDB codes: {failed[:20]}")

    # Print summary statistics for each computed property
    print(f"\n=== Property Statistics ===")
    for prop in ["MW", "logP", "HBD", "HBA", "TPSA", "rotatable_bonds",
                 "formal_charge", "aromatic_rings", "fsp3"]:
        col = merged[prop].dropna()
        print(f"  {prop:>16s}: mean={col.mean():.2f}, std={col.std():.2f}, "
              f"min={col.min():.2f}, max={col.max():.2f}")

    # --- Save outputs ---
    output_path = OUTPUT_DIR / "pdbbind_with_properties.csv"
    merged.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path} ({len(merged)} rows, {merged.shape[1]} columns)")

    # Save list of PDB codes that failed to parse (for debugging/investigation)
    if failed:
        failed_path = OUTPUT_DIR / "ligand_parse_failures.txt"
        with open(failed_path, "w") as f:
            f.write("\n".join(failed))
        print(f"Saved failure list: {failed_path}")


if __name__ == "__main__":
    main()
