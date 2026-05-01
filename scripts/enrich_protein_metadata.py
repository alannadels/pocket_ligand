"""
Step 3: Enrich dataset with protein metadata from the RCSB PDB API.

This script takes our filtered dataset (with ligand properties) and queries the
RCSB PDB's GraphQL API to add protein-level metadata for each entry. This is
critical for grouping entries by unique protein — the same protein (e.g., HIV
protease) appears many times in PDBbind, each co-crystallized with a different
ligand. We need to know which entries belong to the same protein so we can pool
their ligands and compute per-pocket property distributions.

Entity selection strategy (spatial check):
    A PDB entry can contain multiple polymer entities (protein chains). For
    example, 1G30 contains thrombin (chains A,B) and hirudin (chain C). The
    ligand binds to thrombin, not hirudin. To correctly identify which entity
    the ligand binds, we:

    1. Parse the pocket PDB file (pre-extracted by PDBbind as atoms near the
       ligand) to find which chain(s) form the binding pocket.
    2. The most frequent chain in the pocket file is the chain the ligand
       actually contacts.
    3. We match that chain to the correct polymer entity from the RCSB API,
       which gives us the right UniProt ID, protein name, and organism.

    This is more accurate than simply picking the first or longest entity,
    which can fail for antibody-antigen complexes, multi-domain structures,
    or entries where a small peptide inhibitor is listed as entity 1.

    Fallback: If the pocket file can't be parsed or the chain doesn't match
    any entity, we fall back to the longest entity by sequence length.

Input:
    ../data/pdbbind_with_properties.csv  (produced by compute_ligand_properties.py)

Output:
    ../data/pdbbind_enriched.csv
        The dataset with 6 new columns: uniprot_id, protein_name, organism,
        classification, struct_title, and pocket_chain.

    ../data/protein_groups.csv
        Summary table: one row per unique UniProt ID, showing how many
        PDBbind entries (structures/ligands) exist for that protein.

API used:
    RCSB PDB GraphQL endpoint: https://data.rcsb.org/graphql
    We use GraphQL because it supports batch queries — we can request metadata
    for up to ~50 PDB codes in a single request, avoiding 10,000+ individual
    REST API calls.

Dependencies:
    pip install requests pandas
"""

import collections
import requests
import pandas as pd
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"
INPUT_CSV = OUTPUT_DIR / "pdbbind_with_properties.csv"
PDBBIND_DIR = Path("/Users/alannadels/Desktop/datasets/PDBbind/P-L")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# RCSB PDB GraphQL endpoint — supports batch queries for multiple PDB codes
GRAPHQL_URL = "https://data.rcsb.org/graphql"

# Number of PDB codes to request per GraphQL query.
# The API can handle large batches, but we keep it moderate to avoid
# timeouts and to get progress updates.
BATCH_SIZE = 50

# Seconds to wait between batches to be respectful to the API
DELAY_BETWEEN_BATCHES = 0.5

# ---------------------------------------------------------------------------
# GraphQL query template
# ---------------------------------------------------------------------------

# This query retrieves protein metadata for a list of PDB codes.
# Each PDB entry can have multiple "polymer entities" (protein chains).
# We request pdbx_strand_id (chain letters) and sequence length for each
# entity so we can match entities to the pocket's chain and fall back
# to the longest entity when spatial matching fails.
GRAPHQL_QUERY = """
{
  entries(entry_ids: %s) {
    rcsb_id
    struct {
      title
    }
    struct_keywords {
      pdbx_keywords
    }
    polymer_entities {
      rcsb_polymer_entity {
        pdbx_description
      }
      rcsb_entity_source_organism {
        ncbi_scientific_name
      }
      rcsb_polymer_entity_container_identifiers {
        uniprot_ids
      }
      entity_poly {
        pdbx_strand_id
        rcsb_sample_sequence_length
      }
    }
  }
}
"""


# ---------------------------------------------------------------------------
# Pocket chain extraction
# ---------------------------------------------------------------------------

def get_pocket_chain(pdb_code, data_subdir):
    """
    Parse a pocket PDB file to determine which chain the ligand binds to.

    PDBbind provides pre-extracted pocket files (*_pocket.pdb) containing
    all protein atoms within ~5 Angstroms of the co-crystallized ligand.
    By counting which chain ID appears most frequently among these atoms,
    we identify the protein chain that forms the binding pocket.

    PDB format column 22 (0-indexed: 21) contains the chain identifier.
    We ignore blank chain IDs (sometimes used for ligand/solvent atoms).

    Args:
        pdb_code:    4-character PDB identifier, e.g. "1hsg"
        data_subdir: Time-period folder name, e.g. "1981-2000"

    Returns:
        The most common chain letter (e.g., "A", "B"), or None if the
        pocket file doesn't exist or can't be parsed.
    """
    pocket_path = PDBBIND_DIR / data_subdir / pdb_code / f"{pdb_code}_pocket.pdb"

    if not pocket_path.exists():
        return None

    chain_counts = collections.Counter()
    with open(pocket_path, "r") as f:
        for line in f:
            # ATOM and HETATM records contain coordinate data
            if line.startswith("ATOM") or line.startswith("HETATM"):
                chain = line[21]  # Column 22 in PDB format (1-indexed)
                if chain.strip():  # Ignore blank chain IDs
                    chain_counts[chain] += 1

    if not chain_counts:
        return None

    # Return the chain with the most atoms in the pocket
    return chain_counts.most_common(1)[0][0]


# ---------------------------------------------------------------------------
# API query logic
# ---------------------------------------------------------------------------

def query_batch(pdb_codes):
    """
    Query the RCSB PDB GraphQL API for metadata on a batch of PDB codes.

    Returns ALL polymer entities per entry (with their chain IDs and sequence
    lengths) so that the caller can select the correct entity using spatial
    matching against the pocket file.

    Args:
        pdb_codes: List of 4-character PDB identifiers (e.g., ["1hsg", "1a42"]).

    Returns:
        Dictionary mapping each PDB code (lowercase) to a dict with keys:
        - classification: Protein family keyword
        - struct_title: Full PDB deposition title
        - entities: List of dicts, each with keys: chains (set of single
          letters), seq_length, protein_name, organism, uniprot_id
    """
    # Format PDB codes as a JSON array string for the GraphQL query
    # PDB codes must be uppercase for the API
    codes_str = str([code.upper() for code in pdb_codes]).replace("'", '"')
    query = GRAPHQL_QUERY % codes_str

    try:
        response = requests.post(GRAPHQL_URL, json={"query": query}, timeout=30)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as e:
        print(f"  API error: {e}")
        return {code: _empty_api_result() for code in pdb_codes}

    results = {}
    entries = data.get("data", {}).get("entries", [])

    for entry in entries:
        if entry is None:
            continue

        pdb_id = entry["rcsb_id"].lower()
        struct_title = entry.get("struct", {}).get("title", "")
        classification = (entry.get("struct_keywords", {}) or {}).get("pdbx_keywords", "")

        # Parse all polymer entities with their chain assignments
        entity_list = []
        for ent in entry.get("polymer_entities", []):
            # Chain IDs: "A,B" -> {"A", "B"}
            poly = ent.get("entity_poly") or {}
            strand_str = poly.get("pdbx_strand_id", "")
            chains = set(c.strip() for c in strand_str.split(",") if c.strip())
            seq_length = poly.get("rcsb_sample_sequence_length") or 0

            # Protein name
            protein_name = (ent.get("rcsb_polymer_entity", {}) or {}).get(
                "pdbx_description", ""
            )

            # Organism
            orgs = ent.get("rcsb_entity_source_organism") or []
            organism = orgs[0].get("ncbi_scientific_name", "") if orgs else ""

            # UniProt ID
            identifiers = (
                ent.get("rcsb_polymer_entity_container_identifiers", {}) or {}
            )
            uniprot_ids = identifiers.get("uniprot_ids") or []
            uniprot_id = uniprot_ids[0] if uniprot_ids else ""

            entity_list.append({
                "chains": chains,
                "seq_length": seq_length,
                "protein_name": protein_name,
                "organism": organism,
                "uniprot_id": uniprot_id,
            })

        results[pdb_id] = {
            "classification": classification,
            "struct_title": struct_title,
            "entities": entity_list,
        }

    # Fill in missing PDB codes
    for code in pdb_codes:
        if code not in results:
            results[code] = _empty_api_result()

    return results


def _empty_api_result():
    """Return an empty API result for PDB codes that fail to resolve."""
    return {
        "classification": "",
        "struct_title": "",
        "entities": [],
    }


def select_entity(api_result, pocket_chain):
    """
    Select the correct polymer entity for a given pocket chain.

    Strategy:
    1. SPATIAL MATCH (preferred): Find the entity whose chain set contains
       the pocket's dominant chain. This is the most reliable method because
       it uses the actual 3D structure to determine which protein chain the
       ligand contacts.

    2. LONGEST ENTITY (fallback): If no spatial match is found (pocket chain
       is None, or doesn't match any entity's chains), select the entity with
       the longest sequence. This handles cases where the pocket file is
       missing or the chain IDs don't align.

    Args:
        api_result: Dict from query_batch() with 'entities' list.
        pocket_chain: Single chain letter from get_pocket_chain(), or None.

    Returns:
        Dict with keys: uniprot_id, protein_name, organism, matched_by
        (either "spatial" or "longest").
    """
    entities = api_result.get("entities", [])

    if not entities:
        return {"uniprot_id": "", "protein_name": "", "organism": "", "matched_by": "none"}

    # Strategy 1: Spatial match — find entity containing the pocket chain
    if pocket_chain:
        for ent in entities:
            if pocket_chain in ent["chains"]:
                return {
                    "uniprot_id": ent["uniprot_id"],
                    "protein_name": ent["protein_name"],
                    "organism": ent["organism"],
                    "matched_by": "spatial",
                }

    # Strategy 2: Fallback to longest entity by sequence length
    longest = max(entities, key=lambda e: e["seq_length"])
    return {
        "uniprot_id": longest["uniprot_id"],
        "protein_name": longest["protein_name"],
        "organism": longest["organism"],
        "matched_by": "longest",
    }


def query_all(pdb_codes):
    """
    Query metadata for all PDB codes, processing in batches.

    Splits the full list into batches of BATCH_SIZE, queries each batch
    via GraphQL, and merges all results. Includes progress reporting and
    a small delay between batches to avoid overwhelming the API.

    Args:
        pdb_codes: List of all PDB codes to query.

    Returns:
        Dictionary mapping every PDB code to its full API result dict.
    """
    all_results = {}
    total = len(pdb_codes)

    for i in range(0, total, BATCH_SIZE):
        batch = pdb_codes[i : i + BATCH_SIZE]
        batch_results = query_batch(batch)
        all_results.update(batch_results)

        # Progress reporting
        done = min(i + BATCH_SIZE, total)
        if done % 500 == 0 or done == total:
            print(f"  Queried {done}/{total} PDB codes")

        # Rate limiting — small delay between batches
        if i + BATCH_SIZE < total:
            time.sleep(DELAY_BETWEEN_BATCHES)

    return all_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    # Load the dataset with computed ligand properties
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} entries from {INPUT_CSV.name}")

    # Get unique PDB codes to minimize API calls
    unique_codes = df["pdb_code"].unique().tolist()
    print(f"Unique PDB codes to query: {len(unique_codes)}")

    # --- Step A: Extract pocket chains from PDBbind structure files ---
    print(f"\nExtracting pocket chains from PDBbind pocket files...")
    pocket_chains = {}
    for _, row in df.iterrows():
        pocket_chains[row["pdb_code"]] = get_pocket_chain(
            row["pdb_code"], row["data_subdir"]
        )
    n_with_chain = sum(1 for v in pocket_chains.values() if v is not None)
    print(f"  Pocket chain extracted for {n_with_chain}/{len(pocket_chains)} entries")

    # --- Step B: Query RCSB PDB API for all entity metadata ---
    print(f"\nQuerying RCSB PDB API (batch size={BATCH_SIZE})...")
    api_results = query_all(unique_codes)

    # --- Step C: Match each entry to the correct entity ---
    # For each PDB code, use the pocket chain to select the right entity,
    # falling back to longest entity if spatial matching fails.
    print(f"\nMatching pocket chains to protein entities...")

    uniprot_ids = []
    protein_names = []
    organisms = []
    classifications = []
    struct_titles = []
    pocket_chain_col = []
    match_methods = []

    for _, row in df.iterrows():
        pdb_code = row["pdb_code"]
        api_result = api_results.get(pdb_code, _empty_api_result())
        chain = pocket_chains.get(pdb_code)

        # Select the correct entity using spatial match or fallback
        selected = select_entity(api_result, chain)

        uniprot_ids.append(selected["uniprot_id"])
        protein_names.append(selected["protein_name"])
        organisms.append(selected["organism"])
        classifications.append(api_result["classification"])
        struct_titles.append(api_result["struct_title"])
        pocket_chain_col.append(chain or "")
        match_methods.append(selected["matched_by"])

    df["uniprot_id"] = uniprot_ids
    df["protein_name"] = protein_names
    df["organism"] = organisms
    df["classification"] = classifications
    df["struct_title"] = struct_titles
    df["pocket_chain"] = pocket_chain_col

    # --- Report matching results ---
    method_counts = collections.Counter(match_methods)
    print(f"\n=== Entity Matching Results ===")
    print(f"  Spatial match (pocket chain -> entity): {method_counts.get('spatial', 0)}")
    print(f"  Longest entity fallback:                {method_counts.get('longest', 0)}")
    print(f"  No entities found:                      {method_counts.get('none', 0)}")

    # --- Drop entries without a UniProt ID ---
    # These are primarily engineered antibodies/immunoglobulins (~116),
    # de novo designed proteins (~9), and other synthetic constructs that
    # don't exist in UniProt. They're not representative drug targets and
    # can't be grouped by protein identity, so we exclude them.
    n_before = len(df)
    n_no_uniprot = df["uniprot_id"].isna().sum() + (df["uniprot_id"] == "").sum()
    df = df[df["uniprot_id"].notna() & (df["uniprot_id"] != "")].reset_index(drop=True)
    print(f"\n=== Dropping entries without UniProt ID ===")
    print(f"  Removed: {n_no_uniprot} entries (mostly antibodies/synthetic constructs)")
    print(f"  Remaining: {len(df)} entries")

    n_unique_proteins = df["uniprot_id"].nunique()

    print(f"\n=== Enrichment Results ===")
    print(f"Entries with UniProt ID: {len(df)}/{n_before} "
          f"({100 * len(df) / n_before:.1f}%)")
    print(f"Unique proteins (UniProt IDs): {n_unique_proteins}")

    # Top classifications
    print(f"\nTop 10 protein classifications:")
    class_counts = df.loc[df["classification"] != "", "classification"].value_counts().head(10)
    for cls, count in class_counts.items():
        print(f"  {cls}: {count}")

    # Top organisms
    print(f"\nTop 10 organisms:")
    org_counts = df.loc[df["organism"] != "", "organism"].value_counts().head(10)
    for org, count in org_counts.items():
        print(f"  {org}: {count}")

    # --- Build protein groups summary ---
    # This shows how many PDBbind entries (i.e., how many different ligands)
    # exist for each unique protein. Proteins with many ligands are the most
    # valuable training examples because they give richer property distributions.
    grouped = (
        df.groupby("uniprot_id")
        .agg(
            protein_name=("protein_name", "first"),
            organism=("organism", "first"),
            classification=("classification", "first"),
            n_entries=("pdb_code", "count"),
            pdb_codes=("pdb_code", lambda x: ",".join(sorted(x))),
        )
        .sort_values("n_entries", ascending=False)
        .reset_index()
    )

    print(f"\nTop 20 proteins by number of ligands:")
    for _, row in grouped.head(20).iterrows():
        print(f"  {row['uniprot_id']:10s} | {row['n_entries']:4d} ligands | "
              f"{row['protein_name'][:50]}")

    # --- Save outputs ---
    enriched_path = OUTPUT_DIR / "pdbbind_enriched.csv"
    groups_path = OUTPUT_DIR / "protein_groups.csv"

    df.to_csv(enriched_path, index=False)
    grouped.to_csv(groups_path, index=False)

    print(f"\nSaved enriched dataset: {enriched_path} ({len(df)} rows, {df.shape[1]} columns)")
    print(f"Saved protein groups: {groups_path} ({len(grouped)} unique proteins)")


if __name__ == "__main__":
    main()
