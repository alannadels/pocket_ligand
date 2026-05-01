# Column Descriptions

This document describes every column in the project's CSV datasets.

## pdbbind_all.csv and pdbbind_filtered.csv

These files are produced by `scripts/parse_pdbbind_index.py`. Each row represents one protein-ligand complex from PDBbind.

| Column | Type | Example | Description |
|--------|------|---------|-------------|
| pdb_code | string | 1hsg | 4-character Protein Data Bank identifier. Uniquely identifies a crystal structure. Used to locate the structure files in the PDBbind dataset (protein PDB, pocket PDB, ligand SDF/MOL2). |
| resolution | float | 2.00 | X-ray crystallography resolution in Angstroms. Measures how precisely atom positions are determined. Lower values = sharper structures. Typical range: 0.75 - 4.60. Null for NMR structures. |
| is_nmr | bool | False | Whether the structure was solved by NMR spectroscopy rather than X-ray crystallography. NMR structures lack a numeric resolution and generally have less precise coordinates. All NMR entries are removed in the filtered dataset. |
| year | int | 1998 | Year the crystal structure was deposited in the Protein Data Bank. Ranges from 1982 to 2019. Used to determine which PDBbind subdirectory contains the structure files. |
| affinity_type | string | Ki | Type of binding affinity measurement. One of: Kd (dissociation constant — direct binding equilibrium), Ki (inhibition constant — inhibitor potency), IC50 (half-maximal inhibitory concentration — 50% activity reduction), or EC50 (half-maximal effective concentration). All measure binding strength but through different experimental assays. |
| affinity_operator | string | = | Comparison operator for the affinity value. '=' means exact measurement. '<', '>', '<=', '>=' mean only a bound is known. '~' means approximate. Only '=' entries are kept in the filtered dataset. |
| affinity_nM | float | 0.4 | Binding affinity converted to nanomolar (nM). All original units (fM, pM, nM, uM, mM) are converted to nM for consistent comparison. Lower values = stronger binding. In the filtered dataset, range is 0 to 10,000 nM. |
| affinity_raw | string | Ki=0.4nM | Original affinity string exactly as it appears in the PDBbind index file. Preserved for reference and debugging. |
| ligand_name | string | MK1 | 3-letter ligand identifier code from the PDB structure. Some entries have longer names (e.g., "6-mer" for peptide ligands) or slash-separated isomers (e.g., "GLA/GLB"). |
| is_covalent | bool | False | Whether PDBbind flagged this entry as a covalent complex (ligand permanently bonded to protein). All covalent entries are removed in the filtered dataset. |
| is_incomplete | bool | False | Whether PDBbind flagged this ligand structure as incomplete (missing atoms). All incomplete entries are removed in the filtered dataset. |
| is_isomer | bool | False | Whether PDBbind noted isomeric forms for this ligand. Kept for reference but not filtered on. |
| reference | string | 1hsg.pdf (MK1) | Full reference/notes section from the PDBbind index line. Contains the PDF citation filename, ligand name, and any additional annotations (covalent, incomplete, compound identifiers, etc.). |
| data_subdir | string | 1981-2000 | PDBbind subdirectory containing this entry's structure files. One of: "1981-2000", "2001-2010", "2011-2019". The full path to structure files is: `PDBbind/P-L/{data_subdir}/{pdb_code}/`. |

## pdbbind_with_properties.csv

This file is produced by `scripts/compute_ligand_properties.py`. It contains all columns from pdbbind_filtered.csv plus 9 computed physicochemical property columns. Each property is computed from the ligand's 3D structure file using RDKit.

### Inherited columns

All 14 columns from pdbbind_filtered.csv (described above) are included unchanged.

### Computed property columns

These are the physicochemical descriptors that our model will learn to predict from pocket geometry. They describe what a binding molecule "looks like" in terms of size, shape, polarity, charge, and flexibility.

| Column | Type | Unit | Example | Description |
|--------|------|------|---------|-------------|
| MW | float | Daltons | 615.73 | Molecular weight. The total mass of the molecule. Larger molecules tend to fill larger binding pockets. Lipinski's Rule of Five suggests oral drugs typically have MW <= 500. |
| logP | float | dimensionless | 2.45 | Partition coefficient (Crippen method). Measures the balance between hydrophobicity and hydrophilicity. Positive values = hydrophobic (prefers oil/nonpolar environments). Negative values = hydrophilic (prefers water). A pocket lined with hydrophobic residues (Leu, Val, Phe) will favor high-logP ligands; a polar pocket (Ser, Thr, Asp) will favor low-logP ligands. |
| HBD | int | count | 3 | Hydrogen bond donor count. Number of NH and OH groups on the molecule that can donate a hydrogen to the protein. These must find complementary hydrogen bond acceptors (C=O, N:) in the pocket. Lipinski threshold: <= 5. |
| HBA | int | count | 7 | Hydrogen bond acceptor count. Number of nitrogen and oxygen atoms on the molecule that can accept a hydrogen from the protein. These must find complementary donors (NH, OH) in the pocket. Lipinski threshold: <= 10. |
| TPSA | float | Angstroms^2 | 92.35 | Topological polar surface area. The total surface area occupied by polar atoms (nitrogen, oxygen, and their attached hydrogens). High TPSA = more polar molecule. Affects solubility, membrane permeability, and must match the pocket's polar surface distribution. |
| rotatable_bonds | int | count | 8 | Number of freely rotating single bonds (excluding terminal bonds and ring bonds). Measures molecular flexibility. Tight/enclosed pockets tend to bind rigid molecules (few rotatable bonds), while open/shallow pockets can accommodate flexible ones. |
| formal_charge | int | elementary charges | -1 | Net formal charge on the molecule. Determined by protonation state. Must complement the pocket's electrostatic character — a negatively charged pocket (Asp, Glu residues) will favor positively charged ligands, and vice versa. |
| aromatic_rings | int | count | 3 | Number of aromatic ring systems in the molecule. Aromatic rings can form pi-stacking interactions with aromatic amino acids in the pocket (Phe, Tyr, Trp, His). They also contribute to molecular rigidity and planarity. |
| fsp3 | float | fraction (0-1) | 0.35 | Fraction of sp3-hybridized carbon atoms. Measures the 3D complexity of the molecule's carbon skeleton. fsp3 = 0 means all carbons are flat (aromatic/sp2). fsp3 = 1 means all carbons are tetrahedral (sp3). Higher fsp3 correlates with better clinical success rates and more 3D shape complementarity to pockets. |

## pdbbind_enriched.csv

This file is produced by `scripts/enrich_protein_metadata.py`. It contains all columns from pdbbind_with_properties.csv plus 6 protein metadata columns retrieved from the RCSB PDB GraphQL API. This is the most complete per-complex dataset, with 29 columns total.

Entries without a UniProt ID (150 entries, mostly engineered antibodies and synthetic constructs) are dropped during this step, reducing the dataset from 10,692 to 10,542 entries. See `documentation/quality_filters.md` for details.

### Inherited columns

All 23 columns from pdbbind_with_properties.csv (14 from index parsing + 9 computed ligand properties) are included unchanged.

### Protein metadata columns

These columns identify which protein each complex belongs to, enabling us to group entries by unique protein and pool their ligands together.

The correct protein entity is identified using a **spatial matching** strategy: the pocket PDB file (which contains protein atoms near the ligand) is parsed to find the dominant chain ID, and that chain is matched to the corresponding polymer entity in the RCSB API response. This is more reliable than taking the first or longest entity, which can fail for multi-chain complexes (e.g., protease + peptide inhibitor). For the ~1% of entries where spatial matching fails (chain ID mismatches between PDBbind and RCSB), the longest entity by sequence length is used as a fallback.

| Column | Type | Example | Description |
|--------|------|---------|-------------|
| uniprot_id | string | P03367 | UniProt accession identifier for the protein. This is the primary key for grouping — all PDBbind entries with the same UniProt ID are the same protein co-crystallized with different ligands. All entries in this file have a valid UniProt ID (entries without one have been dropped). |
| protein_name | string | HIV-1 PROTEASE | Human-readable protein name from the PDB deposition. Taken from the polymer entity whose chain matches the binding pocket (via spatial matching). |
| organism | string | Homo sapiens | Source organism of the protein (scientific name). Most entries are human (Homo sapiens). Other common organisms include HIV-1, mouse (Mus musculus), rat (Rattus norvegicus), and E. coli. |
| classification | string | HYDROLASE | Protein family/function keyword from the PDB. Broad categories like HYDROLASE, TRANSFERASE, LYASE, OXIDOREDUCTASE, etc. Useful for ensuring dataset diversity across protein families. |
| struct_title | string | CRYSTAL STRUCTURE OF HIV-1 PROTEASE... | Full title of the PDB deposition. Contains detailed information about the experiment but is primarily kept for reference. |
| pocket_chain | string | A | The protein chain that forms the ligand-binding pocket, determined by parsing the pocket PDB file and finding the most frequent chain ID among pocket atoms. Used for spatial matching to the correct RCSB polymer entity. |

## protein_groups.csv

This file is produced by `scripts/enrich_protein_metadata.py`. It provides a summary view with one row per unique protein (UniProt ID), showing how many PDBbind entries exist for that protein. Proteins with more entries have more known ligands, making them richer training examples for learning property distributions.

| Column | Type | Example | Description |
|--------|------|---------|-------------|
| uniprot_id | string | P00918 | UniProt accession identifier. Primary key. |
| protein_name | string | CARBONIC ANHYDRASE II | Protein name (taken from the first entry for this protein). |
| organism | string | Homo sapiens | Source organism. |
| classification | string | LYASE | Protein family keyword. |
| n_entries | int | 366 | Number of PDBbind entries (i.e., distinct ligands/structures) for this protein. Higher = more data for property distribution estimation. |
| pdb_codes | string | 1a42,1bcd,2xyz,... | Comma-separated list of all PDB codes belonging to this protein. |

## pdbbind_with_pockets.csv

This file is produced by `scripts/compute_pocket_distributions.py`. It contains all columns from pdbbind_enriched.csv plus a pocket_id column that identifies which specific binding pocket each ligand binds to. 10,372 rows, 30 columns. 170 entries from pdbbind_enriched.csv are dropped here because their structures couldn't be reliably aligned (see `documentation/quality_filters.md`, Filter 7).

A single protein can have multiple binding pockets (orthosteric site, allosteric sites, cofactor sites, etc.). Different ligands for the same protein may bind different pockets. This file assigns each ligand to a specific pocket based on 3D structural alignment and ligand centroid clustering.

### Inherited columns

All 29 columns from pdbbind_enriched.csv are included unchanged.

### New column

| Column | Type | Example | Description |
|--------|------|---------|-------------|
| pocket_id | string | P00918_1 | Unique identifier for the specific binding pocket this ligand occupies. Format is `{uniprot_id}_{pocket_number}`. Pocket numbers are assigned by coordinate-based clustering: protein structures are structurally aligned using the Kabsch algorithm on backbone CA atoms, then ligand centroids are clustered by spatial proximity (within 10 Angstroms = same pocket). CA atoms are matched by residue ID when numbering is consistent (~90% of alignments), or by Smith-Waterman local sequence alignment when numbering differs across PDB depositions (~8%). Entries where alignment fails or produces RMSD > 5 Angstroms (~2%) are dropped. For proteins with only one binding site (the majority), all ligands share the same pocket_id. For proteins with multiple sites (133 out of 2,137), ligands are split across 2+ pocket_ids. |

## pocket_distributions.csv

This file is produced by `scripts/compute_pocket_distributions.py`. It contains one row per unique binding pocket, with aggregate statistics across all ligands assigned to that pocket. This is the **training target** for the model — for each pocket, what physicochemical property distributions do its known binders have? 2,295 rows, 61 columns.

### Pocket metadata columns

| Column | Type | Example | Description |
|--------|------|---------|-------------|
| pocket_id | string | P00918_1 | Unique pocket identifier (same format as in pdbbind_with_pockets.csv). |
| uniprot_id | string | P00918 | UniProt accession of the parent protein. |
| protein_name | string | CARBONIC ANHYDRASE II | Protein name. |
| organism | string | Homo sapiens | Source organism. |
| classification | string | LYASE | Protein family keyword. |
| n_ligands | int | 362 | Number of known binders assigned to this pocket. Higher = more reliable distribution estimates. |
| pdb_codes | string | 1a42,1bcd,... | Comma-separated list of all PDB codes for ligands in this pocket. |

### Property distribution columns

For each of the 9 physicochemical properties, 6 statistics are computed across all ligands in the pocket. Column naming follows the pattern `{property}_{statistic}`.

| Statistic | Description |
|-----------|-------------|
| _mean | Mean value across all ligands in this pocket. The central prediction target. |
| _std | Standard deviation. Measures how much variation exists among binders. A pocket with low std is highly selective for a narrow property range; high std means the pocket accommodates diverse molecules. Zero if only one ligand. |
| _median | Median value. More robust to outliers than the mean. |
| _min | Minimum value observed among binders. |
| _max | Maximum value observed among binders. |
| _count | Number of ligands with a valid (non-null) value for this property. Usually equals n_ligands unless some ligands had parsing issues. |

Properties: MW, logP, HBD, HBA, TPSA, rotatable_bonds, formal_charge, aromatic_rings, fsp3.

Example: For pocket P03367_1 (HIV-1 protease main site, 79 ligands), `MW_mean=621`, `MW_std=96` means binders typically weigh 621 ± 96 Daltons — reflecting the large, peptide-like inhibitors that fit HIV protease's deep symmetric pocket.

## pocket_summary.csv

This file is produced by `scripts/compute_pocket_distributions.py`. It provides a per-protein view of the pocket clustering results, showing how many distinct pockets were identified for each protein. 2,137 rows (one per protein).

| Column | Type | Example | Description |
|--------|------|---------|-------------|
| uniprot_id | string | P00918 | UniProt accession identifier. |
| protein_name | string | CARBONIC ANHYDRASE II | Protein name. |
| n_entries | int | 366 | Total number of PDBbind entries for this protein (before dropping unalignable entries). |
| n_assigned | int | 362 | Number of entries successfully assigned to a pocket. Equals n_entries minus n_dropped. |
| n_dropped | int | 4 | Number of entries dropped due to unreliable structural alignment (fusion proteins, chimeras, etc.). |
| n_pockets | int | 2 | Number of distinct binding pockets identified by clustering. 1 means all ligands bind the same site. >1 means the protein has multiple binding sites with different ligands. |
| pocket_sizes | string | [360, 2] | List of ligand counts per pocket (in order of pocket number), counting only assigned entries. |

## ligand_parse_failures.txt

A plain text file listing PDB codes (one per line) for entries whose ligand structure files could not be parsed by RDKit. These entries have null values for all 9 property columns in pdbbind_with_properties.csv. Typical failure causes include unusual bond types, invalid valences, or corrupted structure files.
