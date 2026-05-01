# Quality Filters Applied to PDBbind Dataset

The raw PDBbind v.2020R1 dataset contains 19,037 protein-ligand complexes. We apply five quality filters to produce a clean subset of ~10,692 entries suitable for training and analysis. All filters are combined with AND logic — an entry must pass all five to be retained.

## Filter 1: Resolution <= 2.5 Angstroms

**What it does:** Removes entries with X-ray crystallography resolution worse than 2.5 Angstroms, and all NMR-solved structures.

**Why:** Resolution measures how precisely atom positions are determined in the crystal structure. At resolutions worse than 2.5 Angstroms, individual atom positions become ambiguous — you can see the overall protein fold but not the exact geometry of the binding pocket. Since our model learns from pocket shape, inaccurate atom positions would introduce noise into both the input (pocket point cloud) and the ground truth (ligand positioning determines which pocket it maps to).

NMR structures are excluded because they don't have a single numeric resolution value and generally have less precise atomic coordinates than good X-ray structures.

**Entries removed:** ~3,750 (304 NMR + ~3,446 low-resolution X-ray)

## Filter 2: No Covalent Complexes

**What it does:** Removes entries flagged as "covalent complex" in the PDBbind index.

**Why:** Covalent inhibitors form a permanent chemical bond with an amino acid in the binding pocket (typically a cysteine residue). This is a fundamentally different binding mechanism from the non-covalent interactions (hydrogen bonds, hydrophobic contacts, van der Waals forces, electrostatics) that our model aims to learn. The physicochemical properties of covalent binders are driven by their reactive warhead chemistry, not by complementarity to pocket shape. Including them would pollute the training signal.

**Entries removed:** 644

## Filter 3: No Incomplete Ligand Structures

**What it does:** Removes entries flagged as "incomplete ligand" or "incomplete ligand structure" in the PDBbind index.

**Why:** If atoms are missing from the ligand's crystal structure (often because part of the molecule is disordered and not visible in the electron density), we cannot accurately compute its physicochemical properties. Molecular weight would be underestimated, hydrogen bond counts could be wrong, and logP would be unreliable. Since these computed properties serve as our ground truth labels for training, inaccurate values would directly degrade model performance.

**Entries removed:** 541

## Filter 4: Exact Affinity Measurements Only

**What it does:** Keeps only entries where the affinity operator is '=' (exact measurement). Removes entries with '<', '>', '<=', '>=', or '~' operators.

**Why:** Entries with inequality operators (e.g., "Ki<10nM" or "IC50>1uM") indicate that the true binding affinity was not precisely measured — we only know a bound. Entries with '~' are approximate. While we could include these as rough estimates, exact measurements give us cleaner, more reliable data. This is especially important when we later group ligands by protein and compute property distribution statistics — imprecise affinity values would affect which ligands we consider "true binders" for a given target.

**Entries removed (combined with Filter 5):** ~4,637

## Filter 5: Binding Affinity <= 10,000 nM (10 micromolar)

**What it does:** Removes entries where the binding affinity is weaker than 10,000 nM (i.e., the numeric value is greater than 10,000).

**Why:** Binding affinity is measured as a concentration — lower values mean tighter binding. Molecules with affinities weaker than 10 micromolar are barely binding to the protein. At these weak affinities, the interaction is often non-specific: the molecule may sit loosely in the pocket without forming meaningful complementary contacts. Including these weak binders would dilute the signal of what physicochemical properties a pocket actually selects for, since weakly-binding molecules don't necessarily reflect the pocket's geometric and chemical preferences.

A 10 uM cutoff is a standard threshold in the field for distinguishing active compounds from inactive ones.

**Note:** This filter is applied together with Filter 4 (exact measurements only), so the combined removal count is ~4,637 entries.

## Filter 6: Must Have a UniProt ID (applied in Step 3)

**What it does:** Removes entries where the protein could not be mapped to a UniProt accession identifier via the RCSB PDB API.

**Why:** UniProt IDs are essential for grouping entries by unique protein — without one, we can't determine which other entries share the same protein target, and thus can't pool ligands for property distribution estimation. The unmappable entries are overwhelmingly engineered antibodies/immunoglobulins (~116), de novo designed proteins (~9), and other synthetic constructs (~25). These are not representative drug targets: antibody binding sites are shaped by in-vitro engineering rather than natural evolution, and de novo proteins don't correspond to real biological targets.

**Entries removed:** 150

**Applied in:** `scripts/enrich_protein_metadata.py` (Step 3), after spatial matching and API enrichment.

## Filter 7: Reliable Structural Alignment (applied in Step 4)

**What it does:** Removes entries whose protein structure cannot be reliably aligned to other structures of the same protein. Specifically, entries are dropped if: (a) structural alignment produces RMSD > 5 Angstroms (indicating a bad superposition), or (b) sequence alignment finds fewer than 10 matching CA atoms (too few for alignment).

**Why:** In Step 4, we align all crystal structures of the same protein into a shared coordinate frame to determine which ligands bind the same pocket (by comparing ligand centroid positions after alignment). If an entry's structure can't be aligned, its ligand centroid position is unreliable, and assigning it to a pocket would be guesswork. The dropped entries are almost exclusively fusion proteins, chimeras, or multi-domain constructs where the pocket chain contains significantly more (or less) protein than the target domain — these are unusual PDB depositions, not standard drug-target structures.

**Entries removed:** 170

**Applied in:** `scripts/compute_pocket_distributions.py` (Step 4), after structural alignment.

## Summary

| Filter | Criterion | Reason | Approx. Removed | Applied In |
|--------|-----------|--------|-----------------|------------|
| 1 | Resolution <= 2.5 Angstroms | Unreliable atom positions at low resolution | ~3,750 | Step 1 |
| 2 | No covalent complexes | Different binding mechanism | 644 | Step 1 |
| 3 | No incomplete ligands | Can't compute accurate properties | 541 | Step 1 |
| 4 | Exact affinity only (=) | Imprecise measurements add noise | (combined below) | Step 1 |
| 5 | Affinity <= 10,000 nM | Weak binders are non-specific | ~4,637 (with #4) | Step 1 |
| 6 | Must have UniProt ID | Can't group by protein without it | 150 | Step 3 |
| 7 | Reliable structural alignment | Can't determine pocket without alignment | 170 | Step 4 |

**Result: 19,037 -> 10,692 (Step 1) -> 10,542 (Step 3) -> 10,372 (Step 4)**

Note: Filters 1-5 overlap (one entry can fail multiple filters), so individual removal counts don't sum to the total removed. Filters 6 and 7 are applied separately in later steps.
