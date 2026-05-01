# Pocket Point Cloud Feature Specification

This document describes every aspect of the pocket point cloud preprocessing
performed by `scripts/compute_pocket_pointclouds.py` (Step 5). It covers the
file format, what was discarded and why, what was remapped and why, and the
meaning of every dimension in the per-atom feature vector.

## Overview

Each pocket PDB file from PDBbind is parsed into a point cloud: a set of atoms
with 3D coordinates and chemical features. One `.npz` file is saved per PDB
entry (10,372 total), each containing:

- `positions`: float32 array of shape (N, 3) — centroid-centered xyz coordinates in Angstroms
- `features`: float32 array of shape (N, 46) — per-atom chemical feature vector

These are the direct inputs for the SE(3)-Transformer model.

## Source Files

Pocket PDB files come from PDBbind v2020R1:
`~/Desktop/datasets/PDBbind/P-L/{data_subdir}/{pdb_code}/{pdb_code}_pocket.pdb`

These files contain protein atoms within ~10 Angstroms of the co-crystallized
ligand. PDBbind has already added hydrogens computationally. The files include
both ATOM records (standard amino acids) and HETATM records (waters, metals,
cofactors, modified amino acids).

## Coordinate Centering

All coordinates are centered by subtracting the mean position (centroid) of all
retained atoms. This removes absolute position information so the model cannot
memorize crystallographic coordinate frames. The SE(3)-Transformer architecture
provides rotation/translation invariance on top of this.

## Alternate Conformations

PDB files can contain multiple conformations for the same atom (indicated by the
altloc character at column 16). Only the first conformation is kept: atoms with
altloc ' ' (blank) or 'A' are retained; all others (B, C, etc.) are skipped.
This prevents duplicate atoms at slightly different positions.

---

## Discarded Atoms

The following atoms are removed entirely during preprocessing. They are
crystallographic artifacts — introduced by the experimental technique, not
present in the biological protein.

### Discarded Elements

| Element | Name | Reason for Discarding |
|---------|------|----------------------|
| HG | Mercury | Heavy atom soaking for X-ray phasing. Mercury binds opportunistically to cysteine sulfurs. Not present in the biological protein. Found in 48 pocket files. |
| YB | Ytterbium | Lanthanide used for anomalous dispersion phasing (MAD/SAD). Not biologically present. Found in 4 pocket files. |
| EU | Europium | Lanthanide used for phasing. Not biologically present. Found in 2 pocket files. |
| IR | Iridium | Heavy atom derivative for phasing. Not biologically present. Found in 1 pocket file. |
| CS | Cesium | Added to crystallization buffer as an additive. Not biologically present in the protein. Found in 2 pocket files. |
| SR | Strontium | Used as a calcium substitute in crystallization experiments. Not the native ion. Found in 1 pocket file. |
| RB | Rubidium | Used as a potassium substitute in crystallization experiments. Not the native ion. Found in 1 pocket file. |
| IN | Indium | Heavy atom derivative for phasing. Not biologically present. Found in 1 pocket file. |
| GA | Gallium | Heavy atom derivative for phasing. Not biologically present. Found in 1 pocket file. |

### Discarded Residues

| Residue | Name | Reason for Discarding |
|---------|------|----------------------|
| MSE | Selenomethionine | Engineered methionine analog where sulfur is replaced by selenium. Used universally in MAD/SAD phasing to solve the crystallographic phase problem. Functionally identical to methionine in terms of shape and non-bonded interactions, but the selenium atom has different electronic properties than sulfur. Not present in the biological protein — every MSE residue in the living cell is actually MET. Found in 120 pocket files. All atoms of MSE residues are discarded (typically 8 heavy + 9 hydrogen = 17 atoms per residue). |

---

## Remapped Atoms

### Cadmium → Zinc

| Original | Remapped To | Reason |
|----------|-------------|--------|
| CD (Cadmium) | ZN (Zinc) | Cadmium is used as a crystallographic substitute for zinc because Cd²⁺ has a stronger anomalous X-ray scattering signal while maintaining similar tetrahedral coordination geometry. All 77 cadmium-containing entries in our filtered dataset were investigated: every one is a standard drug target (kinases, proteases, phosphorylases — not cadmium-binding proteins like metallothioneins). The cadmium occupies a site that contains zinc in the biological protein. Both ions are divalent with similar ionic radii (Zn²⁺: 0.74 Å, Cd²⁺: 0.95 Å) and coordination preferences. |

### Deuterium → Hydrogen

| Original | Remapped To | Reason |
|----------|-------------|--------|
| D (Deuterium) | H (Hydrogen) | Deuterium is a heavier isotope of hydrogen (one extra neutron). Chemically identical to hydrogen for the purposes of molecular interactions — same bonding, same van der Waals radius, same hydrogen bonding behavior. Found as D1, D2 atoms in DOD (deuterated water) residues from neutron diffraction experiments. 152 atoms across 14 pocket files. Remapped to H so they receive element index 4 (Hydrogen) instead of 15 (Other). |

### Deuterated Water → Water

| Original | Remapped To | Reason |
|----------|-------------|--------|
| DOD (Deuterated water, D₂O) | HOH (Water, H₂O) | DOD appears in structures solved by neutron diffraction, which uses heavy water. Structurally and chemically identical to regular water for the purposes of binding site characterization. Found in 14 pocket files. The residue name is changed to HOH before feature assignment. |

---

## Element Categories (Feature Indices 0–16)

17-class one-hot encoding. Non-metals first (C, N, O, S, P, H), then metals,
then Other. The element is read from PDB columns 77-78. If that field is blank
(rare, older files), the element is inferred from the atom name.

| Index | Category | Element | Description |
|-------|----------|---------|-------------|
| 0 | C | Carbon | The most abundant element in proteins. Forms the backbone and side chain skeletons. Hydrophobic when bonded only to C and H. |
| 1 | N | Nitrogen | Found in backbone amides, side chain amines (Lys, Arg), imidazoles (His), indoles (Trp), and amides (Asn, Gln). Key hydrogen bonding participant. |
| 2 | O | Oxygen | Found in backbone carbonyls, side chain hydroxyls (Ser, Thr, Tyr), carboxylates (Asp, Glu), amides (Asn, Gln), and water molecules. The primary hydrogen bond acceptor in proteins. |
| 3 | S | Sulfur | Found only in Cys (thiol) and Met (thioether). Can form disulfide bonds (Cys-Cys) and coordinate metals. Larger and more polarizable than O. |
| 4 | P | Phosphorus | Found in cofactors (PLP/pyridoxal phosphate, ATP, NAD) and phosphorylated residues. Forms phosphate groups (PO₄³⁻) that are strongly negatively charged and make specific hydrogen bonding patterns. 105 atoms across the dataset. |
| 5 | H | Hydrogen | Present on all polar and non-polar groups. PDBbind adds hydrogens computationally. Polar H (on N, O) participates in hydrogen bonds. Non-polar H (on C) defines van der Waals surface shape. Includes remapped deuterium (D → H, 152 atoms). |
| 6 | ZN | Zinc | Catalytic or structural metal ion. Coordinates 4 ligands in tetrahedral geometry (typically His, Cys, Asp/Glu). Essential in carbonic anhydrases, matrix metalloproteinases, zinc finger proteins. Found in 1,955 pocket files. Also includes remapped cadmium (77 entries). |
| 7 | MG | Magnesium | Coordinates 6 ligands in octahedral geometry (typically Asp, water). Essential in kinases (coordinates ATP phosphates), ATPases, and polymerases. Found in 986 pocket files. |
| 8 | CA_ion | Calcium | Coordinates 6-8 ligands with flexible geometry. Structural role in some enzymes, signaling role in others. Distinguished from C-alpha atoms by the element column in the PDB file: calcium has element "CA", while C-alpha has element "C". Found in 553 pocket files. |
| 9 | MN | Manganese | Coordinates 6 ligands octahedrally, similar to Mg but redox-active (Mn²⁺/Mn³⁺). Found in arginases, superoxide dismutases, some phosphatases. Found in 366 pocket files. |
| 10 | FE | Iron | Redox-active metal (Fe²⁺/Fe³⁺). Found in heme groups (cytochromes, globins), iron-sulfur clusters (electron transfer), and non-heme iron enzymes (dioxygenases). Includes both FE and FE2 PDB designations. Found in 106 pocket files. |
| 11 | CO | Cobalt | Found in vitamin B12-dependent enzymes and some metalloenzymes as a zinc substitute. Redox-active (Co²⁺/Co³⁺). Found in 66 pocket files. |
| 12 | CU | Copper | Redox-active metal (Cu⁺/Cu²⁺). Found in electron transfer proteins (plastocyanin, azurin) and oxidases (tyrosinase, laccase). Found in 19 pocket files. |
| 13 | NI | Nickel | Found in ureases, hydrogenases, and some superoxide dismutases. Typically octahedral coordination. Found in 83 pocket files. |
| 14 | K | Potassium | Monovalent cation. Structural cofactor in some kinases (e.g., pyruvate kinase) and ion channel selectivity filters. Found in 153 pocket files. |
| 15 | NA_ion | Sodium | Monovalent cation. Structural role in some enzymes, often found in crystallization buffer but can be biologically relevant. Distinguished from nitrogen atoms by the element column: sodium has element "NA", while nitrogen has element "N". Found in 387 pocket files. |
| 16 | Other | Any other element | Catch-all for rare elements not covered above. Includes: AS (arsenic, 40 atoms in CAS modified cysteine), F (fluorine, 11 atoms), Y (yttrium, 6 atoms). Each individually rare (57 total atoms across the dataset). |

---

## Residue Categories (Feature Indices 17–41)

25-class one-hot encoding. Categorization priority:

1. If residue name is one of the 20 standard amino acids → indices 17-36
2. If residue name is HOH or DOD → Water (index 37)
3. If the atom is a single-atom HETATM with a metallic element → Metal ion (index 38)
4. If residue name is in the modified amino acid list → Modified residue (index 39)
5. If HETATM and multi-atom → Cofactor (index 40)
6. Otherwise → Other (index 41)

### Standard Amino Acids (Indices 16–35)

| Index | 3-Letter Code | 1-Letter | Full Name | Key Properties |
|-------|---------------|----------|-----------|----------------|
| 17 | ALA | A | Alanine | Small, hydrophobic, non-polar |
| 18 | ARG | R | Arginine | Large, positively charged (guanidinium), H-bond donor |
| 19 | ASN | N | Asparagine | Polar, H-bond donor and acceptor (amide) |
| 20 | ASP | D | Aspartate | Negatively charged (carboxylate), H-bond acceptor, metal coordinator |
| 21 | CYS | C | Cysteine | Contains thiol (-SH), can form disulfide bonds, coordinates metals |
| 22 | GLN | Q | Glutamine | Polar, H-bond donor and acceptor (amide), longer than Asn |
| 23 | GLU | E | Glutamate | Negatively charged (carboxylate), H-bond acceptor, metal coordinator |
| 24 | GLY | G | Glycine | Smallest amino acid, no side chain, backbone flexibility |
| 25 | HIS | H | Histidine | Aromatic (imidazole), can be positively charged, H-bond donor/acceptor, metal coordinator |
| 26 | ILE | I | Isoleucine | Large, hydrophobic, branched aliphatic |
| 27 | LEU | L | Leucine | Large, hydrophobic, branched aliphatic |
| 28 | LYS | K | Lysine | Positively charged (amino group), H-bond donor, long flexible side chain |
| 29 | MET | M | Methionine | Contains thioether (-S-CH₃), moderately hydrophobic |
| 30 | PHE | F | Phenylalanine | Aromatic (phenyl ring), hydrophobic, pi-stacking |
| 31 | PRO | P | Proline | Cyclic, rigid, no backbone amide H (not a donor), introduces kinks |
| 32 | SER | S | Serine | Small, polar, hydroxyl (-OH), H-bond donor and acceptor |
| 33 | THR | T | Threonine | Polar, hydroxyl (-OH), H-bond donor and acceptor, branched |
| 34 | TRP | W | Tryptophan | Largest amino acid, aromatic (indole bicyclic), H-bond donor (NH), pi-stacking |
| 35 | TYR | Y | Tyrosine | Aromatic (phenol), H-bond donor (OH), pi-stacking |
| 36 | VAL | V | Valine | Hydrophobic, branched aliphatic |

### Non-Standard Residue Categories (Indices 37–41)

| Index | Category | What Maps Here | Description |
|-------|----------|---------------|-------------|
| 37 | Water | HOH, DOD | Crystallographic water molecules trapped in or near the binding pocket. Water participates in hydrogen bonding networks and can bridge between protein and ligand. Present in ~94% of pocket files. |
| 38 | Metal ion | Single-atom HETATM with metallic element | Catalytic or structural metal ions. The specific metal identity is encoded in the element one-hot (indices 6-15). This residue category flag indicates "this atom is a free ion, not part of an amino acid or cofactor." |
| 39 | Modified residue | See list below | Non-standard amino acids resulting from post-translational modification or chemical modification. These are real biological modifications (unlike MSE which is an artifact). They occupy positions in the protein backbone just like standard amino acids. |
| 40 | Cofactor | Multi-atom HETATM not in other categories | Prosthetic groups, coenzymes, substrates, and other non-protein molecules bound in or near the pocket. Examples: heme (HEM), pyridoxal phosphate (PLP), FAD, NAD, sugars (NAG, MAN). |
| 41 | Other | Anything uncategorized | Fallback for residue types not matching any above category. Should be rare. |

### Modified Amino Acid List

These residue names are mapped to the "Modified residue" category (index 38):

| Residue | Full Name | Parent Amino Acid | Modification Type |
|---------|-----------|-------------------|-------------------|
| KCX | Lysine NZ-carboxylic acid | LYS | Carboxylation (biological, e.g., in beta-lactamases) |
| LLP | Lysine-pyridoxal-5'-phosphate | LYS | PLP covalent adduct (in transaminases) |
| CSO | S-hydroxycysteine | CYS | Oxidation |
| TPO | Phosphothreonine | THR | Phosphorylation (signaling) |
| SEP | Phosphoserine | SER | Phosphorylation (signaling) |
| PTR | O-phosphotyrosine | TYR | Phosphorylation (signaling) |
| CSD | 3-sulfinoalanine | CYS | Oxidation |
| CME | S,S-(2-hydroxyethyl)thiocysteine | CYS | Chemical modification |
| OCS | Cysteinesulfonic acid | CYS | Oxidation |
| ALY | N(6)-acetyllysine | LYS | Acetylation (epigenetic) |
| CSX | S-oxy cysteine | CYS | Oxidation |
| CMH | S-methylcysteine | CYS | Methylation |
| MLY | N-dimethyllysine | LYS | Methylation (epigenetic) |
| CCS | Carbamylated cysteine | CYS | Carbamylation |
| NLE | Norleucine | LEU | Non-standard amino acid |
| DAL | D-alanine | ALA | D-amino acid |
| CAS | S-dimethylarsenic-cysteine | CYS | Arsenic modification |
| PHD | Aspartyl phosphate | ASP | Phosphorylation |
| CSS | 1,3-thiazole-4-carboxylic acid | CYS | Modified cysteine |
| PCA | Pyroglutamic acid | GLU | Cyclization |
| HYP | Hydroxyproline | PRO | Hydroxylation |
| TPQ | Topaquinone | TYR | Oxidation (in copper amine oxidases) |
| M3L | N-trimethyllysine | LYS | Methylation |
| NEP | N1-phosphonohistidine | HIS | Phosphorylation |
| AGM | Agmatine | ARG | Decarboxylation |
| FME | N-formylmethionine | MET | Formylation |
| MHO | Methionine sulfoxide | MET | Oxidation |
| SCH | S-methyl thiocysteine | CYS | Methylation |
| SMC | S-methylcysteine | CYS | Methylation |
| SEC | Selenocysteine | CYS | Biological selenium incorporation |
| DVA | D-valine | VAL | D-amino acid |
| AIB | Alpha-aminoisobutyric acid | ALA | Non-standard |
| ABA | Alpha-aminobutyric acid | ALA | Non-standard |
| ORN | Ornithine | LYS | Non-standard (urea cycle) |
| IAS | Aspartyl isopeptide | ASP | Modified aspartate |
| DLE | D-leucine | LEU | D-amino acid |
| DAR | D-arginine | ARG | D-amino acid |

Any HETATM residue with 2+ atoms not in this list and not water/metal is classified as Cofactor (index 40).

---

## Binary Features (Feature Indices 42–45)

### Is Backbone (Index 42)

Set to 1 if the atom is part of the protein backbone (not a side chain atom)
AND the residue is a standard or modified amino acid. Backbone atoms are the
repeating N-Cα-C(=O) chain plus their hydrogens.

Backbone atom names: `N`, `CA`, `C`, `O`, `H`, `HA`

Note: Proline lacks the backbone `H` (amide hydrogen) due to its cyclic structure.
The `CA` atom name here refers to the alpha carbon, not a calcium ion — these are
distinguished by the element column (C vs CA).

Water, metal, and cofactor atoms are never backbone (always 0).

### Is Hydrogen Bond Donor (Index 43)

Set to 1 if the atom can donate a hydrogen bond (has an N-H or O-H group, or is
a water oxygen). Only heavy atoms are marked as donors — the attached hydrogen
atoms carry this information implicitly through their spatial proximity.

**Backbone donors** (all standard amino acids except PRO):
- `N` (backbone amide nitrogen — carries one H)

**Side chain donors by residue:**

| Residue | Donor Atom Names | Chemical Group |
|---------|-----------------|----------------|
| ARG | NE, NH1, NH2 | Guanidinium (3 NH groups, 5 hydrogens total) |
| ASN | ND2 | Amide NH₂ |
| CYS | SG | Thiol S-H (weak donor) |
| GLN | NE2 | Amide NH₂ |
| HIS | ND1, NE2 | Imidazole N-H (protonation-dependent; both marked since protonation state is ambiguous at crystal pH) |
| LYS | NZ | Amino NH₃⁺ |
| SER | OG | Hydroxyl O-H |
| THR | OG1 | Hydroxyl O-H |
| TRP | NE1 | Indole N-H |
| TYR | OH | Phenol O-H |

**Water:** `O` in HOH/DOD — water is both donor and acceptor.

### Is Hydrogen Bond Acceptor (Index 44)

Set to 1 if the atom can accept a hydrogen bond (has a lone pair available for
interaction with an H atom).

**Simple rule:** Almost all oxygen atoms in proteins are acceptors. Nitrogen
atoms are only acceptors if they have a lone pair (not all do).

**All O atoms** → acceptor = 1. This covers:
- Backbone carbonyl O
- ASP OD1, OD2 (carboxylate)
- GLU OE1, OE2 (carboxylate)
- ASN OD1 (amide carbonyl)
- GLN OE1 (amide carbonyl)
- SER OG, THR OG1, TYR OH (hydroxyl — also donors)
- Water O (also donor)

**N atoms** → acceptor = 1 only for:
- HIS ND1, NE2 (imidazole nitrogens — can act as both donor and acceptor depending on protonation)

N atoms that are NOT acceptors (no available lone pair):
- Backbone N (lone pair is in the peptide bond)
- LYS NZ (NH₃⁺ — all lone pairs occupied by hydrogens)
- ARG NE, NH1, NH2 (guanidinium — delocalized, no free lone pair)
- ASN ND2, GLN NE2 (amide N — lone pair conjugated into C=O)

**S atoms** → acceptor = 1 for:
- MET SD (thioether sulfur — has two lone pairs)
- CYS SG (thiol sulfur — weak acceptor)

### Is Aromatic (Index 45)

Set to 1 if the atom is part of an aromatic ring system. Aromatic rings enable
pi-stacking interactions with ligand aromatic groups and contribute to
hydrophobic character and molecular rigidity.

| Residue | Aromatic Atom Names (Heavy) | Aromatic H Atom Names | Ring System |
|---------|---------------------------|----------------------|-------------|
| PHE | CG, CD1, CD2, CE1, CE2, CZ | HD1, HD2, HE1, HE2, HZ | 6-membered phenyl ring |
| TYR | CG, CD1, CD2, CE1, CE2, CZ | HD1, HD2, HE1, HE2 | 6-membered phenol ring (OH attached to CZ is not in the ring) |
| TRP | CG, CD1, CD2, NE1, CE2, CE3, CZ2, CZ3, CH2 | HD1, HE1, HE3, HZ2, HZ3, HH2 | Bicyclic indole (5-membered pyrrole fused to 6-membered benzene) |
| HIS | CG, ND1, CD2, CE1, NE2 | HD1, HD2, HE1, HE2 | 5-membered imidazole ring |

Note: Tyrosine's OH group and its hydrogen (HH) are NOT marked aromatic — they
are attached to the ring but not part of the aromatic system itself.

---

## Complete Feature Vector Reference

Each atom is represented by a 46-dimensional binary feature vector (all values 0 or 1).

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | elem_C | one-hot | Element is Carbon |
| 1 | elem_N | one-hot | Element is Nitrogen |
| 2 | elem_O | one-hot | Element is Oxygen |
| 3 | elem_S | one-hot | Element is Sulfur |
| 4 | elem_P | one-hot | Element is Phosphorus |
| 5 | elem_H | one-hot | Element is Hydrogen (includes remapped Deuterium) |
| 6 | elem_ZN | one-hot | Element is Zinc (includes remapped Cadmium) |
| 7 | elem_MG | one-hot | Element is Magnesium |
| 8 | elem_CA_ion | one-hot | Element is Calcium |
| 9 | elem_MN | one-hot | Element is Manganese |
| 10 | elem_FE | one-hot | Element is Iron |
| 11 | elem_CO | one-hot | Element is Cobalt |
| 12 | elem_CU | one-hot | Element is Copper |
| 13 | elem_NI | one-hot | Element is Nickel |
| 14 | elem_K | one-hot | Element is Potassium |
| 15 | elem_NA_ion | one-hot | Element is Sodium |
| 16 | elem_Other | one-hot | Element is something else (AS, F, Y, etc.) |
| 17 | res_ALA | one-hot | Residue is Alanine |
| 18 | res_ARG | one-hot | Residue is Arginine |
| 19 | res_ASN | one-hot | Residue is Asparagine |
| 20 | res_ASP | one-hot | Residue is Aspartate |
| 21 | res_CYS | one-hot | Residue is Cysteine |
| 22 | res_GLN | one-hot | Residue is Glutamine |
| 23 | res_GLU | one-hot | Residue is Glutamate |
| 24 | res_GLY | one-hot | Residue is Glycine |
| 25 | res_HIS | one-hot | Residue is Histidine |
| 26 | res_ILE | one-hot | Residue is Isoleucine |
| 27 | res_LEU | one-hot | Residue is Leucine |
| 28 | res_LYS | one-hot | Residue is Lysine |
| 29 | res_MET | one-hot | Residue is Methionine |
| 30 | res_PHE | one-hot | Residue is Phenylalanine |
| 31 | res_PRO | one-hot | Residue is Proline |
| 32 | res_SER | one-hot | Residue is Serine |
| 33 | res_THR | one-hot | Residue is Threonine |
| 34 | res_TRP | one-hot | Residue is Tryptophan |
| 35 | res_TYR | one-hot | Residue is Tyrosine |
| 36 | res_VAL | one-hot | Residue is Valine |
| 37 | res_water | one-hot | Residue is Water (HOH or DOD) |
| 38 | res_metal | one-hot | Atom is a metal ion |
| 39 | res_modified | one-hot | Residue is a modified amino acid |
| 40 | res_cofactor | one-hot | Residue is a cofactor or other non-protein molecule |
| 41 | res_other | one-hot | Residue is uncategorized |
| 42 | is_backbone | binary | Atom is part of the protein backbone |
| 43 | is_hbond_donor | binary | Atom can donate a hydrogen bond |
| 44 | is_hbond_acceptor | binary | Atom can accept a hydrogen bond |
| 45 | is_aromatic | binary | Atom is part of an aromatic ring system |

---

## HETATM Residue Type Catalog

All 119 HETATM residue types found across the 19,037 PDBbind pocket files,
and how each is handled in preprocessing.

### Water (→ residue category: Water, index 37)

| Residue | Files | Handling |
|---------|-------|----------|
| HOH | 17,649 | Kept as water |
| DOD | 14 | Remapped to HOH, kept as water |

### Metal Ions (→ residue category: Metal ion, index 38)

| Residue | Files | Element Index | Handling |
|---------|-------|--------------|----------|
| ZN | 1,955 | 6 (ZN) | Kept |
| MG | 986 | 7 (MG) | Kept |
| CA | 553 | 8 (CA_ion) | Kept |
| NA | 387 | 15 (NA_ion) | Kept |
| MN | 366 | 9 (MN) | Kept |
| K | 153 | 14 (K) | Kept |
| NI | 83 | 13 (NI) | Kept |
| FE | 76 | 10 (FE) | Kept |
| CO | 66 | 11 (CO) | Kept |
| FE2 | 30 | 10 (FE) | Kept (ferrous iron, same element as FE) |
| CD | 28 | 6 (ZN) | Remapped to ZN (crystallographic zinc substitute) |
| CU | 19 | 12 (CU) | Kept |
| CU1 | ~1 | 12 (CU) | Kept (cuprous copper, same element as CU) |
| HG | 48 | — | **DISCARDED** (crystallization artifact) |
| YB | 4 | — | **DISCARDED** (crystallization artifact) |
| EU | 2 | — | **DISCARDED** (crystallization artifact) |
| CS | 2 | — | **DISCARDED** (crystallization artifact) |
| SR | 1 | — | **DISCARDED** (crystallization artifact) |
| RB | 1 | — | **DISCARDED** (crystallization artifact) |
| IR | 1 | — | **DISCARDED** (crystallization artifact) |
| IN | 1 | — | **DISCARDED** (crystallization artifact) |
| GA | 1 | — | **DISCARDED** (crystallization artifact) |

### Modified Amino Acids (→ residue category: Modified residue, index 39)

| Residue | Files | Handling |
|---------|-------|----------|
| MSE | 120 | **DISCARDED** (selenomethionine — crystallographic artifact) |
| KCX | 84 | Kept as modified residue |
| LLP | 70 | Kept as modified residue |
| CAS | 57 | Kept as modified residue |
| CSO | 46 | Kept as modified residue |
| TPO | 42 | Kept as modified residue |
| CME | 39 | Kept as modified residue |
| OCS | 26 | Kept as modified residue |
| SEP | 23 | Kept as modified residue |
| CSD | 17 | Kept as modified residue |
| ALY | 10 | Kept as modified residue |
| PTR | 9 | Kept as modified residue |
| NLE | 9 | Kept as modified residue |
| DAL | 8 | Kept as modified residue |
| CSX | 6 | Kept as modified residue |
| CMH | 6 | Kept as modified residue |
| MLY | 5 | Kept as modified residue |
| CCS | 5 | Kept as modified residue |
| (and others) | 1-4 each | Kept as modified residue |

### Cofactors and Other Multi-Atom HETATM (→ residue category: Cofactor, index 40)

| Residue | Files | Description |
|---------|-------|-------------|
| NAG | 104 | N-acetylglucosamine (glycosylation sugar) |
| PLP | 27 | Pyridoxal-5'-phosphate (vitamin B6 cofactor) |
| HEM | 16 | Heme (iron porphyrin cofactor) |
| MAN | 14 | Mannose (sugar) |
| FUC | 11 | Fucose (sugar) |
| SF4 | 7 | 4Fe-4S iron-sulfur cluster |
| F3S | 4 | 3Fe-4S iron-sulfur cluster |
| BMA | 4 | Beta-D-mannose |
| BCB | 4 | Bacteriochlorophyll B |
| HEC | 3 | Heme C |
| FMN | 2 | Flavin mononucleotide |
| (and others) | 1-4 each | Various cofactors, substrates, buffer molecules |

Any multi-atom HETATM residue not in the modified amino acid list and not
water/metal is categorized as Cofactor.
