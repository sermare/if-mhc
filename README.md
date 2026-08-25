# Inverse Folding Models Sample Sequence Space Shaped by MHC Anchor Identity and TCR Context

Sergio E. Mares — Adimab · Center for Computational Biology, UC Berkeley

Paper: [`paper/paper_backbone.pdf`](paper/paper_backbone.pdf) · LaTeX source: [`paper/paper_backbone.tex`](paper/paper_backbone.tex) · Overleaf bundle: [`paper/latex.zip`](paper/latex.zip)

---

## Abstract

Inverse-folding models have emerged as potent tools for designing protein sequence from structure,
and the degree to which they capture the biophysical constraints of a specific molecular interface
stands as an open question. TCR cross-reactivity has caused fatal toxicity in the clinic, yet the
peptide sequence space is far too large to screen experimentally. We therefore ask whether inverse-
folding models can recover the native peptide from structure alone, and what their sampling
distribution reveals about the constraints that give rise to cross-reactivity. We applied four
inverse-folding models to fifty solved pMHC-TCR structures under two structural contexts, generating
3.9 million peptide designs. Every allele-matched design was scored against two independent binding
predictors, and we characterized the full sampling distribution each model induces over peptide
sequence space. No model recovered the native peptide. Recovery is instead bimodal across the two
anchor positions that dominate MHC binding. The C-terminal anchor PΩ, though structurally load-
bearing, is largely missed, while the position-2 anchor recovers at or above the interior average.
Removing the T-cell receptor increases sampling diversity and raises predicted binding affinity for
all four models, though the size of that affinity gain differs almost sixfold between them. It also
lowers recovery at experimentally validated contact positions, and leaves every model that reports a
confidence score less confident in its own designs. Both effects indicate that the models read the
receptor as part of the interface. This suggests that inverse-folding models encode peptide
compatibility as a set of position-specific structural constraints rather than a single optimal
sequence. Our investigation establishes the sampling distribution as a probe of the constraints
governing peptide recognition, and highlights both the promise and the limitations of structure-
based models for predicting cross-reactivity.

## Design

| | |
|---|---|
| **Structures** | 50 solved pMHC–TCR crystals — 40 HLA-A\*02:01, 3 HLA-B, 2 mouse H-2L^d, 5 MHC class II |
| **Models** | ProteinMPNN · ProteinMPNN (no MHC) · ESM-IF1 · LigandMPNN |
| **Contexts** | `full` (MHC + β₂m + TCRαβ) and `mhconly` (TCR removed) |
| **Sampling** | 9,984 designs per (structure, model, context) at *T* = 0.1, a uniform depth every cell can supply |
| **Total** | **3.9 million peptide designs** |
| **External scoring** | MHCflurry and ESMCBA on every unique design from the 40 A\*02:01 structures |
| **External validation** | SKEMPI 2.0 ΔΔG and IEDB, matched to each structure's own TCR by CDR3 |

Only the peptide chain is designed; all other chains are held fixed as structural context.

## Key results

**The native peptide is never recovered — 0 of 3,923,712 designs** — yet the sampling distribution is
far from random. Recovery is sharply bimodal: at P2, 58% of (crystal, model) cells recover the native
residue in ≥99% of designs while 19% recover it in ≤1%.

**The two MHC anchors behave oppositely, and the difference is chemical rather than positional.**
P2 (pocket B) demands a specific side chain and recovers at 0.65 against an interior average of 0.50;
PΩ (pocket F) imposes a permissive constraint many side chains satisfy and recovers at 0.26,
significantly *below* interior (*p* = 2×10⁻⁴). PΩ matches side-chain *class* at 0.67 while matching
identity at only 0.20 — a pocket that fixes chemistry, not sequence.

**The anchor split is HLA-A\*02:01 biology, and it survives deduplication.** On the 40 A2 structures
it is 0.68 / 0.52 / 0.20. It vanishes on the five class II complexes, which have no anchor pockets
(P2 0.32 vs interior 0.31, n.s.), and inverts on mouse H-2L^d. The panel holds only 33 distinct
peptides and 26 distinct CDR3β, so a non-redundant set (one structure per TCR family and per peptide,
30 structures) is also reported: P2 0.64 vs 0.52, and the PΩ deficit *deepens* to 0.17.

**Designs look like real binders, unevenly by model.** Two independent predictors agree at
*r* = 0.82–0.93. At the 500 nM cutoff 52–64% of ProteinMPNN-family and LigandMPNN designs qualify
versus 41% for ESM-IF1 (χ² = 238.50).

**Removing the TCR moves three things at once.** Unique designs rise ×1.7–2.3 in every model;
predicted affinity rises in every model but by margins spanning almost sixfold (+4.0 to +25.7
percentage points); recovery falls from 0.43–0.47 to 0.29–0.36, and every model that emits a
confidence score becomes less confident. Pooled over 55 measured-ΔΔG positions in 17 SKEMPI
complexes, the recovery gain from TCR context is larger where mutation costs binding energy
(+0.150 vs +0.018, *p* = 0.047) — a consistent direction on an underpowered sample, and the
distinction disappears entirely once the receptor is removed.

**The models read conformation, not affinity.** 6AM5 and 6AMU are the same receptor and groove solved
with different peptides; each ranks its own crystallized peptide 2.3/13 on its own backbone and
11.0/13 on the other. Against published KD for those same peptides, the score carries no information
(|ρ| ≤ 0.32, all *p* > 0.4).

## Repository layout

```
paper/           paper_backbone.tex/.pdf  — the manuscript
                 latex.zip                — self-contained Overleaf bundle
notebooks/panel/ 01-08  the canonical analysis; every figure and statistic in the paper
py/              build_panel_*.py emit the notebooks; design_corpus.py loads all 3.9M designs
designs/         the design tables, one gzipped CSV per model and arm
inputs/          pmhc_tcr_dataset/  the 50 structures, canonicalised to chains A–E
figures/         paper/     the manuscript's figures, named by the number they carry
                 fig_panel* raw notebook output, one directory per notebook
docs/            backbone_generation/  docs for the earlier backbone-generation manuscript
```

Every figure is produced by a `py/build_panel_*.py` script that emits a notebook under
`notebooks/panel/`; re-running the script and re-executing the notebook reproduces the numbers
exactly. `py/organize_paper_figures.py` then copies each figure into `figures/paper/` under the
number it carries in the manuscript.

| notebook | produces |
|---|---|
| `01_dataset_presentation` | panel composition, peptide lengths, resolution, MHC contact density |
| `02_design_presentation` | sampling redundancy, sequence logos, unique-design counts |
| `03_recovery_presentation` | per-position recovery, the anchor result, non-redundant test sets |
| `04_replicate_structures` | same-peptide replicate groups that isolate resolution |
| `05_skempi_validation` | SKEMPI ΔΔG and IEDB at validated TCR-contact positions |
| `06_mhcflurry_esmcba_umap` | predicted-affinity scoring, shared embedding, confidence shift |
| `07_chemistry` | anchor hydrophobicity and side-chain class |
| `08_dmf5_6am_kd_scores` | the 6AM5/6AMU pair: conformation vs. measured affinity |

## A second manuscript lives here

This repository also contains an earlier, separate project — contact-conditioned Cα backbone
generation in the cross-reactive DMF5 system ([`paper/paper.pdf`](paper/paper.pdf)). Its documentation
is under [`docs/backbone_generation/`](docs/backbone_generation/). The two share inputs and
infrastructure but are independent studies; the abstract above describes only the inverse-folding work.
