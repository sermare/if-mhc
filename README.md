# Inverse Folding Models Track Structural Compatibility in pMHC-TCR Complexes

Sergio E. Mares, Brian Petersen, Kyle Barlow — Adimab · Center for Computational Biology, UC Berkeley

Paper: [`paper/paper_gem.pdf`](paper/paper_gem.pdf) · LaTeX source: [`paper/paper_gem.tex`](paper/paper_gem.tex)

---

## Abstract

Predicting cross-reactivity in T-cell receptor (TCR)-based cancer therapies is a major challenge.
Previous clinical studies failed to identify cross-reactive targets, ultimately resulting in fatal
cardiac toxicity in two patients, and the combinatorial vastness of the promiscuous peptide-MHC (pMHC)
landscape is beyond the reach of experimental sampling. While inverse folding (IF) models can sample
peptide sequences from a distribution conditioned on structure, whether they recover the native peptide
or map the tolerated neighborhood underlying cross-reactivity is unknown. Here we sampled peptides from
four IF models across 43 pMHC-TCR crystal structures and scored them for predicted presentation and
structural compatibility. We show that IF models do not recover the native peptide or rank binding
strength, but instead sample from the sequence space compatible with the structural context they are
conditioned on, offering a handle on the tolerated neighborhood that drives cross-reactivity. Together,
these results establish IF model sampling distributions as an effective probe of the constraints
governing peptide recognition and cross-reactivity, critical for TCR-based therapies.

## Design

| | |
|---|---|
| **Structures** | 43 solved pMHC–TCR crystals, all human MHC class I — 40 HLA-A\*02:01, 1 HLA-B\*51:01 (4MJI), 1 HLA-B\*08:01 (1MI5), 1 HLA-B\*35 (2AK4). Mean resolution 2.48 Å |
| **Models** | ProteinMPNN · ProteinMPNN (no MHC) · ESM-IF1 · LigandMPNN |
| **Contexts** | pMHC+TCR (`full`) and pMHC only (`mhconly`, TCR removed) |
| **Sampling** | up to 10,000 designs per (structure, model, context) at *T* = 0.1 |
| **Total** | **~3.4 million peptide designs** |
| **External scoring** | MHCflurry and ESMCBA on every unique design from the 40 HLA-A\*02:01 structures |
| **External validation** | SKEMPI 2.0 ΔΔG (23 positions, 11 complexes) and IEDB (21 positions), matched to each structure's own TCR by CDR3 |

Only the peptide chain is designed; all other chains are held fixed as structural context.

## Key results

**No model recovers the native peptide, and recovery is sharply anchor-dependent.** Pooled across all
40 HLA-A\*02:01 structures and 4 models, P2 (pocket B) recovers the native residue at 0.649 against an
interior average of 0.496, and matches side-chain chemistry class at 0.861 vs. 0.708 interior. PΩ
(pocket F) recovers at only 0.220 — *below* the interior average — while still matching chemistry class
at 0.672 (n.s. vs. 0.708 interior). On the 30-nonamer subset used for the paper's significance test,
this gap is P2 70% vs. 49% interior (*p* = 5.2×10⁻⁶) and PΩ 28% vs. 49% interior (*p* = 7.5×10⁻⁷).

**The split survives deduplication.** The panel holds 29 distinct peptides and 22 distinct CDR3β
sequences, so a non-redundant set (one structure per TCR family and per peptide, 30 HLA-A\*02:01
structures over 26 peptides) is also reported: interior 0.523, P2 0.644 (*p* = 0.0095), and the PΩ
deficit *deepens* to 0.175 (*p* < 10⁻⁴).

**Designs look like real binders, unevenly by model.** Two independent predictors (MHCflurry, ESMCBA)
correlate at *r* = 0.83–0.93 in full context (0.82–0.93 without the TCR). At the 500 nM cutoff, 47–55%
of ProteinMPNN-family and LigandMPNN designs qualify versus 32% for ESM-IF1 (χ² = 266.64,
*p* = 2×10⁻⁵⁷).

**Removing the TCR moves three things at once.** Unique designs rise ×1.67–2.85 across the four models
(ProteinMPNN 2121→6049; ProteinMPNN (no MHC) 2434→4070; ESM-IF1 1824→3539; LigandMPNN 1641→4343);
predicted affinity rises for every model but unevenly (strong-binder rate +4.0 to +25.7 percentage
points); recovery falls from 44–48% to 29–36%; and every model that reports a confidence score becomes
less confident (Mann-Whitney *p* < 10⁻²⁰⁰ for all three that report one). Pooled over 23 measured-ΔΔG
positions across 11 SKEMPI complexes, the recovery gain from TCR context at critical mutant positions
(+0.143 vs. +0.111) is not significant (*p* = 0.26) — a consistent direction on an underpowered sample.

**The models read conformation, not affinity.** 6AM5 and 6AMU are the same DMF5 receptor and groove
solved with two different peptides; each ranks its own crystallized peptide 2.3/13 on its own backbone
and 11.0/13 on the other. Against published K_D for those same peptides, the score carries no
information (|ρ| ≤ 0.32, all *p* > 0.4).

## Repository layout

```
paper/           paper_gem.tex/.pdf  — the manuscript (GEM workshop submission)
                 references.bib      — bibliography
notebooks/panel/ 01-08  the canonical analysis; every figure and statistic in the paper
py/              build_panel_*.py emit the notebooks; design_corpus.py loads the designs;
                 skempi_*.py / score_*.py generated and scored the designs on the cluster
designs/         the design tables, one gzipped CSV per model and arm
inputs/          pmhc_tcr_dataset/  the panel structures, canonicalised to chains A–E
figures/         paper/     the manuscript's figures, named by the number they carry
                 fig_panel* raw notebook output, one directory per notebook
docs/            backbone_generation/  docs for a separate, earlier project (contact-conditioned
                 Cα backbone generation in the cross-reactive DMF5 system); no compiled
                 manuscript for that project lives in this repo yet
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
