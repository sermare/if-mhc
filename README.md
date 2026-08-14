# Inverse Folding Models Sample Sequence Space Shaped by MHC Anchor Identity and TCR Context

Sergio E. Mares — Adimab · Center for Computational Biology, UC Berkeley

Paper: [`paper/paper_backbone.pdf`](paper/paper_backbone.pdf) · LaTeX source: [`paper/paper_backbone.tex`](paper/paper_backbone.tex) · Overleaf bundle: [`paper/latex.zip`](paper/latex.zip)

---

## Abstract

Inverse-folding models have emerged as potent tools for designing protein sequence from structure, and
the degree to which they capture the biophysical constraints of a specific molecular interface stands as
an open question. TCR cross-reactivity has caused fatal toxicity in the clinic, yet the peptide sequence
space is far too large to screen experimentally. We therefore ask whether inverse-folding models can
recover the native peptide from structure alone, and what their sampling distribution reveals about the
constraints that give rise to cross-reactivity. We applied four inverse-folding models to twenty solved
pMHC–TCR structures under two structural contexts, generating 1.6 million peptide designs, scored where
allele-matched against two independent binding predictors, and characterized the full sampling
distribution each model induces over peptide sequence space. No model recovered the native peptide.
Recovery is instead bimodal across the two anchor positions that dominate MHC binding. The C-terminal
anchor PΩ, though structurally load-bearing, is largely missed, while the position-2 anchor recovers at
or above the interior average. Removing the T-cell receptor increases sampling diversity across all four
models and raises predicted binding affinity in a model-dependent manner, but lowers recovery at
experimentally validated contact positions and makes every model that reports a confidence score less
confident in its own designs, showing that the models read the receptor as part of the interface. This
suggests that inverse-folding models encode peptide compatibility as a set of position-specific
structural constraints rather than a single optimal sequence. Our investigation establishes the sampling
distribution as a probe of the constraints governing peptide recognition, and highlights both the promise
and the limitations of structure-based models for predicting cross-reactivity.

## Design

| | |
|---|---|
| **Structures** | 20 solved pMHC–TCR crystals (19 HLA-A\*02:01, 1 HLA-B\*51:01) |
| **Models** | ProteinMPNN · ProteinMPNN (no MHC) · ESM-IF1 · LigandMPNN |
| **Contexts** | `full` (MHC + β₂m + TCRαβ) and `mhconly` (TCR removed) |
| **Sampling** | 10,000 raw designs per (structure, model, context) at *T* = 0.1 |
| **Total** | **1.6 million peptide designs** |
| **External scoring** | MHCflurry and ESMCBA, on every unique design from the 19 A\*02:01 structures |
| **External validation** | SKEMPI 2.0 and IEDB, matched to each structure's own TCR by CDR3 |

Only the peptide chain is designed; all other chains are held fixed as structural context.

## Key results

**The two MHC anchors behave oppositely, and the difference is chemical rather than positional.**
P2 (pocket B) constrains a specific side chain and recovers at 0.64 against an interior-position average
of 0.49. PΩ (pocket F) imposes a permissive backbone constraint that many side chains satisfy, and
recovers at 0.19 — significantly *below* the interior average (*p* = 0.0054). Both positions are equally
load-bearing structurally; only one gives a sequence-design model a signal it can act on.

**Resolution's effect is real but hidden by pooling.** Across twenty different peptides the correlation
is modest (*r* = −0.49). Holding peptide identity fixed with four independent crystals of SLLMWITQC
(NY-ESO-1) makes it unambiguous: *r* = −0.99 and −0.97 for recovery, *r* = 0.97 and 0.96 for unique
design counts.

**Designs look like real binders, unevenly by model.** Two independent predictors agree closely
(*r* = 0.83–0.91). At the 500 nM cutoff, 61–78% of ProteinMPNN-family and LigandMPNN designs qualify
versus 45% for ESM-IF1 (χ² = 238.46).

**Removing the TCR pulls three levers in different directions.** Unique design counts roughly double or
triple (ProteinMPNN 1,081 → 3,671). Predicted affinity rises for some models and not others (no-MHC
77.5% → 95.2%, LigandMPNN 61.1% → 70.7%, the other two barely move). Recovery at the 21
experimentally validated TCR-contact positions falls from 54–58% to 34–38%, and every model that emits a
confidence score becomes less confident in its own output.

## Repository layout

```
paper/           paper_backbone.tex/.pdf  — the manuscript
                 latex.zip                — self-contained Overleaf bundle
                 proposal.tex/.pdf        — one-page fellowship proposal
notebooks/panel/ 01-07  the canonical analysis; every figure and statistic in the paper
notebooks/       17-26  earlier single-system and ablation studies
py/              build_panel_*.py  emit the panel notebooks; one script per notebook
jobs/            generation drivers (cron-supervised campaign runners)
figures/         final renders, one directory per figure
inputs/          pmhc_tcr_dataset/  the 20-structure panel
docs/            backbone_generation/  docs for the earlier backbone-generation manuscript
```

Every figure is produced by a `py/build_panel_*.py` script that emits a notebook under
`notebooks/panel/`; re-running the script and re-executing the notebook reproduces the numbers exactly.

| notebook | produces |
|---|---|
| `01_dataset_presentation` | panel composition, peptide lengths, resolution, MHC contact density |
| `02_design_presentation` | sampling redundancy, sequence logos, unique-design counts |
| `03_recovery_presentation` | per-position recovery, the anchor result, TCR-context benefit |
| `04_replicate_structures` | the same-peptide replicate set that isolates resolution |
| `05_skempi_validation` | SKEMPI/IEDB cross-reference at validated TCR-contact positions |
| `06_mhcflurry_esmcba_umap` | predicted-affinity scoring, shared embedding, confidence shift |
| `07_chemistry` | anchor hydrophobicity and side-chain class |

## A second manuscript lives here

This repository also contains an earlier, separate project — contact-conditioned Cα backbone generation
in the cross-reactive DMF5 system ([`paper/paper.pdf`](paper/paper.pdf)). Its documentation is under
[`docs/backbone_generation/`](docs/backbone_generation/). The two share inputs and infrastructure but are
independent studies; the abstract above describes only the inverse-folding work.
