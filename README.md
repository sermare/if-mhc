# noMHC ProteinMPNN campaign

Can an **MHC-blind** ProteinMPNN variant (`proteinMPNN_noMHC`, trained with MHC chains *excluded*)
design peptides that recover the two experimentally-characterised DMF5 epitopes when placed on
various backbones — and can any backbone/conditioning present **both** peptide conformations at once
(the dual-interface objective)?

Reference epitopes (both 10-mers, HLA-A\*02; anchors **P2** and **PΩ = P10**):

| epitope | sequence | crystal |
|---|---|---|
| **GIG** | `SMLGIGIVPV` | 6AM5 |
| **DRG** | `MMWDRGLGMM` | 6AMU |

Scoring is **pid-aware**: 6AM5 backbones are scored vs GIG (its cognate), 6AMU vs DRG. Recovery = %
positions identical to native; chance ≈ 10%.

## Bottom line

- **noMHC does not bridge the OOD gap.** RFdiffusion backbones sit 8–22 Å from either native basin and
  recover neither peptide (≈ chance). Recovery is real only on near-native backbones.
- Even on the exact crystal backbone, the **MHC anchors P2 and PΩ collapse to ~0%** while the buried /
  TCR-facing core (P4–P7, P9) recovers 75–100% — the model is fundamentally MHC-blind at the anchor pockets.
- **No dual capture** by any designed conditioning; only 5/46 campaigns clear chance on *both* peptides,
  all of them native/relaxed backbones.
- Highest-value next experiment: run the stock **MHC-aware** ProteinMPNN on the same backbones and diff
  per-residue to quantify the anchor cost of MHC-blind training.

Full analysis — inventory, recovery, cross-recovery, per-residue heatmaps, per-category logos, distributions,
structural views, and Hamming-sorted closest designs — lives in the notebook.

## Contents

| path | what |
|---|---|
| `nomhc_campaign_report.ipynb` | Full campaign report (canonical dataset: 528,750 designs across 614 backbones). |
| `inputs/` | Structural inputs: `2P5E` pMHC-TCR complex, the pMHC-TCR reference dataset (`pmhc_tcr_dataset/`), the DMF5 focus set (`focus_6am/`: 6AM5/6AMU/6AMT crystals, contigs, hotspot/ladder specs), and MHC-only / trimmed PDBs. |
| `README.md` | This file. |

Peptides are 10-mers, ProteinMPNN sampling temperature 0.3. See the notebook for the full run setup,
backbone inventory, and per-campaign outcomes.
