# Analysis walkthrough — pMHC-TCR register-crossing project (HLA-A\*02 / DMF5)

**System:** two known crystal registers of the same TCR-MHC pair, differing only in how the peptide
threads the groove — **GIG** (PDB 6AM5) and **DRG** (PDB 6AMU), 2.87 Å apart at the Cα level. The
project's central question: can RFdiffusion + ProteinMPNN recover a peptide's own native register, or
even cross it into the other register, from conditioning alone?

This file is the map. All 17 live notebooks are numbered `00`-`16` and grouped by **methodology**, not
by when they were built. Each group answers a different kind of question; read within a group top to
bottom, groups can be read independently. Every notebook was regenerated from its builder script
(`py/build_*.py`) and re-executed end to end on 2026-07-10 — no notebook here contains stale, unexecuted,
or hand-edited cells. Superseded/redundant notebooks were retired to `archive/notebooks_superseded/`
(see its `README.md` for exactly why each one was cut). Figures live in `figures/`, not scattered at
top level.

---

## Group A — MD (native dynamics; no generative model involved)

Establishes the physical ground truth: how far apart the two registers really are, how wide each
register's own thermal envelope is, and what that implies as a calibration for judging every generated
design later.

| # | notebook | question | headline |
|---|---|---|---|
| 00 | `native_register_components` | Where, structurally, does the 2.87 Å crystal difference actually live? | MHC groove is a rigid, register-agnostic socket (floor 0.20-0.39 Å); peptide carries the register (2.87 Å); CDR3 loops are rigid (0.59/0.47 Å); the "TCR moves" signal is a whole-domain docking-pose shift (3.23 Å), not a CDR conformational change. Confirms no RFdiffusion run was ever seeded from an apo/no-peptide complex. |
| 01 | `md_calibrated_baseline` | What's the right yardstick for "resembles the native peptide" — a point or a distribution? | Replaces the single 2.87 Å crystal distance with the native MD **ensemble**: a calibrated joint test (backbone proximity AND correct F-pocket occupant AND correct burial depth). Registers are cleanly resolved (clouds never overlap). **Updated 2026-07-10** with the extended 370K reruns (Tamarind `lbv69`/`p1yv2`, 2ns equil/50ns production, 5x longer): the DRG band widened from ≤1.72 to **≤2.10 Å**, and under it **4 of the 5 closest de-novo crossing candidates now pass the full joint test** (all targeting DRG from a 6AM5 scaffold); one de-novo design (`promising/6AMU_L5_max`) also now clears calibrated OWN-register recovery (1/1295, was 0/542). Every one of the 5 candidates is also closer to at least one individual 370K thermal MD frame than to the register mean (best: 1.26 Å). Read as "no longer excluded by the current calibration," not "confirmed crossing" — the 50ns run still hasn't converged (see caveat). |
| 02 | `per_residue_register` | Is the register signal localized, or does it affect the whole peptide? | Register is a **C-terminal** event: 8σ at P9, 14.5σ at P10, near-zero at P1-P4 (native RMSF confirms these anchors are genuinely tight, not noisy). **Correction applied this pass:** the old "de-novo fails globally, U-shaped at both termini" claim was an averaging artifact — splitting by threading direction (§3b) shows the dramatic U-shape appears ONLY in reverse-threaded designs; forward-threaded designs alone are much flatter (~3-7 Å), a real near-miss rather than gross mis-placement. |
| 03 | `q30_midpopulation_characterization` | A design's own MD showed 214/1000 frames sitting between both registers — is that a real bridge? | The mid-region population is a genuine, metastable structural state (~2.0/1.9 Å from both registers simultaneously, not transient) — but its own F-pocket register readout is flagged as untrustworthy (non-robust Kabsch fit on a small sample). Open question, not a closed one — good scientific hygiene, not a claim. |

**Caveat carried by 00, 01, 02:** 300K is still the original 10 ns production / 0.2 ns equilibration run
(not yet rerun). 370K was extended 5x (2 ns equil / 50 ns production) specifically to firm up the
calibration — but a convergence check on the LONGER run shows it *still* hasn't converged: both
trajectories keep drifting (net +0.25 Å for 6AM5, +0.82 Å for 6AMU over the 50ns) and the
autocorrelation time came out *longer* than the original 10ns run suggested (5.3 ns / 10.7 ns vs
1.7 ns / 0.8 ns before) — effective independent samples are still only ~9 and ~5 out of 1000 raw
frames. Practical effect: the 370K DRG band widened (≤1.72→≤2.10 Å) while the GIG band tightened
slightly (≤1.86→≤1.49 Å) — this is where the still-drifting trajectory happened to be, not a settled
answer. A true converged estimate likely needs several-hundred-ns runs given the measured ~10 ns
relaxation timescale. This does change some individual verdicts (see notebook 01) — 4/5 close crossing
candidates now pass the joint test at 370K, and one design now clears own-register recovery — so treat
these as the current best reading, not final.

---

## Group B — Non-de-novo (fixed/crystal backbone recovery; no RFdiffusion at all)

Sequence design directly on the *real* crystal backbone — the positive control for "can ProteinMPNN
recover this peptide at all, given the right geometry?"

| # | notebook | question | headline |
|---|---|---|---|
| 04 | `dataset_recovery_22structures` | Across 22 diverse pMHC-TCR crystal structures (not just GIG/DRG), how well does ProteinMPNN recapitulate the native sequence on the true backbone? | Dataset-mean 47.4% identity (non-Pro natives) vs 31.9% (Pro-containing, anchor-constrained); P2 anchor recovers well (68%) vs C-terminal anchor poorly (22%); 0/22 exact full-sequence recovery anywhere. |

(6AM5/6AMU/6AMT crystal-backbone recovery specifically is inside notebook 11, §1 — see Group D; it's kept
there rather than split out because the same notebook's later sections are what make it worth comparing
against the de-novo/T-variable results.)

---

## Group C — De-novo, fixed generation parameters (the core register-crossing question)

RFdiffusion generates the peptide backbone from scratch (contig `10-10`, nothing templated), full
`diffuser.T=30`, only contact hotspots vary. This is the heart of the project.

| # | notebook | question | headline |
|---|---|---|---|
| 05 | `denovo_sampling_ramachandran` | What does RFdiffusion's raw per-residue backbone sampling (φ/ψ) look like, independent of the register question? | 1288+ designs, per-position Ramachandran clouds overlaid on native GIG/DRG dihedrals; noTCR vs withTCR clouds look similar (register is peptide-intrinsic, not TCR-imposed). |
| 06 | `conditioning_correlation` | Does adding more RFdiffusion contact hotspots pull the backbone toward native? | Cognate-RMSD medians sit flat (~10.5-13.7 Å) across 0→8 hotspots — the weak-correlation result anticipated and confirmed (matches project-wide ρ≈0.17 finding). |
| 07 | `contact_ladder_mpnn` | Same hotspot ladder (3→42), but read out via ProteinMPNN sequence identity instead of backbone RMSD — does *identity* respond even if geometry doesn't? | Mean identity roughly doubles L1→L5 (contacts DO shape which sequences MPNN proposes on a given backbone) — but flagged with an explicit caveat that this MPNN-identity signal was never reproduced structurally; the max-identity ceiling saturates because the TCR-facing core stays unrecovered regardless. |
| 08 | `recovery_consistency` | (A) Do independently-launched campaigns under the *same* hotspots land in the same place? (B) Pooling everything, does RFdiffusion fall into a small number of reused "default" folds? | (A) checked via ground-truth `.trb` hotspot sets, Jaccard overlap quantified. (B) Two dominant clusters absorb ~99% of designs — **new this pass:** shown to be near-pure forward vs. reverse threading populations, not a separate "generic attractor" discovery. |
| 09 | `q30_conditioning_arms` | Does interaction-only (fpocket/tcr/apoc) conditioning, without any geometric scaffold, steer output into a register basin? | All non-scaffold arms land 100% "seated" but exactly on their *own seed's* register (near-native echoes) — not crossings. |
| 10 | `q30_intermediate_search` | Across every available backbone (native MD, relaxation snapshots, chimeras, 614 prior designs, design-backbone MD), does anything sit structurally between the two registers? | No static source shows a bridge. **Corrected this pass:** the one real exception (design-backbone MD, 214/1000 mid-region frames) is now included in the "closest any backbone gets to both wells" ranking and explicitly named in the verdict (previously silently omitted, a self-contradiction) — it's the open lead that motivates notebook 03. |

---

## Group D — De-novo, temperature/sampling-parameter variable

The one live notebook where a generation-temperature axis is swept explicitly (ProteinMPNN's sampling
temperature, T = 0.1/0.3/0.5/0.7, layered on top of RFdiffusion contact ladders), alongside its own
fixed-backbone positive control.

| # | notebook | question | headline |
|---|---|---|---|
| 11 | `dmf5_cross_reactivity` | (§1, non-de-novo) Does ProteinMPNN recover 6AMU/6AM5/6AMT on their real crystal backbones? (§2-4) Is recovery structure- and TCR-specific? (§5, T-variable) How does sampling temperature affect de-novo diversity/recovery/NLL? (§6, de-novo) Where does full RFdiffusion generation plateau? | **Corrected this pass:** §1's claim was "recovers both natives exactly" — the data actually shows 0/5000 exact matches; fixed to state the true finding (~48-49% mean identity on the TCR-facing motif, not exact recovery). Cross-specificity gap is real (own ≫ other, +28-36%). Higher MPNN temperature trades confidence (worse NLL) for diversity. De-novo RFdiffusion plateaus at ~30% identity to either native regardless of conditioning or temperature — never reconstructs the specific groove geometry. |

---

## Group E — ProteinMPNN (fixed-backbone sequence design; separate systems)

Two genuinely distinct sub-projects that use ProteinMPNN but aren't about the GIG/DRG register question:
noMHC-context recovery (still on GIG/DRG, but testing whether the MHC is even needed), and a fully
separate system (2P5E / NY-ESO-1) used as an earlier design-methodology pilot.

| # | notebook | question | headline |
|---|---|---|---|
| 12 | `nomhc_campaign_report` | Without the MHC present, can ProteinMPNN still recover the cognate peptide from TCR contacts alone, across every backbone type (native/relaxed/RFD1/RFD3)? | 528,750 designs, 614 backbones. Near-native backbones: 43.2% cognate vs 17.1% other (2.5× steering). Every RFdiffusion-generated backbone (v1 and RFD3): nearchance (~7-8%, vs 10% random). Recovery tracks backbone quality, not conditioning. |
| 13 | `2p5e_mpnn_design_analysis` | Full ProteinMPNN design-space characterization (logos, entropy, PCA, MHCflurry binding) on the 2P5E/NY-ESO-1 peptide — separate system, methodology pilot. | 50,028 designs; 0 exact native recovery; 22.7% predicted <50nM MHC binders; establishes the design-analysis toolkit later reused on GIG/DRG. |
| 14 | `2p5e_campaign_comparison` | How does design behavior change across 4 different MPNN campaign contexts (full-complex, MHC-only, no-Met/no-Pro, relaxed-ensemble)? | Identity to native ranges 10.6%-46.6% depending on context; two placeholder sections (RFdiffusion peptides, 22-structure panel) are honest "not ready yet" stubs, not silent gaps. |
| 15 | `2p5e_relax_ensemble_mpnn` | Does designing across an OpenMM-relaxed backbone *ensemble* (12 conformers) change recovery vs a single static backbone? | 10,671 designs / 3,846 unique across 6 backbones; best single-backbone consensus only 33% identity to native — conformational averaging doesn't rescue recovery either. |

---

## Group F — Synthesis

| # | notebook | question |
|---|---|---|
| 16 | `claims_evaluation_final` | A reviewer proposed six corrected claims about this whole project; this notebook traces every specific number back to its source and either confirms, corrects, or flags it unsupported — live-computed, not asserted. Includes the full occupancy→depth→proximity joint-test waterfall (43→11→**0** genuine crossings) and an explicit note that `occ_crossed` was retired project-wide as a metric after being found to read a spoofable/pinned anchor position, not a real crossing. This is the capstone: read it last. |

---

## The headline result, if you only read one line per group

- **A (MD):** the two registers are real and cleanly separated — but the reference MD is still not
  converged even after a 5x extension (370K, 50ns), so treat exact thresholds as provisional. Under the
  current best (widened) 370K calibration, 4 of the 5 closest de-novo crossing candidates now pass the
  full joint test, and one de-novo design clears own-register recovery for the first time (1/1295) —
  read as "no longer excluded," not "confirmed," pending a genuinely converged run.
- **B/D (non-de-novo):** ProteinMPNN recovers the functionally-decisive motif on the *true* backbone
  (~48-49% identity) but never the exact sequence — geometry, not search depth, is the bottleneck.
- **C (de-novo, fixed-T):** contact conditioning barely moves backbone geometry (ρ≈0.17); ~half of any
  de-novo pool is a reverse-threading artifact that must be filtered before trusting any register
  statistic; recovery/crossing of a de-novo design's own or the other register is exceedingly rare
  (≤0.1% either way) even after removing that artifact and under the widened calibration — not the
  clean "0%, ever" of the original 10ns-MD verdict, but not a capability either.
- **E (ProteinMPNN, separate systems):** the same tools, different peptide/MHC — useful as a methodology
  cross-check, not part of the register-crossing verdict.
- **F (synthesis):** every number in this document has been traced to its source and re-verified; where
  an earlier draft overclaimed, the correction is stated explicitly rather than smoothed over.
