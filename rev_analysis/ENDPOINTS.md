# Pre-registered endpoints for the reviewer-driven GPU experiments

Committed **before** any experiment cell is scored (garden-of-forking-paths guard).
Frozen pool + inclusion rules: `rev_analysis/pool.py`.
Rule: **do not read a cell until it reaches its minimum N.** Min N per cell = **target in the spec**
(120–150 de-novo; 100 partial). Score only at completion, never at partial N.

All register analyses run on the **REGISTER_DEFINED** population (extended ∧ groove-placed ∧
forward-threaded). Primary DRG hit = fpocket_pos==9 ∧ toDRG ≤ 1.58 (95%-CI band).

---

## max-replication (highest priority)
**Why:** one cell (`max`, 42 hotspots) carries 7 of the 11 DRG hits in the whole corpus; hits are
otherwise scattered (tcr1@1, L3@5, mhc_tcr2@9, k18@18 = 1 each), so there is **no monotone breadth
gate**. `max` was the post-hoc extreme of ~14 cells and must be replicated on fresh seeds.
**Primary endpoint:** DRG hit rate (recovery+crossing) with Clopper–Pearson 95% CI, both crystals.
**Prediction:** if real, replicates at ~0.58% (CI overlapping 7/1201); if a seed/batch artifact,
regresses toward the ~0.07% corpus background. Report per crystal separately.

## hotspot-count sweep (k = 1,3,5,7,9,11,13,15)
**Primary:** (a) groove-placement rate vs k; (b) register-hit tail-mass (fraction of REGISTER_DEFINED
with to-own < 2.0 Å) vs k.
**Prediction:** *non-monotonic with a peak at k < 10 then degradation* ⇒ out-of-distribution hotspot
conditioning (the OOD hypothesis). *Monotone rise then saturation* ⇒ mechanistic. Flat ⇒ conditioning
count carries no register info at all.

## template-identity ladder (which residue is templated)
**Primary:** median to-own Cα-RMSD (REGISTER_DEFINED) for ti_cterm2 (P9-10) vs ti_nterm2 (P1-2) vs
ti_mid2 (P5-6) vs ti_cterm1 (P10 only) vs ti_nterm1 (P1 only); hotspots held constant.
**Prediction:** C-terminal templating ≫ N-terminal (ti_cterm2 median to-own lower than ti_nterm2 by
> 1 Å); anchor-only ti_cterm1 (P10) already recovers most of the register (median to-own < 2.0 Å).
This is the sharp localization test that the contact-count ladder cannot make.

## scrambled-hotspot control (same count, receptor residues > 15 Å from peptide)
**Primary:** groove-placement rate, scrambled vs count-matched real (scr15 vs k15; scr08 vs k07/k09).
**Prediction:** if scrambled places in the groove as well as real → contact *identity* is irrelevant for
placement; presence of any conditioning vector is what matters. If scrambled collapses toward null
(~14% groove) → real contacts carry genuine placement information.

## region ablation (MHC-only vs TCR-only vs mixed, fixed count 12)
**Primary:** groove-placement + DRG hit rate, ra_mhc12 vs ra_tcronly vs ra_full12.
**Prediction:** F-pocket-adjacent MHC hotspots dominate; ra_tcronly ≈ null-level; ra_mhc12 ≈ ra_full12.
If TCR-only performs as well as MHC-only, the register signal is not where the C-terminal-anchor
mechanism predicts.

## partial diffusion (seed native, partial_T ∈ {5,10,15,20}, own vs cross hotspots)
**Primary:** fraction landing in the OTHER register (crossing) as a function of partial_T, cross-hotspot
arm vs own-hotspot control.
**Prediction:** barrier-crossing curve — low partial_T stays in the seed register (both arms); as
partial_T rises the cross-hotspot arm begins crossing into the target register while the own-hotspot
control stays put. The partial_T at which crossing appears estimates each basin's radius of attraction.
Null result = cross arm never diverges from own arm ⇒ conditioning cannot redirect even a seeded backbone.

## noise-off / T=50 (lower priority)
**Primary:** groove-placement + DRG hit rate vs the T=30 baseline (`max` condition).
**Prediction:** noise_scale=0 raises in-silico success rate at the cost of diversity; T=50 vs T=30 modest.
Ablative, not mechanistic.
