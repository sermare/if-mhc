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

---
## max-replication DECISION RULE (committed before cells land)
Pooled other-cell rate = (11-7)/(16919-1201) = 4/15718 = **0.025%**. Original `max` = 7/1201 = 0.58%.
- **HOLDS**  if fresh-seed maxrep rate's 95% CI **lower** bound > 0.025% (exceeds background).
- **FAILS**  if its 95% CI **upper** bound < 0.58% (regresses below the original point estimate).
- **INDETERMINATE** otherwise. Report per crystal (max hits split 6AMU 5 / 6AM5 2, so both must be read).
Prior: the 18-contact cell was special in the old corpus (C18 1.61%) but ordinary in the new (k18 0.083%)
-> cross-campaign top-cell effects have not replicated here; a null replication is the expected outcome.

## max-ABLATION (direct test of 3: composition vs coverage) -- runs ALONGSIDE maxrep
`max` is a strict superset of L5_max/k24/k18/k14; k24(24)=0/1203 while max(42)=7/1201, so it is NOT a
monotone coverage threshold. `max` has 12-16 residues (per crystal) in no other scheme.
- **max_drop12** (max minus its unique residues, ~25-30 left): primary = DRG hit rate.
  Prediction: if COMPOSITION, regresses toward k24 (~0%); if COVERAGE, stays ~0.58%.
- **max_uniq12** (only the unique residues as hotspots): primary = DRG hit rate + groove placement.
  Prediction: if those residues carry the effect, elevated over background; else ~0.

---
## ROUND-4 CORRECTIONS (committed before maxrep/ablation cells land)

**Hit definition is now STRICT (<=1.48 A) primary.** Under strict, corpus = 5 hits (4 recovery + 1
crossing), `max` = **4/1201 = 0.333%** [CP 0.091-0.851]; multinomial homogeneity p=1e-4, post-hoc
max-cell null P(any>=4)=0.0018 (survives). Pooled other-cell STRICT rate = 1/15718 = **0.0064%**.

**maxrep DECISION RULE (restated, STRICT):**
- HOLDS if fresh-seed STRICT rate 95% CI lower bound > 0.0064%.
- FAILS if its 95% CI upper bound < 0.333%.
- INDETERMINATE otherwise. Per crystal.

**Directional-asymmetry note is DEAD.** Old paper: 4 crossings / 1 recovery. Strict now: 4 recovery /
1 crossing. Inverts -> delete the note, do not reverse it.

**Ablation PRIMARY endpoint = proximity tail mass (<2.5 A on REGISTER_DEFINED)**, NOT anchor-identity.
Data justification: `max` anchor-identity 25.9% ~ other-rich 27.6% (does not discriminate); forward
fraction 47.0% ~ 48.2% (max carries NO orientation info -- decomposition stage 1 ratio 0.97, refutes the
dual-end hypothesis); but <2.5A proximity 6.4% vs 3.8% (~1.7x, powered at n~600/arm). Secondary =
strict register hits (descriptive; underpowered at 0.3%).
- max_drop12 regresses toward other-rich 3.8% => COMPOSITION (those residues carry the proximity edge).
- max_drop12 stays ~6.4% => the effect is COVERAGE / distributed, not the unique residues.
- **NEITHER** max_drop12 nor max_uniq12 elevated => effect requires the COMBINATION (interaction);
  drop12 loses the unique residues, uniq12 (12-16 contacts) too few to place -> plausible, report as such.

**Conditionality:** the ablations are interpretable ONLY if maxrep replicates. If maxrep is null (the
expected outcome given the k18 prior), the ablations dissect a non-existent effect -> report as moot.

**Native floor SUPERSEDED by fix8.** anchor-flip 0/1205 and offset-slippage 0/1205 on fix8 (in-register,
same pipeline) replaces the 0/~13 MD floor: criterion-2 false-negative < ~0.3% (not < 24.7%). (7) design
slippage 55.7% vs 0% floor at n>1200 -> unassailable.

**(5) corrected:** contact-only MEDIAN (register-defined) = 4.08 A vs fix2 = 2.43 A. So fix2 IS better
than contact-only (register supply starts at fix2, not fix4); the earlier "fix2 buys threading only" mixed
a median against a best-of-N floor and is withdrawn.

---
## ROUND-5 CORRECTIONS (committed before maxrep/ablation land)

**k18 "non-replication" prior RETRACTED -- it was a band artifact.** Old k18 (186 designs) rescored
under the CURRENT 310K DRG definition = 0 strict / 0 CI (best toDRG 1.85A). Old max (52) = 0/52, old
k24 = 0/52 -- EVERY old cell is 0 at 310K strict. The old "1.61%" was under 370K wider bands; the old
"hits" were DRG-crossings at 1.85-2.24A that fail the 310K bands. Old campaigns sampled only 52-186/cell,
underpowered to see a 0.3% effect. => No cross-campaign contradiction; maxrep expectation is **NEUTRAL**
(it is the first adequately-powered measurement, not a regression test).

**`max` 52x gap RESOLVED (proximity, threshold-dependent).** Enrichment of P(toDRG<=t): 2.5A 2.8x,
2.0A 3.1x, 1.58A 21.2x, 1.48A inf. Measuring at 2.5A (1.7x) understated it. P(anchor P9 | <=1.48)=1.00
-> anchor is not the bottleneck; `max` is special at DRG-PROXIMITY, ~21x at <=1.58A. Real steering signal.

**`max` strict hits = 4/1201, ALL 6AMU (recovery), 0 crossing, across 4 distinct batches.** So `max`
enables DRG RECOVERY, not redirection; not a single-batch artifact.

**ENDPOINT = register coordinate (toGIG - toDRG), register-specific.** Absolute <2.5A is NOT
register-specific (2.87A separation). regcoord>1.5: max 0.89% (5/559) vs other 0.03% (1/3126), ~28x.
CAVEAT: register-specific tail mass is rare (~4-7 events at n=600) -> the ablation is UNDERPOWERED for the
register question; that rarity is itself the finding. Report register-coordinate distribution + toDRG<=1.58
count (register-specific at 2.87A separation); hit count secondary.

**PHYSICAL groove gate pinned (pool.py), non-directional.** denovo physical-groove 89.7% vs null 20.9%
(stronger, non-directional placement result for 4). reverse & physical-groove = 7568 (vs 130 under the
directional RMSD gate) -> 3.4's mirror-image reverse population is REAL and large; the RMSD gate was hiding
groove-resident reverse designs. 3.4 headline SURVIVES; restate 4 with the physical (non-directional) gate.

All gates (extended/forward/RMSD-groove/physical-groove/register_defined) now pinned in pool.py with a
printed cross-tab; every round-2..5 number regenerates from that one file.
