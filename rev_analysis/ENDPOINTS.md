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

---
## ROUND-6 CORRECTIONS

**max old == new: identical 37-residue set** -> old-max evidence bears on the current cell.

**Old max is NOT zero (correcting round-5).** A fresh rescore of ~90 old max-conditioned designs on disk
= 1/90 strict at 310K, best toDRG 1.07A (same best as new max). Weak, underpowered (UB 6%), but nonzero and
CONSISTENT with new max 4/1201 -> mild POSITIVE cross-campaign support. maxrep is the powered test.
(The round-5 "old 0/52" used a 52-row CSV subset that missed this design; disk rescore is authoritative.)

**(4) report BOTH gates; RMSD-groove is PRIMARY.** RMSD-groove (directional): denovo 47.6% vs null 20.7%,
ratio 2.3x, within-gate median min-RMSD 3.98A (clean). Physical-groove (non-directional): 89.7% vs 20.5%,
ratio 4.4x, but within-gate median 7.28A / p90 15.78A / only 51% <8A -> ADMITS poorly-folded backbones,
overstates placement. Lead with 2.3x (clean); report 4.4x as the permissive non-directional bound.

**3.4 mirror-image claim WEAKENED symmetrically.** The physical gate that "rescues" reverse-in-groove
(7,568) also admits junk (median 7.28A), so those are mostly structurally-poor, not clean mirror images.
No gate cleanly isolates "well-formed N-term-in-F-pocket". Transferable claim survives only in the weaker
form: ~half of de-novo backbones are not RMSD-close to a forward native and F-pocket argmin labels them
reverse; clean-mirror vs junk is unresolved without a reverse reference.

**max 52x is NOT resolved.** Only the hit-independent 2.0A threshold is honest: ~3.1x. 1.58A (21x) and
1.48A (inf) are circular (= the hit definition). P(anchor|<=1.48)=4/4 (the hits themselves). Defensible
sentence: "modest, threshold-dependent proximity advantage (~3x at 2.0A) that steepens near the acceptance
band, where counts (4 vs 1) are too small to characterize." Not "resolved."

**maxrep DECISION RULE -> 6AMU ARM ONLY.** Effect is crystal-specific: 6AMU 4/600 = 0.667%
[0.18-1.70], 6AM5 0/601 (Fisher p=0.062, suggestive not established). Rule: HOLDS if fresh 6AMU strict
rate lower-CI > 0.006%; FAILS if upper-CI < 0.667%; else indeterminate. 6AM5 is a separate descriptive arm.

**Redirection = n=1 (strict).** Corpus strict = 4 recovery + 1 crossing; max contributes 0 crossings. The
entire redirection result is ONE design in 16,919. Abstract must say so ("redirection observed once in
16,919 draws"); the "Recovering and Redirecting" title oversells the n=1 half.

**Ablation endpoint = register-coordinate DISTRIBUTION (powered), tail count secondary.** register coord
(toGIG-toDRG) on register_defined: max median +0.70 (p10 -0.43) vs other +0.59 (p10 -0.51); MWU max>other
p<0.001 -- a small but significant DRG-ward shift at n=559 vs 3126. Use median/p10/shift for the ablation;
if it still can't discriminate, say so rather than run it for form.

---
## ROUND-7 CORRECTIONS

**BATCH-CONFOUND BRIDGE CONTROL (free) -- verdict IS interpretable.** New maxrep register-coordinate
distribution (groove&forward, per crystal) matches old max: 6AMU new median +0.46 vs old +0.32 (MWU p=0.15);
6AM5 new +0.76 vs old +0.64 (p=0.51). NO downward batch shift -> the 0/463 hit absence is genuine
non-replication of the tight-band hit, NOT a batch artifact. (The ablation -0.45 is therefore a real hotspot
effect, not drift.)

**Near-misses carry the story.** New maxrep 6AMU P9-seated: best toDRG 1.54A, #<=1.48=0, <=1.58=1, <=1.8=4.
The register-coordinate DISTRIBUTION replicates; only the rare tight-band hit does not. Report the near-miss
distribution, not just the binary call (the band sits 0.02A from the data).

**"FAILS" reframed (winner's curse).** 0/600 -> upper CI 0.61% rules out the ORIGINAL 0.667% point estimate,
NOT the effect; a true rate ~0.3% (still ~50x background) is not excluded. `max` was the max of ~14-40
cells -> selected maxima are upward-biased. Honest statement: "max does not replicate at its originally
observed rate; a smaller effect is not excluded." Report the POOLED estimate (old+new combined) and Fisher
(4/600 vs 0/600, p~0.06), not the replication verdict alone.

**Protocol note (for the record):** a partial-N interim read (maxrep 463/512) was taken and used to shape
expectations. The decision rule was fixed in advance and no stopping decision was made on it, but this is a
departure from "score only at min-N" and is logged here for transparency.

---
## exp_partial PRE-REGISTRATION (committed BEFORE the scorer is wired / any cell scored)
Scorer: py/score_q30_basins.py (per-design toGIG/toDRG in the 6AMU groove frame -> basin occupancy).
**PRIMARY endpoint = the DIFFERENCE between the own-register and cross-register conditioning arms in
target-basin occupancy, as a function of partial_T.** Single-arm occupancy is uninformative (low partial_T
recovers the seed by construction; high partial_T degrades to de novo). Steering signal = cross-arm occupancy
of the TARGET (alternate) basin rising above the own-arm control at matched partial_T.
**IDENTIFIABILITY CHECK (explicit):** seed native DRG, LOW partial_T, DRG-side conditioning. If the model
cannot hold DRG under its own conditioning at low partial_T, there is no DRG basin to steer into -- which
would explain every negative in the project. Report this before any crossing claim.
**Sampling range:** the prior seed-echo result (toGIG 0.07, zero variance = identity map, likely partial_T
too low) says the frozen regime is uninformative. Sample DENSELY where output starts to move -- ~15-30 --
not the frozen low end. Report per partial_T with n and CI; score only at each cell's min-N.

---
## ROUND-8 CORRECTIONS

**Ablation is NEGLIGIBLE, not "composition" (report effect size, not p).** On the register coordinate,
register-defined, every contrast moves the coordinate by single-digit % of the 2.87A inter-conformation
separation, and AUC ~ 0.5:
| contrast | AUC | Cliff's delta | dMedian | % of 2.87A | p |
|---|---|---|---|---|---|
| breadth rich vs sparse | 0.523 | +0.047 | +0.032A | 1.1% | 0.002 |
| max vs other-rich | 0.571 | +0.141 | +0.105A | 3.7% | <1e-4 |
| drop12 vs max | 0.454 | -0.092 | -0.057A | 2.0% | 0.017 |
UNIFYING STATEMENT (replaces "breadth doesn't matter"): *contact conditioning moves the register coordinate
by single-digit percentages of the distance separating the two conformations, regardless of composition or
coverage.* Every p is a statement about n (thousands), not effect size. The batch-shift concern is
WITHDRAWN (it was built on the wrong-population -0.45/-0.43; register-defined shows no shift). Binary
ablation (3/720 vs 0/786) remains EXPLORATORY.

**maxrep POOLED (the number for the record).** original 6AMU 4/600=0.67%, replication 1/611=0.16%, Fisher
p=0.21 (NOT a significant regression). POOLED both campaigns = 5/1211 = 0.41% [CP 0.13-0.96%] vs pooled-other
~0.006%. Report pooled + state `max` was the selected max of ~40 cells (upward-biased). Near-misses
straddle the band (best P9-seated toDRG: maxrep 1.26/1.54/1.67; uniq12 1.27/1.38/1.45/1.49): the binary
dichotomizes a dense continuum -- report the near-miss distribution alongside every count.

**CANONICAL fix ladder (Table 7) -- per crystal x FORWARD-only; retires the 4 floating fix0 values
(14.1 / 13.34 / 3.64 / 7.48-15.43).**
| rung | 6AM5 fwd% | 6AM5 med | 6AMU fwd% | 6AMU med |
|---|---|---|---|---|
| fix0 | 49% | 3.87A | 49% | 3.42A |
| fix2 | 100% | 2.48A | 100% | 2.38A |
| fix4 | 100% | 2.02A | 100% | 2.06A |
| fix6 | 100% | 1.22A | 100% | 0.92A |
| fix8 | 100% | 0.43A | 100% | 0.40A |
Ladder is PER SCAFFOLD (6AM5 != 6AMU at fix6/fix8); fix0's 49% forward is the bimodality. State per scaffold.

**Prior partial_T work (agent-verified).** 8 earlier partial_T campaigns existed (pd_sweep, submit_pd_sweep,
q30 x2, md_seeds, marathon, cross, cross2, focus) but ALL ran on the remote /home/ubuntu cloud host and were
NEVER synced; no design PDBs are local (verified). Remote UNREACHABLE (15s check). Surviving value: which
partial_T to sample + score_q30_basins.py exists. Seed-echo "freezes to point" (toGIG 0.07) is LIKELY an
identity-map artifact (zero variance, = fixall 0.07) -> null diagnostic, NOT a finding, pending partial_T
verification. The genuine graded crossing curve was never completed; exp_partial is the first under frozen defs.

---
## ROUND-8 ADDENDUM (precision fixes from review)

**Preserve the SIGN of each contrast; do not collapse into a 0.45–0.57 range.** The AUC<0.5 entry is
**drop12 vs max = 0.454** (Cliff −0.092): drop12 sits BELOW max on the register coordinate = LESS DRG-like =
the **composition-EXPECTED** direction (remove the 12 residues → regress toward GIG), NOT a directional
inversion. All three contrasts point the expected way (rich>sparse +0.047; max>other +0.141; drop12<max
−0.092); all are trivial in magnitude (1–4% of the 2.87 Å separation). List them with signs, not as a range.

**fix0 is a different KIND of point on the ladder, not the bottom of one curve.** Forward fractions
confirmed per crystal: fix0 49%/49%, fix2 100%/100%, fix4 100%/100%, fix6 100%/100%, fix8 100%/100%. So even
after forward-only restriction, the fix0→fix2 step still compares a rung where ~half the population was
discarded (fix0) against rungs where none was (fix2+); part of that step remains a threading effect. The
clean geometry dose-response is **fix2→fix8**; fix0 should be annotated separately, not read as continuous
with it.

**Seed-echo identity-map diagnosis is CIRCUMSTANTIAL, not established.** It rests on toGIG 0.07 with zero
variance coinciding exactly with fixall (0.07). The airtight check — the actual partial_T from the `.trb`
plus a direct input-vs-output coordinate diff — is impossible because those runs are remote and unreachable.
Record it as *likely* identity map (partial_T too low), not proven.

**exp_partial seed-pairing CONFIRMED (the primary endpoint isolates conditioning).** pd{T}_own and
pd{T}_cross share the identical seed (empty input_pdb → {crystal}_trim.pdb) and identical contig
(`... C1-10`); they differ ONLY in ppi.hotspot_res (own = own-register ranked list, cross = other-register
ranked list). So the own-vs-cross basin-occupancy difference vs partial_T isolates conditioning, not seed.

**Other partial_T data — nothing further to present.** The 8-campaign inventory + the two surviving
conclusions (seed-echo identity-map; fix-motif ladder, already in per_design.csv) are the complete local
record. No additional scoreable partial_T designs exist locally; the graded crossing curve is exp_partial
(pending generation), which is starved on lowprio at this writing.
