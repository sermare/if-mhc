# UPDATE — reviewer-driven re-analysis of the completed campaign

**Status:** analysis memo only. **Nothing here is incorporated into `paper/paper.tex`** — the manuscript is
unchanged. This records what a multi-round methodological review found when the full converged campaign
(24,831 designs) was re-scored under the frozen 310 K definitions, plus the status of a follow-up GPU
experiment campaign. All numbers regenerate from `rev_analysis/` (`pool.py` pins the pool and every gate;
`ENDPOINTS.md` is the pre-registered log; `analyze_v*.py` produce the tables).

Read this as *candidate corrections*, not adopted claims. Decide separately whether any belong in the paper.

---

## Frozen analysis pool (`rev_analysis/pool.py`)

- **POOL** = full de novo corpus = **16,919** designs (every allcond150 cell that is neither templating
  `fix*` nor null). Supersedes the earlier 8,722 frozen subset.
- **REGISTER_DEFINED** = extended ∧ groove-placed ∧ forward-threaded = **7,121**. The only population where
  register is a meaningful quantity; all register comparisons run here.
- Hit (primary = **strict**): F-pocket anchor P9 ∧ Cα-RMSD to DRG ≤ **1.48 Å** (the well-converged band).
  CI-inclusive (secondary) uses ≤ 1.58 Å.
- Null control: **0 / 1,533**.

## Headline numbers, one pool, strict primary

| outcome | strict (≤1.48 Å) | CI-inclusive (≤1.58 Å) |
|---|---|---|
| DRG recovery (6AMU) | **4 / 16,919** | 7 / 16,919 |
| DRG crossing (6AM5) | **1 / 16,919** | 4 / 16,919 |

Redirection (crossing) under the primary definition is **a single design in 16,919** — a lone event, not
a rate.

---

## What held, what was corrected (rounds 2–7)

### Placement vs. steering — the clean result
- Conditioning reliably **places** a backbone in the groove: RMSD-groove **47.6%** (CI 46.9–48.4) of de
  novo vs **20.7%** (18.7–22.9) of null — 2.3×, well-powered, non-overlapping CIs. (A permissive physical
  centroid gate gives 89.7% vs 20.5% = 4.4×, but it admits poorly-folded backbones — within-gate median
  min-RMSD 7.3 Å — so 2.3× is the honest number.)
- It does **not** steer register (below).

### Conditioning breadth does not select register (retires old §3.2)
- On REGISTER_DEFINED, near-native tail mass (<2.0 Å) is flat: sparse (1–5 contacts) **0.82%**
  [0.49–1.19], rich (≥9) **0.79%** [0.52–1.09], CIs overlap. Same at <2.5 Å.
- The old binary "≥9 contacts 5/718 vs 3–5 contacts 0/587, Fisher p=0.068" rested on 5 events, a
  post-hoc threshold, and a tier that turned out to be a binning artifact (the mid tier is just
  {mhc, mhc_tcr1}). It does not survive the continuous readout.

### The register signal is one non-replicating scheme (recasts old §3.3)
- 4 of the 5 strict hits come from a single scheme, **`max`** (fullest 37-residue set), all 4 in the
  6AMU recovery arm. Exact multinomial homogeneity p<1e-4; post-hoc max-cell null p=0.0018.
- But `max` was the selected maximum of ~14 schemes (**winner's curse**). `max` is a strict superset of
  the other rich schemes, yet k24(24 contacts)=0/1203 while max(42)=4 — **non-monotonic, not a coverage
  threshold**.
- The advantage is threshold-dependent DRG-**proximity**, honest at ~**3.1× at 2.0 Å** (tighter thresholds
  are circular with the hit definition). It is **6AMU-only**: original max 6AMU **4/600 = 0.67%** vs 6AM5
  **0/601** (Fisher p=0.062).
- **Cross-campaign:** rescoring old-campaign cells under the current bands, the previously "special"
  18-contact cell (old 1.61%) → **0** — the old rate was a 370 K wider-band artifact. Old `max` on disk
  = 1/90 (best 1.07 Å): nonzero, consistent with new, but underpowered. **No informative prior; expectation
  neutral.**
- Best-of-26 cross-scheme spread (0.49 Å) sits inside the ~1.36 Å bootstrap-resampling spread → **not**
  evidence of scheme differences. Draws-vs-hits r=0.657 is near-tautological (Poisson offset kills it).

### Threading artifact — transferable, but stated in defensible form
- Raw Cα-RMSD to a crystal cannot distinguish a forward-threaded design from its reverse-threaded mirror
  image; ~47% of unconstrained de novo backbones carry the reverse label.
- Caveat added: the reverse population is largely *not* RMSD-close to either forward native, and no gate
  cleanly separates a genuine mirror-image conformer from a poorly-folded backbone. The actionable caution
  (classify threading before trusting RMSD-to-native) holds regardless.

### Templating localizes register (supports old §3.5, corrected numbers)
- On the **forward-threaded subset** (pooled medians are a bimodal mixture): fix8 **0.41 Å**, fix6 1.04,
  fix4 2.04, fix2 2.43, fix0 3.64. Register supply begins at **fix2** (better than the ~4.1 Å median of
  contact-only groove-placed forward designs).
- fix0→fix2 also fixes threading (fix0 ~49% forward, fix2+ 100%), so part of that first step is
  orientation, not register. The earlier "fix0 worst at ~11 Å" was a pooled-median artifact — withdrawn.

### Register-scoring criteria — validated
- **Anchor identity is not degenerate.** Native MD frames pick the correct F-pocket anchor in
  **1000/1000** frames (both crystals); crystal margin ~3.2–3.5 Å ≫ ~1 Å thermal envelope. Strong in-pipeline
  floor: fix8 (in-register, n=1205) anchor-flip **0%**.
- **Register slippage** (right shape, wrong phase): among REGISTER_DEFINED de novo, **55.7%** fit ≥0.5 Å
  better at a non-zero index offset, against a **0%** in-register floor (fix8, n=1205). Index-matched RMSD
  scores these as total misses.
- Native-frame proximity: 100% / 99.7% self-pass, 0% cross false-positive — the proximity criterion
  deflates nothing.

---

## GPU experiment campaign (follow-up generation, in progress)

Reviewer-designed experiments, spec-driven (`jobs/exp_*`, `jobs/exp_worker.sh`), on co_nilah/savio3_gpu
lowprio. Pre-registered endpoints in `rev_analysis/ENDPOINTS.md` (committed before scoring).

- **max-replication** (fresh seeds, N=600/crystal): pivotal test of whether the single-cell effect is real.
  **Interim 6AMU 0/469** (best toDRG 1.54 Å); formally indeterminate but trending to non-replication at the
  original 0.67%. Bridge control confirms **no batch confound** (new register-coord distribution matches old,
  p=0.15) — so the hit absence is genuine, not batch drift. Decision rule: 0/600 → "does not replicate at
  the originally observed rate; a smaller effect (~0.3%, winner's curse) is not excluded."
- **max-ablation** (drop-12 / uniq-12): both arms below `max`'s register-coordinate median → neither
  reproduces the effect (pre-registered "combination/interaction, or not real" branch). Moot if max fails.
- **Others queued** (paused behind maxrep): hotspot sweep 1–15 (OOD test), scrambled-hotspot control,
  region ablation, template-identity ladder, partial-diffusion barrier-crossing.

---

## If (and only if) any of this is adopted later

The defensible one-line summary the analysis points to: *receptor-side contact conditioning reliably places
a peptide backbone in the MHC groove but does not steer its register; register lives in a spatially confined
C-terminal signal that only explicit geometry supplies; redirection was observed once in 16,919 designs.*
Whether to reframe the manuscript around that is a separate decision — not taken here.
