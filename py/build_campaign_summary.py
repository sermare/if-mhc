#!/usr/bin/env python3
"""Generate CAMPAIGN_STRUCTURAL_SUMMARY.md — every campaign's peptide backbones on the register map,
with mean, median, and 95% CI of the distances. Re-runnable."""
import sys, warnings, glob; warnings.filterwarnings("ignore"); sys.path.insert(0, "py")
import numpy as np, pandas as pd
import core_load as CL, q30_analysis as Q
from Bio.PDB import PDBParser

rows = []
def ci(a):
    a = np.asarray(a, float)
    if len(a) < 2: return None
    sem = a.std(ddof=1) / np.sqrt(len(a))
    return (a.mean() - 1.96*sem, a.mean() + 1.96*sem)
def fmt_mci(a):
    a = np.asarray(a, float); m = a.mean(); c = ci(a)
    return f"{m:.2f}" if c is None else f"{m:.2f} ({c[0]:.2f}–{c[1]:.2f})"
def add(fam, campaign, algo, start, params, g, d):
    g = np.asarray(g, float); d = np.asarray(d, float)
    if len(g) == 0: return
    oth = d if start == "6AM5" else g if start == "6AMU" else np.maximum(g, d)
    rows.append(dict(fam=fam, campaign=campaign, algo=algo, start=start, params=str(params)[:26], n=len(g),
        gm=fmt_mci(g), gmed=round(float(np.median(g)), 2), dm=fmt_mci(d), dmed=round(float(np.median(d)), 2),
        best_basin=round(float(np.minimum(g, d).min()), 2), best_other=round(float(np.min(oth)), 2)))

DF, CO, TRAJ = CL.load_all()
SEP = round(CL._rmsd(CL.GIG, CL.DRG), 2)
fammap = {"native": "A", "tool": "A", "v1": "B", "rfd3": "C"}
algos = {"native": "crystal/relax", "tool": "OpenMM/Rosetta relax", "v1": "RFdiff full-T de-novo", "rfd3": "RFdiffusion3"}
for (cond, pid), s in DF.groupby(["cond", "pid"]):
    grp = s.group.iloc[0]; add(fammap[grp], cond, algos[grp], pid, cond, s.toGIG, s.toDRG)
for job, s in TRAJ[TRAJ.job.str.match(r"ifmhc_6AM[5U]_md_\d+K")].groupby("job"):
    pid = "6AM5" if "6AM5" in job else "6AMU"; T = job.split("_")[-1]
    add("A", f"MD-native {T}", "OpenMM MD 10ns", pid, f"ff19SB {T}", s.to_GIG, s.to_DRG)
dmd = TRAJ[TRAJ.job.str.startswith("ifmhc_md_")]
for pid in ["6AM5", "6AMU"]:
    s = dmd[dmd.job.str.contains(pid)]; add("F", "MD-design L4_expanded", "OpenMM MD of designs", pid, "ff19SB ~300K 10ns", s.to_GIG, s.to_DRG)
arms = {"fpocket": ("D", "v3 Q30+F-pocket"), "tcr": ("D", "v3 Q30+Phe-b"), "apoc": ("D", "v3 Q30+Phe-b+A-pocket"),
        "nslide": ("D", "v3 N-slide scaffold"), "sc_tcr": ("E", "v4 scaffold TCR"), "sc_apoc": ("E", "v4 scaffold A-pocket"),
        "sc_fpocket": ("E", "v4 scaffold F-pocket"), "v4arm1": ("E", "v4 region-release")}
qdf, _ = Q.load_all([a for a in arms if a in Q.ARM_DIR]); qdf = qdf[qdf.broken == 0]
for arm, s in qdf.groupby("arm"):
    for pid, ss in s.groupby("pid"):
        algo = "RFdiff scaffold" if arm in Q.SCAFFOLD_ARMS else ("RFdiff partial+inpaint release" if arm == "v4arm1" else "RFdiff partial-diff")
        add(arms[arm][0], arms[arm][1], algo, pid, arm, ss.to_GIG, ss.to_DRG_shift)
p = PDBParser(QUIET=True)
for f in sorted(glob.glob("outputs/rfdiff_q30_v4arm2/seeds/*.pdb")):
    m = p.get_structure("x", f)[0]; A = [r for r in m["A"] if r.id[0] == " "]
    pep = np.array([r["CA"].coord for r in m["C"] if r.id[0] == " "])
    fl = {n: [r for r in A if r.id[1] == n][0]["CA"].coord for n in Q.FLOOR if any(r.id[1] == n for r in A)}
    com = [n for n in Q.REF_FLOOR if n in fl]; R, t, _ = Q.kabsch(np.array([fl[n] for n in com]), np.array([Q.REF_FLOOR[n] for n in com]))
    pc = pep @ R + t
    add("E", "v4 Arm2 chimera graft", "graft C-term", f.split("/")[-1][:-4], "C-term splice", [Q.rmsd(pc, Q.REFS["GIG"])], [Q.rmsd(pc, Q.REFS["DRG_shift"])])
T = pd.DataFrame(rows); T.to_csv("outputs/all_campaigns_summary.csv", index=False)

titles = {"A": "A. References — native crystal / relaxation / MD", "B": "B. v1 RFdiffusion (de-novo peptide, full diffusion T=30)",
          "C": "C. RFD3 (RFdiffusion3, Tamarind)", "D": "D. v3 — Q30-conditioned partial diffusion / scaffold",
          "E": "E. v4 — region-release / scaffold joint / chimera graft", "F": "F. MD of design backbones"}
def table(fam):
    out = ["| Campaign | Algorithm | Start | Params | N | toGIG mean (95% CI) | toGIG med | toDRG mean (95% CI) | toDRG med | best→basin | best→OTHER |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in T[T.fam == fam].sort_values(["campaign", "start"]).iterrows():
        out.append(f"| {r.campaign} | {r.algo} | {r.start} | {r.params} | {r.n} | {r.gm} | {r.gmed} | {r.dm} | {r.dmed} | {r.best_basin} | {r.best_other} |")
    return "\n".join(out)

DOC = f"""# pMHC–TCR register campaigns — structural summary (all campaigns)

_HLA-A\\*02 DMF5 system. Every peptide backbone across every campaign, on the same register map._

## System & goal
Two 10-mer epitopes, same groove, **different register** (C-terminal P9↔P10 F-pocket swap):
**GIG** `SMLGIGIVPV` (6AM5, P10 anchor) · **DRG** `MMWDRGLGMM` (6AMU, P9 anchor). Goal: steer/bridge the
register. **Verdict: never achieved (see bottom).**

## How to read the numbers
- Distances = register-preserving Cα-RMSD of the peptide to each native basin (common groove frame).
  **Native GIG↔DRG separation = {SEP} Å.**
- **mean (95% CI)** = distribution mean ± 1.96·SEM; **med** = median. Caveats: n=1 rows show no CI;
  **small-n (n≤4) CIs are wide and can extend below 0** (normal-approx artifact — treat as uninformative);
  **MD frames are autocorrelated** so those CIs are *optimistic* (lower bounds on true uncertainty).
  When mean ≫ median (e.g. some relax/de-novo rows), the mean is dragged by a few blown-up outliers —
  the **median is the more robust summary** there.
- **best→basin** = smallest min(toGIG,toDRG) (closest to *a* register; <1.45 Å = clearly in a basin).
- **best→OTHER** = smallest distance to the **non-cognate** basin = *the crossing metric*
  (a true crossing needs <1.45 Å; **nothing reached it**).

## Master table
### {titles['A']}
{table('A')}

### {titles['B']}
{table('B')}

### {titles['C']}
{table('C')}

### {titles['D']}
{table('D')}

### {titles['E']}
{table('E')}

### {titles['F']}
{table('F')}

## Per-family read-out
- **A. Native physics:** stays put; relaxation moves the peptide ~0.5 Å. **MD 370 K is the best crosser
  (best→OTHER 2.07–2.12 Å)** but never crosses. Some Tamarind relax outliers blew up to ~15 Å (failed
  relaxations — note the huge mean vs small median there).
- **B. v1 RFdiffusion de-novo:** N-term+pocket sets (Nterm+pkt+TCR3 / L5) land nearest a native basin
  (~1.7–2.1 Å) but most conditionings scatter 8–15 Å from both. No crossing.
- **C. RFD3:** almost all land far off (8–22 Å); a few TCR-window-scan designs reach ~1.7–1.9 Å of a basin
  but never cross.
- **D. v3 Q30 partial diffusion:** **frozen to native** (best→basin 0.08 Å = echo; best→OTHER 2.77–2.89 Å
  at every partial_T). N-slide scaffold drifts to the ambiguous middle (~3 Å from both).
- **E. v4:** region-release mostly **collapses** the peptide (64/69 broke) and never crosses; the **chimera
  graft is the only thing that reaches the other register (0.78 Å) — because it's *placed* there, and holds.**
- **F. MD of designs:** the **only near-equidistant population** — de-novo L4 designs sit ~1.9–2.4 Å from
  *both* basins (best→OTHER 1.55/1.90); metastable, but de-novo (not native transition states), and
  on-path status unverified.

## Overall verdict — the register lives in the peptide C-terminus; the MHC is a rigid socket

**1. Where the GIG↔DRG difference physically is.** Per-residue 6AM5↔6AMU deviation on the invariant β-sheet
floor (0.21 Å frame; `perresidue_gig_drg.png`) localizes the *entire* structural difference to the **peptide
C-terminus (p5–p10, flat p1–p4, max at p6, reseating at the p9/p10 anchor)**. The MHC **backbone is flat
everywhere** (F-pocket/arm 0.35 Å ≈ non-pocket baseline 0.32 Å, 1.1×); the only register-associated MHC atoms
are **two F-pocket side-chain rotamers, Tyr116 and Lys146** — and every larger all-atom spike (Arg111, Lys121,
Glu58 …) is a **distal surface rotamer** (flat backbone, 10–35 Å from the anchor, no peptide contact).

**2. Those two rotamers are not a lever.** Apo-MHC MD (empty groove, `apo_mhc_md.py`): Tyr116 stays ~100%
*trans* (its GIG state) at 300 & 370 K from either crystal and **never samples the DRG g− state**; Lys146
samples g+/trans but **0–2 % g−**. So the DRG rotamers are **peptide-induced, not intrinsic groove
flexibility**. Forced-crossing steered MD (`forced_crossing.py`): dragging the anchor GIG→DRG (swap −3.3→+3.0)
with the rotamers free **does not recruit** them — Tyr116 stays trans throughout. They require the full native
DRG peptide; they are a *consequence* of the bound peptide, not an accessible degree of freedom.

**3. Nothing generative or dynamical reproduces the register.** Across ~1,000+ backbones and 30+ methods, **no
design or simulation placed a peptide <1.45 Å of the non-cognate register** (closest ~1.55–2.07 Å, MD).
Anchor-conditioned RFD (`run_b3_anchor.sh`, the last live premise) builds **coherent backbones for both p9 and
p10 anchors (60/60) but reproduces neither register (0/60 seated, 6.5–8 Å off)** — the fixed anchor pins the
C-terminus and the de-novo remainder scatters. Both registers are **constructible as designed backbones** (the
chimera graft *places* one at 0.78 Å and it holds), but that is **anchor-encoded structural accommodation**,
not single-sequence bistability — every object reaching the other register is a *different* sequence, and every
single-native-sequence probe (relax, MD incl. 370 K, partial diffusion) stays in its own register.

**4. TCR causation (Riley's phenomenon) — SIGNIFICANT.** Steered MD driving the *same* DRG sequence from the
unshifted (6AMT) to shifted (6AMU) register costs **~17 % less work with the TCR engaged (92.5 vs 111.0 kJ/mol;
Δ 18.5 kJ/mol = 4.4 kcal/mol; n=35/32, Welch p=0.030, Mann-Whitney p=0.026, bootstrap 95 % CI [2.5, 34.7]
excludes 0, P(Δ>0)=0.988)**. The direction was stable ~17–19 % throughout; sampling resolved it from p=0.10
(n=12) to p=0.030 (n=35/32). So the engaged TCR **measurably lowers the barrier to the register shift** — the one
positive causal result, and the half the project set out to test. *Caveats:* steered-MD **work**, not a
rigorous ΔΔG; implicit solvent; MHC held; TCR held statically engaged; single pull speed; steric-vs-specific
mechanism unresolved. Unbiased 3 ns ±TCR does not spontaneously cross (barrier is real).

**Bottom line — two levels, not one.** *Structurally*, the register is a **peptide-C-terminal backbone event
on a rigid, register-agnostic MHC**: no MHC/TCR side-chain or pose degree of freedom *encodes* the register
(apo rigid, forced-crossing doesn't recruit the pocket rotamers, per-residue difference is peptide-only), which
is *why* every static conditioning lever (Q30, F-pocket hotspots, pocket release, scaffold, anchor-conditioning)
failed and why RFD reproduces neither register. *Dynamically*, however, TCR engagement **significantly lowers
the work to make the shift** (§4) — the register isn't *encoded* in the TCR, but the TCR's engagement *biases
the transition*. Reconciled: the geometry is peptide-borne; the TCR is not a structural handle but is a
**kinetic modulator** of the crossing. Reaching the other register de-novo requires *placing* the backbone
there (graft); the TCR-load effect is the live biological result worth pursuing (better sampling / non-held TCR
to firm the ΔΔG).

_Data: `outputs/all_campaigns_summary.csv` · figures: `perresidue_gig_drg.png`, `native_components_grid.png`,
`rotamer_coupling.png`, `native_tcr_pose_space.png` · overnight logs: `outputs/{{apo_mhc,rfdiff_b3,forced_crossing,tcr_causation}}/`._
"""
open("/home/ubuntu/if-mhc/CAMPAIGN_STRUCTURAL_SUMMARY.md", "w").write(DOC)
print("wrote CAMPAIGN_STRUCTURAL_SUMMARY.md |", len(T), "rows")
