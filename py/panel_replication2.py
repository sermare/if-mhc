#!/usr/bin/env python3
"""Follow-ups: the length-confound sign flip, and notebook 05's SKEMPI ddG test.

1. The panel reported pooled Pearson r=+0.36 between peptide length and recovery.
   This set gives r=-0.41. The panel was all class-I 8-10mers; this set adds five
   class-II complexes carrying 12-13mers, so the comparison is only like-for-like
   after restricting to the same length/class regime.

2. Notebook 05 asks whether measured binding effects agree with recovery/entropy.
   The 28 complexes here carry 374 SKEMPI ddG records, so that test can be run
   directly rather than on a single structure.
"""
import os, re
import numpy as np
import pandas as pd
from scipy import stats

ROOT = "/global/scratch/users/sergiomar10/if-mhc"
R = pd.read_csv(f"{ROOT}/outputs/skempi_if/panel_replication_positions.csv")
man = pd.read_csv(f"{ROOT}/inputs/skempi/manifest.csv")
CLASS2 = set(man[man.pep_fused_to_mhc == 1]["complex"]) | {"3QIB_ABP_CD", "4OZG_ABJ_GH"}


def hdr(t): print(f"\n{'='*82}\n{t}\n{'='*82}")


hdr("FOLLOW-UP 1 -- is the length/recovery sign flip driven by class II 12-13mers?")
cell = R[R.group != "P1"].groupby(["complex", "model", "arm"]).agg(
    recovery=("recovery", "mean"), L=("L", "first")).reset_index()
cell["class2"] = cell["complex"].isin(CLASS2)
print(f"  class II complexes in this set: {sorted(CLASS2)}")
print(f"  length composition: {dict(man.pep_len.value_counts().sort_index())}\n")
for label, sub in [("ALL (as reported)", cell),
                   ("class I only", cell[~cell.class2]),
                   ("class I, 9-mers only", cell[(~cell.class2) & (cell.L == 9)]),
                   ("class I, 9+10mers", cell[(~cell.class2) & cell.L.isin([9, 10])])]:
    for arm in ["full"]:
        s = sub[sub.arm == arm]
        if s.L.nunique() < 2:
            print(f"  {label:24s} arm={arm}  (only one length present, r undefined) n={len(s)}")
            continue
        r, p = stats.pearsonr(s.L, s.recovery)
        print(f"  {label:24s} arm={arm}  r={r:+.3f}  p={p:.3g}  n={len(s)}  "
              f"lengths={sorted(s.L.unique())}")

hdr("FOLLOW-UP 2 (nb05) -- do measured SKEMPI ddG values agree with recovery/entropy?")
mut = pd.read_csv(f"{ROOT}/inputs/skempi/skempi_tcr_pmhc_mutations.csv")
man_i = man.set_index("complex")
rows = []
pat = re.compile(r"^([A-Z])([A-Za-z])(-?\d+[A-Za-z]?)([A-Z])$")
for _, m in mut.iterrows():
    cid = m["#Pdb"]
    if cid not in man_i.index:
        continue
    pep_ch = man_i.loc[cid, "pep_chain"]
    resids = str(man_i.loc[cid, "pep_resids"]).split(";")
    try:
        ddg = (float(m["Affinity_mut_parsed"]), float(m["Affinity_wt_parsed"]))
        if not (ddg[0] > 0 and ddg[1] > 0):
            continue
        RT = 0.001987 * 298.15
        ddG = RT * np.log(ddg[0]) - RT * np.log(ddg[1])   # positive = weaker binding
    except Exception:
        continue
    for tok in str(m["Mutation(s)_cleaned"]).split(","):
        g = pat.match(tok.strip())
        if not g:
            continue
        wt, ch, num, mt = g.groups()
        if ch != pep_ch or num not in resids:
            continue
        rows.append(dict(complex=cid, pos=resids.index(num) + 1, wt=wt, mt=mt, ddG=ddG))

M = pd.DataFrame(rows)
print(f"  {len(mut)} SKEMPI records over {mut['#Pdb'].nunique()} complexes; "
      f"{len(M)} single-residue mutations land on the designed epitope "
      f"({M['complex'].nunique() if len(M) else 0} complexes)")
if len(M) < 10:
    print("  -> too few epitope mutations to test; SKEMPI TCR/pMHC mutations are"
          "\n     overwhelmingly on the TCR, not the peptide. nb05's test cannot be"
          "\n     replicated at panel scale on this set.")
else:
    agg = M.groupby(["complex", "pos"]).ddG.agg(["mean", "max", "count"]).reset_index()
    agg.columns = ["complex", "pos", "ddG_mean", "ddG_max", "n_mut"]
    for arm in ["full", "notcr"]:
        rr = (R[(R.arm == arm) & (R.group != "P1")]
              .groupby(["complex", "pos"])
              .agg(recovery=("recovery", "mean"), ent20=("ent20", "mean")).reset_index())
        j = agg.merge(rr, on=["complex", "pos"])
        if len(j) < 10:
            continue
        r1, p1 = stats.spearmanr(j.ddG_mean, j.recovery)
        r2, p2 = stats.spearmanr(j.ddG_mean, j.ent20)
        print(f"\n  arm={arm}  n={len(j)} (complex,position) cells with measured ddG")
        print(f"    ddG vs recovery : Spearman rho={r1:+.3f} p={p1:.3g}"
              f"   -> {'agrees (higher ddG = better recovered)' if (r1>0 and p1<0.05) else 'no significant agreement'}")
        print(f"    ddG vs entropy  : Spearman rho={r2:+.3f} p={p2:.3g}"
              f"   -> {'agrees (higher ddG = lower entropy)' if (r2<0 and p2<0.05) else 'no significant agreement'}")
        j.to_csv(f"{ROOT}/outputs/skempi_if/skempi_ddg_vs_recovery_{arm}.csv", index=False)

hdr("FOLLOW-UP 3 -- replicate structures sharing an identical epitope (nb04 design)")
grp = man.groupby("pep_seq")["complex"].apply(list)
grp = grp[grp.map(len) > 1]
print(f"  {len(grp)} replicate groups in this set:")
rr = R[(R.group != "P1")].groupby(["complex", "arm"]).recovery.mean().unstack()
for pep, members in grp.items():
    mem = [c for c in members if c in rr.index]
    if len(mem) < 2:
        continue
    v = rr.loc[mem, "full"]
    print(f"    {pep:15s} n={len(mem)}  recovery(full) range {v.min():.3f}-{v.max():.3f} "
          f"spread={v.max()-v.min():.3f}   {', '.join(mem)}")
spreads = []
for pep, members in grp.items():
    mem = [c for c in members if c in rr.index]
    if len(mem) >= 2:
        spreads.append(rr.loc[mem, "full"].max() - rr.loc[mem, "full"].min())
allspread = rr["full"].max() - rr["full"].min()
print(f"\n  mean within-group spread = {np.mean(spreads):.3f}   "
      f"across the whole panel = {allspread:.3f}")
print(f"  -> identical peptide, different crystal still moves recovery by "
      f"{np.mean(spreads):.3f} on average ({100*np.mean(spreads)/allspread:.0f}% of the full range)")
