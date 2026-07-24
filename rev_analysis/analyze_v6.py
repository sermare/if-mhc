import sys, glob, os; sys.path.insert(0, "/global/scratch/users/sergiomar10/if-mhc/rev_analysis")
import numpy as np, pandas as pd
from scipy import stats
import pool as P
DF = P.load(); dn = P.denovo(DF)
def hr(t): print("\n"+"="*72+f"\n{t}\n"+"="*72)
def cp(k, n):
    lo = stats.beta.ppf(.025, k, n-k+1) if k > 0 else 0.0
    hi = stats.beta.ppf(.975, k+1, n-k) if k < n else 1.0
    return 100*k/n, 100*lo, 100*hi

hr("A. max OLD vs NEW hotspot list (does 'old max 0/52' bear on the current cell?)")
def grab_max(fn):
    for ln in open(fn):
        p = ln.rstrip("\n").split("\t")
        row = "\t".join(p)
        if ("6AM5" in row) and ("\tmax\t" in row or "6AM5_max\t" in row):
            for f in p:
                if f.count(",") >= 5 and f.split(",")[0][0] in "ABDE":
                    return set(f.split(","))
    return set()
new = grab_max("jobs/allcond150_spec.tsv"); old = grab_max("jobs/maxcond_spec.tsv")
print(f"  new max n={len(new)}  old max n={len(old)}  identical={new==old}")
if new and old and new != old:
    print(f"    old-not-new: {sorted(old-new)}  new-not-old: {sorted(new-old)}")

hr("B. leftover OLD max designs on disk (free replication set?)")
d = pd.read_csv("outputs/denovo_scores/per_design.csv")
scored_max = d[d.cond == "max"]
disk = [f for f in glob.glob("outputs/**/*max*.pdb", recursive=True)
        if "traj" not in f and "allcond150" not in f and "L5_max" not in os.path.basename(f)]
print(f"  old max scored in per_design.csv: {len(scored_max)}")
print(f"  old max PDBs on disk (excl allcond150/L5_max): {len(disk)}")
print(f"  -> {'PDBs gone; only the 52 CSV rows survive' if len(disk)==0 else 'unscored PDBs exist = free replication'}")

hr("C. within-groove RMSD DISTRIBUTION (is 89.7% admitting junk?) + (4) under BOTH gates")
mind = DF[["toGIG", "toDRG"]].min(1)
DF["mind"] = mind
for gate in ("groove", "phys_groove"):
    for g in ("denovo", "null"):
        s = DF[(DF.g == g) & DF[gate]]
        if len(s):
            print(f"  [{gate}][{g}] n={len(s):5d}  min-RMSD median={s.mind.median():.2f}A  "
                  f"p90={np.percentile(s.mind,90):.2f}A  <8A={100*(s.mind<8).mean():.0f}%")
print("  (4) placement, both gates:")
for gate, lab in (("groove", "RMSD-groove(<8A, directional)"), ("phys_groove", "physical(centroid, non-dir)")):
    dr, dlo, dhi = cp(int(DF[(DF.g=="denovo")&DF[gate]].shape[0]), int((DF.g=="denovo").sum()))
    nr, nlo, nhi = cp(int(DF[(DF.g=="null")&DF[gate]].shape[0]), int((DF.g=="null").sum()))
    print(f"    {lab:<32} denovo {dr:.1f}% [{dlo:.1f}-{dhi:.1f}] vs null {nr:.1f}% [{nlo:.1f}-{nhi:.1f}]  ratio {dr/nr:.1f}")

hr("D. `max` 52x -- HONEST decomposition (only 2.0A is threshold-independent)")
mx = dn[dn.cond == "max"]; oth = dn[(dn.ncon >= 9) & (dn.cond != "max")]
for t in (2.5, 2.0, 1.58, 1.48):
    m = (mx.toDRG <= t).mean(); o = (oth.toDRG <= t).mean()
    note = "" if t == 2.0 else (" <- circular (= hit defn)" if t <= 1.58 else "")
    print(f"    P(toDRG<=%.2f): max %d/%d vs other %d/%d  ratio %s%s" % (
        t, (mx.toDRG<=t).sum(), len(mx), (oth.toDRG<=t).sum(), len(oth),
        f"{m/o:.1f}" if o else "inf", note))
nclose = (mx.toDRG <= 1.48).sum()
print(f"  P(anchor P9 | toDRG<=1.48) = {(mx.fpos[mx.toDRG<=1.48]==9).sum()}/{nclose}  <- N is the hits themselves")
print(f"  HONEST: modest proximity advantage ~3x at 2.0A, steepens near the band where counts are 4 vs 1.")

hr("E. maxrep power: crystal-specific rates + Fisher (effect is 6AMU-only)")
for cry in ("6AMU", "6AM5"):
    s = mx[mx.cry == cry]; k = int(((s.fpos==9)&(s.toDRG<=1.48)).sum()); r, lo, hi = cp(k, len(s))
    print(f"  max {cry}: strict {k}/{len(s)} = {r:.3f}% [{lo:.3f}-{hi:.3f}]")
a = int(((mx[mx.cry=='6AMU'].fpos==9)&(mx[mx.cry=='6AMU'].toDRG<=1.48)).sum()); na = (mx.cry=='6AMU').sum()
b = int(((mx[mx.cry=='6AM5'].fpos==9)&(mx[mx.cry=='6AM5'].toDRG<=1.48)).sum()); nb = (mx.cry=='6AM5').sum()
_, pf = stats.fisher_exact([[a, na-a], [b, nb-b]])
print(f"  6AMU {a}/{na} vs 6AM5 {b}/{nb}: Fisher p={pf:.3f} -> crystal-specificity suggestive, not established")
print("  -> maxrep decision rule should be against the 6AMU rate (0.67%); 6AM5 descriptive.")

hr("F. ABLATION endpoint = register-coordinate DISTRIBUTION (powered), not tail count")
DF["regcoord"] = DF.toGIG - DF.toDRG
rd = P.denovo(DF); rd = rd[rd.register_defined]
mxr = rd[rd.cond == "max"]; othr = rd[(rd.ncon >= 9) & (rd.cond != "max")]
u, pmw = stats.mannwhitneyu(mxr.regcoord, othr.regcoord, alternative="greater")
print(f"  register coord (toGIG-toDRG) on register_defined:  max n={len(mxr)} median={mxr.regcoord.median():+.2f} "
      f"p10={np.percentile(mxr.regcoord,10):+.2f} p90={np.percentile(mxr.regcoord,90):+.2f}")
print(f"                                                     other n={len(othr)} median={othr.regcoord.median():+.2f} "
      f"p10={np.percentile(othr.regcoord,10):+.2f} p90={np.percentile(othr.regcoord,90):+.2f}")
print(f"  distribution shift (MWU, max>other): p={pmw:.3f}  <- powered (thousands), use for ablation not tail count")
print("V6_DONE")
