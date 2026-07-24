import sys; sys.path.insert(0, "/global/scratch/users/sergiomar10/if-mhc/rev_analysis")
import numpy as np, pandas as pd
from scipy import stats
import pool as P
DF = P.load(); dn = P.denovo(DF); rd = P.regdef(DF)
rng = np.random.default_rng(0)
def cp(k, n):
    lo = stats.beta.ppf(.025, k, n-k+1) if k > 0 else 0.0
    hi = stats.beta.ppf(.975, k+1, n-k) if k < n else 1.0
    return 100*k/n, 100*lo, 100*hi
def hr(t): print("\n"+"="*72+f"\n{t}\n"+"="*72)
def toown(s): return np.where(s.cry == "6AM5", s.toGIG, s.toDRG)

hr("2 (CORRECTED). BREADTH effect on REGISTER_DEFINED only, via TAIL MASS (not MW)")
rd = rd.assign(toown=toown(rd))
def tier(n):
    if n <= 5: return "sparse(1-5)"
    if n <= 8: return "mid(6-8)"
    return "rich(>=9)"
rd["tier"] = rd.ncon.map(tier)
print(f"  REGISTER_DEFINED de-novo n={len(rd)}  (register only meaningful here)")
print(f"  {'tier':<12}{'n':>6}{'<2.0A':>9}{'<2.5A':>9}{'p1.0':>7}{'p0.5':>7}  (tail mass = real signal, not bulk shift)")
for t in ("sparse(1-5)", "mid(6-8)", "rich(>=9)"):
    s = rd[rd.tier == t]; n = len(s)
    if n == 0: continue
    f20 = (s.toown < 2.0).mean()*100; f25 = (s.toown < 2.5).mean()*100
    p1 = np.percentile(s.toown, 1); p05 = np.percentile(s.toown, 0.5)
    # bootstrap CI on the <2.0A fraction
    bs = [ (s.toown.values[rng.integers(0,n,n)] < 2.0).mean()*100 for _ in range(2000)]
    print(f"  {t:<12}{n:>6}{f20:>7.2f}% [{np.percentile(bs,2.5):.2f}-{np.percentile(bs,97.5):.2f}]{f25:>7.2f}%{p1:>7.2f}{p05:>7.2f}")
print("  -> if tail mass (<2.0/2.5A) is flat across tiers, breadth carries NO register signal even where defined")

hr("3 (CORRECTED). Is `max` special? exact multinomial homogeneity + post-hoc max-cell null")
tab = dn.groupby("cond").agg(N=("hit","size"), k=("hit","sum"), ncon=("ncon","first")).sort_values("k", ascending=False)
K = int(tab.k.sum()); Ntot = int(tab.N.sum()); pcell = tab.N.values/Ntot
# exact test: under one rate, distribute K hits multinomially ∝ N_i; is max-cell count >= observed max?
obs_max = int(tab.k.max()); sims = rng.multinomial(K, pcell, size=200000)
p_maxcell = (sims.max(1) >= obs_max).mean()
# chi-square-like discrepancy under multinomial (Monte-Carlo p, valid with tiny expected counts)
exp = K*pcell; obs_chi = (((tab.k.values-exp)**2)/exp).sum()
sim_chi = (((sims-exp)**2)/exp).sum(1); p_homog = (sim_chi >= obs_chi).mean()
print(f"  {K} DRG hits across {len(tab)} de-novo cells (N~1200 each).")
print(tab.assign(**{'rate%':(100*tab.k/tab.N).round(3)}).head(6).to_string())
print(f"  exact multinomial homogeneity (MC): p={p_homog:.4f}")
print(f"  post-hoc MAX-CELL null: P(any cell >= {obs_max} hits | one rate) = {p_maxcell:.4f}  "
      f"(Bonferroni-safe; this IS the corrected test for `max`)")
r, lo, hi = cp(int(tab.loc['max','k']), int(tab.loc['max','N']))
print(f"  max cell alone: {int(tab.loc['max','k'])}/{int(tab.loc['max','N'])} = {r:.3f}% [CP {lo:.3f}-{hi:.3f}]")
# max composition: crystal + which batch/jobs
mx = dn[dn.cond == "max"]; mxh = mx[mx.hit]
print(f"  `max` composition: hits by crystal = {mxh.cry.value_counts().to_dict()}")
print(f"    hit files: {list(mxh.file.str[:38])}")

hr("4 (CORRECTED). Groove-placement: conditioning vs null, with Clopper-Pearson CIs")
for g in ("denovo", "null"):
    s = DF[DF.g == g]; n = len(s); k = int(s.groove.sum()); r, lo, hi = cp(k, n)
    print(f"  {g:<8} groove-placed {k}/{n} = {r:.1f}% [CP {lo:.1f}-{hi:.1f}]")
gpn = DF[(DF.g == "null") & DF.register_defined]
print(f"  NOTE: register-defined NULL n={len(gpn)} -> {int(gpn.hit.sum())} hits. "
      f"0/{len(gpn)} cannot exclude 0.15%, so 'conditioning IMPROVES register GIVEN groove' is NOT testable.")
print("  SUPPORTED: conditioning places backbones in the groove (large n, clean vs null).")
print("  NOT SUPPORTED yet: conditioning improves register given groove placement.")

hr("5 (CORRECTED). Templating ladder: forward-only medians + FORWARD FRACTION per rung")
lad = DF[DF.cond.str.startswith("fix")].copy()
print(f"  {'rung':<6}{'n':>5}{'fwd%':>7}{'pooled-median':>15}{'fwd-only-median':>17}{'best':>7}")
for cond, s in lad.groupby("cond"):
    own = np.where(s.cry == "6AM5", s.toGIG, s.toDRG)
    fwd = (s.thread == "forward"); ownf = own[fwd.values]
    print(f"  {cond:<6}{len(s):>5}{100*fwd.mean():>6.1f}%{np.median(own):>13.2f}A{(np.median(ownf) if fwd.sum() else float('nan')):>15.2f}A{own.min():>6.2f}A")
print("  -> if fwd% rises with template fraction, part of the ladder is DIRECTION (threading), not register supply")
print("\nV2_DONE")
