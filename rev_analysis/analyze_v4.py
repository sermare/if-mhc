import sys; sys.path.insert(0, "/global/scratch/users/sergiomar10/if-mhc/rev_analysis")
import numpy as np, pandas as pd
from scipy import stats
import pool as P
DF = P.load(); dn = P.denovo(DF); C, GIG, DRG = P.coords(); rng = np.random.default_rng(0)
def hr(t): print("\n"+"="*72+f"\n{t}\n"+"="*72)
def cp(k, n):
    lo = stats.beta.ppf(.025, k, n-k+1) if k > 0 else 0.0
    hi = stats.beta.ppf(.975, k+1, n-k) if k < n else 1.0
    return 100*k/n, 100*lo, 100*hi
def toown(s): return np.where(s.cry == "6AM5", s.toGIG, s.toDRG)
CENT = P.__dict__.get("CENT")
import score_sd as S; CENT = S.FPOCKET_CENTROID

hr("1. THIRD FINDING under the PRIMARY (STRICT <=1.48) hit definition")
dn = dn.copy()
dn["hit_s"] = (dn.fpos == 9) & (dn.toDRG <= P.DRG_POINT) & ((dn.cry == "6AMU") | (dn.cry == "6AM5"))
tab = dn.groupby("cond").agg(N=("hit_s", "size"), k=("hit_s", "sum")).sort_values("k", ascending=False)
K = int(tab.k.sum()); Ntot = int(tab.N.sum()); pcell = tab.N.values/Ntot
obs_max = int(tab.k.max()); sims = rng.multinomial(K, pcell, size=200000)
p_maxcell = (sims.max(1) >= obs_max).mean()
exp = K*pcell; obs_chi = (((tab.k.values-exp)**2)/exp).sum()
p_homog = ((((sims-exp)**2)/exp).sum(1) >= obs_chi).mean()
print(f"  STRICT hits total = {K} (was 11 CI-inclusive). per-cell:")
print(tab[tab.k > 0].assign(**{'rate%': (100*tab.k/tab.N).round(3)}).to_string())
mr, mlo, mhi = cp(int(tab.iloc[0].k), int(tab.iloc[0].N))
print(f"  max STRICT: {int(tab.iloc[0].k)}/{int(tab.iloc[0].N)} = {mr:.3f}% [CP {mlo:.3f}-{mhi:.3f}]")
print(f"  multinomial homogeneity p={p_homog:.4f}; post-hoc max-cell null P(any>= {obs_max})={p_maxcell:.4f}")
oth_k = K-int(tab.iloc[0].k); oth_n = Ntot-int(tab.iloc[0].N)
print(f"  pooled other-cell STRICT rate = {oth_k}/{oth_n} = {100*oth_k/oth_n:.4f}%  <- maxrep decision threshold")

hr("2. THREADING x GROOVE cross-tab (denovo + null) -- is the groove gate directional?")
for g in ("denovo", "null"):
    s = DF[DF.g == g]
    ct = pd.crosstab(s.thread, s.groove)
    print(f"  [{g}]  (rows=threading, cols=groove-placed)"); print(ct.to_string())
    gp = s[s.groove]
    print(f"     groove-placed: forward {100*(gp.thread=='forward').mean():.1f}%  reverse {100*(gp.thread=='reverse').mean():.1f}%")
    rev_grv = ((s.thread == "reverse") & s.groove).sum()
    print(f"     REVERSE & groove (true mirror-image population) = {rev_grv}  "
          f"({100*rev_grv/len(s):.1f}% of {g})")
print("  -> if NULL groove-placed is also ~forward, the gate is directional by construction (not placement).")

hr("3. `max` DECOMPOSITION: where does the 7x advantage live? (forward->groove->prox->anchor)")
rd = dn
mx = rd[rd.cond == "max"]; oth = rd[(rd.ncon >= 9) & (rd.cond != "max")]
def stage(sub, cond_prev):
    return sub[cond_prev] if cond_prev is not None else sub
def rate(sub, mask):
    return (100*mask.mean(), mask.sum(), len(sub))
print(f"  {'stage':<34}{'max':>18}{'other-rich':>18}{'ratio':>7}")
# stage 1: forward fraction
for name, m_sub, o_sub, mfn, ofn in [
    ("1 forward fraction", mx, oth, mx.forward, oth.forward),
]:
    mr = mfn.mean(); orr = ofn.mean()
    print(f"  {name:<34}{100*mr:>16.1f}%{100*orr:>16.1f}%{mr/orr:>7.2f}")
# stage 2: groove | forward
mf = mx[mx.forward]; of = oth[oth.forward]
mr, orr = mf.groove.mean(), of.groove.mean()
print(f"  {'2 groove | forward':<34}{100*mr:>16.1f}%{100*orr:>16.1f}%{mr/orr:>7.2f}")
# stage 3: proximity <2.0 to own | groove&forward
mg = mf[mf.groove].assign(t=lambda d: toown(d)); og = of[of.groove].assign(t=lambda d: toown(d))
mr, orr = (mg.t < 2.0).mean(), (og.t < 2.0).mean()
print(f"  {'3 <2.0A prox | groove&fwd':<34}{100*mr:>16.2f}%{100*orr:>16.2f}%{(mr/orr if orr else np.nan):>7.2f}")
# stage 4: strict hit | proximate(<2.5)
mp = mg[mg.t < 2.5]; op = og[og.t < 2.5]
mr, orr = mp.hit_s.mean() if len(mp) else 0, op.hit_s.mean() if len(op) else 0
print(f"  {'4 strict-hit | prox<2.5':<34}{100*mr:>16.1f}%{100*orr:>16.1f}%{(mr/orr if orr else np.nan):>7.2f}")
print("  -> stage with the biggest ratio is what `max` actually carries (forward=orientation, not register).")
# is max a threading outlier?
print(f"  corpus forward fraction: {100*rd.forward.mean():.1f}% ; max: {100*mx.forward.mean():.1f}% ; "
      f"other-rich: {100*oth.forward.mean():.1f}%")

hr("4. IN-REGISTER FLOOR from fix8 (n~1200, same pipeline) instead of 13 MD frames")
def offset_frac(sub):
    idx = sub.index.values; noff = 0
    for i in idx:
        ref = GIG if DF.cry[i] == "6AM5" else DRG
        r0 = np.sqrt(((C[i]-ref)**2).sum()/10); best = (r0, 0)
        for k in range(-3, 4):
            if k == 0: continue
            j = np.arange(10)+k; ok = (j >= 0) & (j < 10)
            if ok.sum() < 6: continue
            r = np.sqrt(((C[i][np.arange(10)[ok]]-ref[j[ok]])**2).sum()/ok.sum())
            if r < best[0]: best = (r, k)
        if best[1] != 0 and best[0] < r0-0.5: noff += 1
    return noff, len(idx)
for cond in ("fix8", "fix6"):
    s = DF[DF.cond == cond]
    correct = np.where(s.cry == "6AM5", 10, 9)
    flip = (s.fpos.values != correct).mean()
    no, n = offset_frac(s)
    print(f"  {cond}: n={len(s)}  anchor-flip {100*flip:.1f}%   offset-slippage {100*no/n:.1f}% (in-register floor)")
print("  vs de-novo REGISTER_DEFINED slippage 55.7%. If fix8 ~0%, (7) is unassailable with n in thousands.")

hr("5. contact-only FLOOR as a MEDIAN on groove&forward (not best-of-N) -- 5 comparison")
cof = dn[dn.register_defined].assign(t=lambda d: toown(d))
print(f"  contact-only (register-defined denovo) median to-own = {np.median(cof.t):.2f}A  (fix2 median = 2.43A)")
print(f"  -> if this < 2.43A, 'fix2 buys threading only' becomes 'fix2 is WORSE than contact-only'")
print("V4_DONE")
