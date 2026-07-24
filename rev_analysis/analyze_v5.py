import sys; sys.path.insert(0, "/global/scratch/users/sergiomar10/if-mhc/rev_analysis")
import numpy as np, pandas as pd
from scipy import stats
import pool as P, score_sd as S
DF = P.load(); dn = P.denovo(DF); C, GIG, DRG = P.coords()
def hr(t): print("\n"+"="*72+f"\n{t}\n"+"="*72)
def cp(k, n):
    lo = stats.beta.ppf(.025, k, n-k+1) if k > 0 else 0.0
    hi = stats.beta.ppf(.975, k+1, n-k) if k < n else 1.0
    return 100*k/n, 100*lo, 100*hi

# ---- PHYSICAL groove predicate (non-directional): peptide centroid near the groove center ----
gc = (GIG.mean(0) + DRG.mean(0)) / 2                      # groove center (both natives occupy it)
pep_cent = C.mean(1)                                     # (N,3) each design's peptide centroid
DF["cdist"] = np.linalg.norm(pep_cent - gc, axis=1)
# calibrate radius on the natives: GIG/DRG centroids are 0 from their own mean; use spread
nat_r = np.linalg.norm(np.array([GIG.mean(0), DRG.mean(0)]) - gc, axis=1).max()
PHYS_R = nat_r + 6.0
DF["phys_groove"] = DF.cdist < PHYS_R
dn = P.denovo(DF)

hr("A. PHYSICAL groove (centroid-based, NON-directional) vs RMSD-groove; threading cross-tab")
print(f"  groove center from natives; physical radius = {PHYS_R:.1f}A (native max {nat_r:.1f}A + 6)")
for g in ("denovo", "null"):
    s = DF[DF.g == g]
    print(f"  [{g}] RMSD-groove {s.groove.mean()*100:.1f}%   PHYSICAL-groove {s.phys_groove.mean()*100:.1f}%")
    pg = s[s.phys_groove]
    ct = {k: int(v) for k, v in pg.thread.value_counts().items()}
    print(f"     physically-in-groove threading: {ct}  (forward {100*(pg.thread=='forward').mean():.1f}%)")
    print(f"     REVERSE & physical-groove = {int(((s.thread=='reverse')&s.phys_groove).sum())}  "
          f"(vs REVERSE & RMSD-groove = {int(((s.thread=='reverse')&s.groove).sum())})")
print("  -> if reverse&PHYSICAL-groove >> reverse&RMSD-groove, the RMSD gate was hiding groove-resident reverse designs")

hr("B. `max` 52x gap: decompose the HIT at its ACTUAL threshold (<=1.48A) + conditional anchor")
mx = dn[dn.cond == "max"]; oth = dn[(dn.ncon >= 9) & (dn.cond != "max")]
def toDRGv(s): return s.toDRG.values
for lab, s in (("max", mx), ("other-rich", oth)):
    n = len(s)
    close148 = (s.toDRG <= 1.48); close158 = (s.toDRG <= 1.58); close20 = (s.toDRG <= 2.0)
    anch_close = ((s.fpos == 9) & close148).sum()
    p_anchor_given_close = (s.fpos[close148] == 9).mean() if close148.sum() else np.nan
    print(f"  {lab:<11} n={n:5d}  toDRG<=1.48: {close148.sum():3d} ({100*close148.mean():.3f}%)  "
          f"<=1.58: {close158.sum():3d}  <=2.0: {close20.sum():3d}  | P(anchor P9 | <=1.48)={p_anchor_given_close:.2f}")
# enrichment vs threshold
print("  enrichment (max/other) of P(toDRG<=t):")
for t in (2.5, 2.0, 1.58, 1.48):
    m = (mx.toDRG <= t).mean(); o = (oth.toDRG <= t).mean()
    print(f"    t={t}A: max {100*m:.3f}% vs other {100*o:.3f}%  ratio {m/o if o else float('inf'):.1f}")

hr("C. batch/crystal split of the 4 STRICT hits (single-batch artifact?)")
h = mx[(mx.fpos == 9) & (mx.toDRG <= 1.48)]
h = h.assign(batch=h.file.str.extract(r'_j(\d+)_')[0])
print(f"  4 strict max hits: crystals={h.cry.value_counts().to_dict()}  batches={h.batch.value_counts().to_dict()}")
for _, r in h.iterrows(): print(f"    {r.cry} toDRG={r.toDRG:.2f} toGIG={r.toGIG:.2f} batch=j{r.batch} file={r.file[:40]}")

hr("D. REGISTER-COORDINATE endpoint (register-specific, replaces absolute proximity)")
DF["regcoord"] = DF.toGIG - DF.toDRG                     # >0 = DRG-like
dn = P.denovo(DF); mx = dn[dn.cond == "max"]; oth = dn[(dn.ncon >= 9) & (dn.cond != "max")]
rd = dn[dn.register_defined]; mxr = rd[rd.cond == "max"]; othr = rd[(rd.ncon >= 9) & (rd.cond != "max")]
print("  register coord = toGIG - toDRG (positive => steered toward DRG). Tail mass on register_defined:")
for cut in (1.0, 1.5, 2.0):
    m = (mxr.regcoord > cut).mean(); o = (othr.regcoord > cut).mean()
    mk = int((mxr.regcoord > cut).sum()); ok = int((othr.regcoord > cut).sum())
    print(f"    regcoord>{cut}: max {100*m:.2f}% ({mk}/{len(mxr)}) vs other {100*o:.2f}% ({ok}/{len(othr)})  ratio {m/o if o else float('inf'):.2f}")
print("  -> use THIS (register-specific, hundreds of events) as ablation+maxrep primary, not absolute <2.5A.")
print("V5_DONE")
