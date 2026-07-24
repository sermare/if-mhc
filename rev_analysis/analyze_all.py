import sys, os, time, glob
sys.path.insert(0, "/global/scratch/users/sergiomar10/if-mhc/rev_analysis")
import numpy as np, pandas as pd
from scipy import stats
import score_sd as S
RA = "/global/scratch/users/sergiomar10/if-mhc/rev_analysis"
ROOT = "/global/scratch/users/sergiomar10/if-mhc"

# wait for cache
while not (os.path.exists(f"{RA}/design_cache.pkl") and os.path.exists(f"{RA}/design_coords.npz")
          and "CACHE_DONE" in open(f"{RA}/cache.log").read()):
    time.sleep(20)
DF = pd.read_pickle(f"{RA}/design_cache.pkl")
Z = np.load(f"{RA}/design_coords.npz"); C = Z["coords"]; GIG = Z["GIG"]; DRG = Z["DRG"]
CENT = S.FPOCKET_CENTROID
DF = DF.reset_index(drop=True)
dn = DF[DF.g == "denovo"]; dni = dn.index.values
def hr(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)

# ---- cond -> contact count + region composition (from the live spec) ----
spec = {}
for ln in open(f"{ROOT}/jobs/allcond150_spec.tsv"):
    p = ln.rstrip("\n").split("\t")
    if len(p) >= 6:
        hs = [h for h in p[5].split(",") if h.strip()]
        spec[p[1]] = dict(n=len(hs), mhc=sum(h[0] == "A" for h in hs), tcr=sum(h[0] in "DE" for h in hs))
NCON = {c: v["n"] for c, v in spec.items()}

hr("1. ANCHOR-MARGIN DISTRIBUTION (designs) -- is criterion 2 near a tie?")
d = np.linalg.norm(C - CENT, axis=2)                 # (N,10) Ca->F-pocket
margin = d[:, 8] - d[:, 9]                            # d_P9 - d_P10 ; >0 => P10 closer (GIG-like)
DF["margin"] = margin; DF["fp_argmin"] = d.argmin(1) + 1
for cry, sub in DF.groupby("cry"):
    near_tie = (sub.margin.abs() < 0.5).mean() * 100
    print(f"  {cry}: |margin|<0.5A in {near_tie:4.1f}% of designs   median|margin|={sub.margin.abs().median():.2f}A")
print(f"  crystal refs: GIG margin +3.21A, DRG margin -3.54A (>> thermal ~1A) -> not degenerate at natives")
print(f"  design argmin distribution: {DF.fp_argmin.value_counts().sort_index().to_dict()}")

hr("2. CONTINUOUS REGISTER COORDINATE (projection on DRG-GIG diff @ P7-P10)")
POS = np.array([6, 7, 8, 9])                          # P7-P10 (0-indexed)
diff = (DRG - GIG)[POS].reshape(-1); diff /= np.linalg.norm(diff)
mid = ((GIG + DRG) / 2)[POS].reshape(-1)
proj = (C[:, POS, :].reshape(len(C), -1) - mid) @ diff   # >0 => DRG-like, <0 => GIG-like
DF["reg_coord"] = proj
for g, sub in DF.groupby("g"):
    print(f"  {g:<10} n={len(sub):5d}  reg_coord mean={sub.reg_coord.mean():+.2f} sd={sub.reg_coord.std():.2f}")
# rich vs sparse denovo, as DISTRIBUTIONS (the underpowered Fisher test -> Mann-Whitney on a scalar)
dn2 = DF[DF.g == "denovo"].copy(); dn2["ncon"] = dn2.cond.map(NCON)
rich = dn2[dn2.ncon >= 9].reg_coord.dropna(); spar = dn2[dn2.ncon.between(3, 5)].reg_coord.dropna()
if len(rich) and len(spar):
    u, p = stats.mannwhitneyu(rich, spar)
    print(f"  rich(>=9,n={len(rich)}) vs sparse(3-5,n={len(spar)}) reg_coord: "
          f"MWU p={p:.3g}  medians {rich.median():+.2f} vs {spar.median():+.2f}  (n now hundreds, not 5 events)")

hr("3. FUNNEL -- what fraction is even extended & groove-placed?")
e2e_nat = np.linalg.norm(GIG[0] - GIG[9]), np.linalg.norm(DRG[0] - DRG[9])
lo, hiE = min(e2e_nat) - 6, max(e2e_nat) + 6
DF["extended"] = DF.e2e.between(lo, hiE)
DF["groove"] = DF[["toGIG", "toDRG"]].min(1) < 8.0    # within 8A of either native register
for g, sub in DF.groupby("g"):
    n = len(sub); ext = sub.extended.sum(); grv = (sub.extended & sub.groove).sum()
    fwd = (sub.extended & sub.groove & (sub.thread == "forward")).sum()
    print(f"  {g:<10} N={n:5d} -> extended {ext:5d} ({100*ext/n:4.1f}%) -> groove {grv:5d} "
          f"({100*grv/n:4.1f}%) -> +forward {fwd:5d}")
print(f"  native end-to-end: GIG {e2e_nat[0]:.1f}A, DRG {e2e_nat[1]:.1f}A ; extended window [{lo:.0f},{hiE:.0f}]A")
# conditional recovery rate among groove-placed forward denovo
gp = DF[(DF.g == "denovo") & DF.extended & DF.groove & (DF.thread == "forward")]
DRG_HI = 1.58
rec = ((gp.cry == "6AMU") & (gp.fpos == 9) & (gp.toDRG <= DRG_HI)).sum()
cro = ((gp.cry == "6AM5") & (gp.fpos == 9) & (gp.toDRG <= DRG_HI)).sum()
print(f"  CONDITIONAL (groove-placed fwd denovo, n={len(gp)}): recovery {rec}, crossing {cro} "
      f"-> rate {100*(rec+cro)/max(len(gp),1):.2f}% (vs 0.04% on full pool)")

hr("4. FORWARD-SUBSET TEMPLATING LADDER (reviewer: pooled median may be a bimodal artifact)")
lad = DF[DF.cond.str.startswith("fix")].copy()
if len(lad):
    for cond, sub in lad.groupby("cond"):
        own = np.where(sub.cry == "6AM5", sub.toGIG, sub.toDRG)
        fwd = sub.thread == "forward"
        ownf = np.where(sub[fwd].cry == "6AM5", sub[fwd].toGIG, sub[fwd].toDRG)
        print(f"  {cond:<7} n={len(sub):4d}  median-to-own POOLED={np.median(own):5.2f}A  "
              f"FWD-only(n={fwd.sum():4d})={np.median(ownf) if fwd.sum() else float('nan'):5.2f}A  best={own.min():.2f}A")
else:
    print("  no fix* templating cells in this corpus slice")

hr("5. BOOTSTRAP best-of-26 (is the 0.49A cross-cell spread just resampling noise?)")
pool = np.where(dn[dn.thread == "forward"].cry == "6AM5",
                dn[dn.thread == "forward"].toGIG, dn[dn.thread == "forward"].toDRG)
pool = pool[np.isfinite(pool)]
if len(pool) > 260:
    rng = np.random.default_rng(0)
    best26 = [pool[rng.integers(0, len(pool), 26)].min() for _ in range(20000)]
    b = np.array(best26)
    print(f"  pooled fwd denovo n={len(pool)}; best-of-26 distribution: "
          f"mean={b.mean():.2f} sd={b.sd() if hasattr(b,'sd') else b.std():.2f}  "
          f"5-95%: {np.percentile(b,5):.2f}-{np.percentile(b,95):.2f}A (spread {np.percentile(b,95)-np.percentile(b,5):.2f}A)")
    print(f"  -> observed cross-cell best-of-26 spread ~0.49A sits INSIDE resampling spread "
          f"{'(YES, tautological)' if (np.percentile(b,95)-np.percentile(b,5))>=0.49 else '(NO, real differences)'}")

hr("6. REGISTER SLIPPAGE (slide +-3 positions; right shape wrong phase?)")
def best_offset(pa, ref):
    best = (1e9, 0)
    for k in range(-3, 4):
        idx = np.arange(10) + k; ok = (idx >= 0) & (idx < 10)
        if ok.sum() < 6: continue
        r = np.sqrt(((pa[np.arange(10)[ok]] - ref[idx[ok]]) ** 2).sum() / ok.sum())
        if r < best[0]: best = (r, k)
    return best
noff = 0; improved = []
samp = dn.sample(min(3000, len(dn)), random_state=1)
for i in samp.index:
    ref = GIG if DF.cry[i] == "6AM5" else DRG
    r0 = np.sqrt(((C[i] - ref) ** 2).sum() / 10)
    rb, k = best_offset(C[i], ref)
    if k != 0 and rb < r0 - 0.5: noff += 1; improved.append(r0 - rb)
print(f"  of {len(samp)} denovo: {noff} ({100*noff/len(samp):.1f}%) fit >=0.5A better at a NON-zero offset")
print(f"  -> 'right shape, wrong phase' population; median improvement {np.median(improved) if improved else 0:.2f}A")

hr("7. RATE HOMOGENEITY across denovo cells (replaces tautological draws-vs-hits r)")
dn3 = DF[DF.g == "denovo"].copy()
dn3["hit"] = (((dn3.cry == "6AMU") & (dn3.fpos == 9) & (dn3.toDRG <= DRG_HI)) |
              ((dn3.cry == "6AM5") & (dn3.fpos == 9) & (dn3.toDRG <= DRG_HI)))
tab = dn3.groupby("cond").agg(N=("hit", "size"), k=("hit", "sum"))
tab["rate%"] = 100 * tab.k / tab.N
K, Ntot = tab.k.sum(), tab.N.sum(); p0 = K / Ntot
chi = (((tab.k - tab.N * p0) ** 2) / (tab.N * p0 + 1e-9)).sum()
dof = len(tab) - 1
print(tab.sort_values("rate%", ascending=False).head(12).to_string())
print(f"  pooled rate {100*p0:.3f}% ; Poisson homogeneity chi2={chi:.1f} dof={dof} "
      f"p={1-stats.chi2.cdf(chi,dof):.3f} -> {'cells DIFFER' if 1-stats.chi2.cdf(chi,dof)<0.05 else 'consistent with ONE rate (only budget varies)'}")

DF.to_pickle(f"{RA}/design_cache_aug.pkl")
print("\nANALYZE_DONE")
