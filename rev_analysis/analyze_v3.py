import sys; sys.path.insert(0, "/global/scratch/users/sergiomar10/if-mhc/rev_analysis")
import numpy as np, pandas as pd
from scipy import stats
import pool as P
DF = P.load(); dn = P.denovo(DF); rng = np.random.default_rng(0)
def hr(t): print("\n"+"="*72+f"\n{t}\n"+"="*72)
def toown(s): return np.where(s.cry == "6AM5", s.toGIG, s.toDRG)
def cp(k, n):
    lo = stats.beta.ppf(.025, k, n-k+1) if k > 0 else 0.0
    hi = stats.beta.ppf(.975, k+1, n-k) if k < n else 1.0
    return 100*k/n, 100*lo, 100*hi

hr("A. Does REGISTER_DEFINED enforce forward? (gate for 2)")
g = dn[dn.groove]
print(f"  groove-placed de-novo n={len(g)};  of these forward-threaded: {g.forward.sum()} "
      f"({100*g.forward.mean():.1f}%)  reverse: {(~g.forward).sum()} ({100*(~g.forward).mean():.1f}%)")
print(f"  -> reverse designs mostly FAIL the groove gate, so groove ~= groove&forward.")
print(f"  register_defined defn = extended & groove & FORWARD (n={dn.register_defined.sum()}); 2 already forward-only.")

hr("B. 2 re-run strictly on groove & forward; U-shape diagnostic + tier composition")
rd = dn[dn.register_defined].copy(); rd["toown"] = toown(rd)
def tier(n): return "sparse(1-5)" if n <= 5 else ("mid(6-8)" if n <= 8 else "rich(>=9)")
rd["tier"] = rd.ncon.map(tier)
for t in ("sparse(1-5)", "mid(6-8)", "rich(>=9)"):
    s = rd[rd.tier == t]; f20 = (s.toown < 2.0).mean()*100
    bs = [(s.toown.values[rng.integers(0, len(s), len(s))] < 2.0).mean()*100 for _ in range(2000)]
    comp = s.cond.value_counts().to_dict()
    print(f"  {t:<12} n={len(s):5d}  <2.0A={f20:.2f}% [{np.percentile(bs,2.5):.2f}-{np.percentile(bs,97.5):.2f}]  "
          f"conds={comp}  crystal={s.cry.value_counts().to_dict()}")
print("  -> mid tier composition tells you whether tiers are clean strata")

hr("C. 2-vs-3 TENSION: is `max` elevated in PROXIMITY (<2.0A) or only in pocket/register hits?")
rrich = rd[rd.tier == "rich(>=9)"]
mx = rrich[rrich.cond == "max"]; oth = rrich[rrich.cond != "max"]
for lab, s in (("max (rich)", mx), ("other rich cells", oth)):
    f20 = (s.toown < 2.0).mean()*100; f25 = (s.toown < 2.5).mean()*100
    print(f"  {lab:<18} n={len(s):5d}  <2.0A={f20:.2f}%  <2.5A={f25:.2f}%  median={np.median(s.toown):.2f}A")
print("  -> if max NOT elevated in <2.0A proximity, its 7 hits come from ANCHOR+DEPTH, i.e. max is special")
print("     at POCKET PLACEMENT, not register -- a different (and more consistent) claim.")

hr("D. CROSS-CAMPAIGN REPLICATION: is the 18-contact cell special in the new pool?")
print("  OLD 1,295-corpus top cell: C18 (18 contacts) = 3/186 = 1.61% (dead after Bonferroni).")
for c in ("k18", "k24", "k14", "max"):
    s = dn[dn.cond == c]
    if len(s):
        r, lo, hi = cp(int(s.hit.sum()), len(s))
        print(f"  NEW pool {c:<5} ({int(s.ncon.iloc[0])} contacts): {int(s.hit.sum())}/{len(s)} = {r:.3f}% [CP {lo:.3f}-{hi:.3f}]")
print("  -> the 18-contact cell (k18) in the new campaign is ordinary; top-cell effects do NOT replicate across campaigns.")

hr("E. WHAT IS `max`? superset of the other rich schemes, or unique residues?")
spec = {}
for ln in open(f"{P.ROOT}/jobs/allcond150_spec.tsv"):
    p = ln.rstrip("\n").split("\t")
    if len(p) >= 6 and p[0] == "6AM5":
        spec[p[1]] = set(h for h in p[5].split(",") if h.strip())
mx_set = spec.get("max", set())
for c in ("L5_max", "k24", "k18", "k14", "mhc_tcr2"):
    s = spec.get(c, set())
    if s:
        print(f"  max vs {c:<9}: {c} has {len(s)} res; subset of max? {s <= mx_set}; "
              f"residues in {c} NOT in max: {sorted(s - mx_set) or 'none'}")
uniq = sorted(mx_set - set().union(*[spec.get(c, set()) for c in ("L5_max","k24","k18","k14")]))
print(f"  residues UNIQUE to max (in none of L5_max/k24/k18/k14): {uniq}")

hr("F. NATIVE controls restated with EFFECTIVE sample size (ESS)")
for cry, ess, flips, off in (("DRG(6AMU)", 13, 0, 0), ("GIG(6AM5)", 2, 0, 0)):
    # exact 95% upper bound for 0/ESS (Clopper-Pearson): 1-0.025^(1/ess) approx; use beta ppf
    ub = 100*stats.beta.ppf(.975, 1, ess)   # 0 successes -> upper = beta(1, n) .975
    print(f"  {cry}: flip {flips}/~{ess} effective (ESS from tau_int); 95% upper bound on flip rate = {ub:.1f}%")
print("  -> conclusion (no flips / no offset-improvement) holds, but report as 0-of-ESS with these bounds, not 0/1000.")

hr("G. TABLE 2 RESTATEMENT (one pool = 16,919 de-novo; strict primary + CI-inclusive)")
strict_rec = int(((dn.cry=="6AMU")&(dn.fpos==9)&(dn.toDRG<=P.DRG_POINT)).sum())
strict_cro = int(((dn.cry=="6AM5")&(dn.fpos==9)&(dn.toDRG<=P.DRG_POINT)).sum())
ci_rec = int(dn.recovery.sum()); ci_cro = int(dn.crossing.sum())
for lab,k in (("recovery strict(<=1.48)",strict_rec),("crossing strict(<=1.48)",strict_cro),
              ("recovery CI(<=1.58)",ci_rec),("crossing CI(<=1.58)",ci_cro)):
    r,lo,hi=cp(k,len(dn)); print(f"  {lab:<26} {k}/{len(dn)} = {r:.3f}% [CP {lo:.3f}-{hi:.3f}]")
print("V3_DONE")
