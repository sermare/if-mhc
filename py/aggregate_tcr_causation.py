#!/usr/bin/env python3
"""Pool ALL steered-pull work values (base + parallel-worker logs) → final TCR-causation statistic.
+TCR = steer_TCR*.log, -TCR = steer_noTCR*.log. Welch t-test, Mann-Whitney, bootstrap 95% CI on Δ."""
import glob, re, numpy as np
def works(pat):
    v=[]
    for f in glob.glob(pat): v += [float(m) for m in re.findall(r"work=\s*([\d.]+) kJ/mol", open(f).read())]
    return np.array(v)
T=works("outputs/tcr_causation/steer_TCR*.log"); N=works("outputs/tcr_causation/steer_noTCR*.log")
print(f"+TCR n={len(T)}: mean {T.mean():.1f} ± {T.std(ddof=1):.1f} kJ/mol | median {np.median(T):.1f}")
print(f"-TCR n={len(N)}: mean {N.mean():.1f} ± {N.std(ddof=1):.1f} kJ/mol | median {np.median(N):.1f}")
diff=N.mean()-T.mean()
se=np.sqrt(T.var(ddof=1)/len(T)+N.var(ddof=1)/len(N)); t=diff/se
df=(T.var(ddof=1)/len(T)+N.var(ddof=1)/len(N))**2/((T.var(ddof=1)/len(T))**2/(len(T)-1)+(N.var(ddof=1)/len(N))**2/(len(N)-1))
try:
    from scipy import stats; p=2*stats.t.sf(abs(t),df); U,pmw=stats.mannwhitneyu(T,N,alternative="two-sided")
    mw=f" | Mann-Whitney p={pmw:.3f}"
except Exception:
    from math import erf; p=2*(1-0.5*(1+erf(abs(t)/np.sqrt(2)))); mw=""
rng=np.random.default_rng(0)
bs=np.array([rng.choice(N,len(N)).mean()-rng.choice(T,len(T)).mean() for _ in range(20000)])
print(f"\nΔ(−TCR − +TCR) = {diff:.1f} kJ/mol ({diff/4.184:.1f} kcal/mol) = {100*diff/N.mean():.0f}% less work with TCR")
print(f"Welch t={t:.2f} df={df:.0f} p={p:.3f}{mw}")
print(f"bootstrap 95% CI on Δ = [{np.percentile(bs,2.5):.1f}, {np.percentile(bs,97.5):.1f}] kJ/mol | P(Δ>0)={(bs>0).mean():.3f}")
print(f"VERDICT: {'SIGNIFICANT — TCR lowers the barrier' if p<0.05 else 'still not significant at p<0.05' } (direction: TCR eases shift, {(bs>0).mean()*100:.0f}% of bootstraps)")
