#!/usr/bin/env python3
"""Join MHCflurry + ESMCBA scores to the designs and test nb06's conclusions.

nb06 §7 reported that removing the TCR increases sampled diversity and, for two
of the four models, predicted affinity. That is the claim under test here.

ESMCBA's output scale is not documented in the checkpoint, so its direction is
established empirically against MHCflurry before anything is concluded from it.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

ROOT = "/global/scratch/users/sergiomar10/if-mhc"
# Phase selection: which (dataset, temperature) run to analyse. Defaults to the
# T=0.1 SKEMPI phase so existing invocations keep working; set SK_DATASET /
# SK_TEMP to point the same analysis at another phase.
DATASET = os.environ.get("SK_DATASET", "skempi")
TEMP = os.environ.get("SK_TEMP", "0.1")
TAG = "t" + TEMP.replace(".", "")
SUF = f"_{DATASET}_T{TEMP}"

OUT = f"{ROOT}/outputs/skempi_if"

T = pd.read_csv(f"{OUT}/peptides_to_score{SUF}.csv")
mf = pd.read_csv(f"{OUT}/mhcflurry_scores{SUF}.csv")
eb = pd.read_csv(f"{OUT}/esmcba_scores{SUF}.csv")
D = (T.merge(mf, on=["seq", "mhcflurry_allele"], how="left")
       .merge(eb, on=["seq", "esmcba_allele"], how="left"))
D["log_affinity"] = np.log10(D["affinity"])
D.to_csv(f"{OUT}/designs_scored{SUF}.csv", index=False)
print(f"joined {len(D):,} (complex,arm,model,peptide) rows; "
      f"mhcflurry missing {D.affinity.isna().sum()}, esmcba missing {D.esmcba_pred.isna().sum()}")


def hdr(t): print(f"\n{'='*80}\n{t}\n{'='*80}")


hdr("0. What direction is ESMCBA's score? (calibrate against MHCflurry)")
u = D.drop_duplicates("seq")
r, p = stats.spearmanr(u.esmcba_pred, u.log_affinity)
print(f"  ESMCBA pred vs log10(MHCflurry affinity nM): Spearman rho={r:+.3f} (p={p:.3g}, n={len(u)})")
DIRECTION = "higher = WEAKER binding" if r > 0 else "higher = STRONGER binding"
print(f"  -> ESMCBA {DIRECTION} (MHCflurry affinity is nM, so higher nM = weaker)")
nat = D[D.is_native].drop_duplicates("seq")
print(f"  native epitopes: median MHCflurry {nat.affinity.median():.1f} nM, "
      f"median ESMCBA {nat.esmcba_pred.median():.3f}")
print(f"  designs:         median MHCflurry {u.affinity.median():.1f} nM, "
      f"median ESMCBA {u.esmcba_pred.median():.3f}")

hdr("1. (nb06 §7) Does removing the TCR change predicted binding?")
print("  count-weighted per (complex, model, arm), then paired across arms by complex\n")
w = D.dropna(subset=["affinity"]).copy()


def wmean(g, col):
    return np.average(g[col], weights=g["count"])


cell = (w.groupby(["complex", "model", "arm"])
         .apply(lambda g: pd.Series({
             "log_aff": wmean(g, "log_affinity"),
             "pres": wmean(g, "presentation_score"),
             "esmcba": np.average(g.dropna(subset=["esmcba_pred"])["esmcba_pred"],
                                  weights=g.dropna(subset=["esmcba_pred"])["count"])
                        if g.esmcba_pred.notna().any() else np.nan,
             "uniq": g.seq.nunique()}), include_groups=False)
         .reset_index())

for metric, label, better in [("log_aff", "log10 affinity (nM)", "lower = stronger"),
                              ("pres", "presentation score", "higher = better"),
                              ("esmcba", "ESMCBA score", DIRECTION),
                              ("uniq", "unique peptides", "higher = more diverse")]:
    print(f"  {label}  [{better}]")
    for m in sorted(cell.model.unique()):
        s = cell[cell.model == m].pivot(index="complex", columns="arm", values=metric).dropna()
        if len(s) < 5:
            continue
        st, p = stats.wilcoxon(s["full"], s["notcr"])
        d = (s["notcr"] - s["full"]).mean()
        arrow = "notcr HIGHER" if d > 0 else "notcr LOWER"
        print(f"    {m:20s} full={s['full'].mean():8.3f} notcr={s['notcr'].mean():8.3f} "
              f"delta={d:+7.3f} ({arrow})  p={p:.3g} n={len(s)}")
    print()

hdr("2. Are designs better predicted binders than the native epitope?")
for m in sorted(D.model.unique()):
    for arm in ["full", "notcr"]:
        g = D[(D.model == m) & (D.arm == arm)].dropna(subset=["affinity"])
        if g.empty:
            continue
        natv = g[g.is_native]
        frac_strong = np.average(g.affinity < 500, weights=g["count"])
        print(f"  {m:20s} {arm:6s} designs <500nM: {100*frac_strong:5.1f}%   "
              f"median design {np.average(g.affinity, weights=g['count']):8.1f} nM")

hdr("3. Do the two predictors agree on which designs are good?")
for m in sorted(D.model.unique()):
    g = D[(D.model == m)].dropna(subset=["affinity", "esmcba_pred"]).drop_duplicates("seq")
    if len(g) < 20:
        continue
    r, p = stats.spearmanr(g.esmcba_pred, g.log_affinity)
    print(f"  {m:20s} rho={r:+.3f} (n={len(g)}, p={p:.2g})")
