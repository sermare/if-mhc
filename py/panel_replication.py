#!/usr/bin/env python3
"""Test the notebooks/panel conclusions on the new SKEMPI design set.

The panel notebooks were built on 20 pMHC-TCR crystals from
inputs/pmhc_tcr_dataset. This sweep used the SAME protocol -- 4 models,
full (MHC+TCR) vs notcr/mhconly, T=0.1, epitope-only design, 10k per cell -- on
28 SKEMPI complexes chosen to NOT overlap that panel. So this is an out-of-sample
replication, not a re-analysis.

Their statistical design is reproduced deliberately: average the 4 models first
so the independent unit is the crystal, compare the two pre-registered anchors
(P2 and P-omega) against a FIXED interior reference group (P3..P(omega-1),
excluding P1), paired Wilcoxon, Bonferroni-corrected by 2.
"""
import glob, os, sys
import numpy as np
import pandas as pd
from scipy import stats

ROOT = "/global/scratch/users/sergiomar10/if-mhc"
DES = f"{ROOT}/designs/skempi/t01"
AA = "ACDEFGHIKLMNPQRSTVWY"
MODELS = ["esmif", "proteinmpnn", "proteinmpnn_nomhc", "ligandmpnn"]

CHEM = {}
for a in "AGILMPV": CHEM[a] = "nonpolar_aliphatic"
for a in "FWY":     CHEM[a] = "aromatic"
for a in "STCNQ":   CHEM[a] = "polar_uncharged"
for a in "KRH":     CHEM[a] = "basic"
for a in "DE":      CHEM[a] = "acidic"
KD = {"A":1.8,"R":-4.5,"N":-3.5,"D":-3.5,"C":2.5,"Q":-3.5,"E":-3.5,"G":-0.4,
      "H":-3.2,"I":4.5,"L":3.8,"K":-3.9,"M":1.9,"F":2.8,"P":-1.6,"S":-0.8,
      "T":-0.7,"W":-0.9,"Y":-1.3,"V":4.2}

def hdr(t): print(f"\n{'='*82}\n{t}\n{'='*82}")

# ---------------------------------------------------------------- load
frames = []
for m in MODELS:
    f = f"{DES}/{m}.csv.gz"
    if os.path.exists(f):
        d = pd.read_csv(f, usecols=["complex", "arm", "seq", "native", "recovery"])
        d["model"] = m
        frames.append(d)
df = pd.concat(frames, ignore_index=True)
df["seq"] = df.seq.astype(str); df["native"] = df.native.astype(str)
print(f"loaded {len(df):,} designs, {df['complex'].nunique()} complexes, "
      f"{df.model.nunique()} models, arms={sorted(df.arm.unique())}")

# ------------------------------------------- per (complex, model, arm, position)
rec_rows, chem_rows = [], []
for (cid, m, arm), g in df.groupby(["complex", "model", "arm"]):
    nat = g.native.iloc[0]; L = len(nat)
    S = np.array([list(s[:L].ljust(L, "X")) for s in g.seq])
    for i in range(L):
        col = S[:, i]
        grp = ("P2" if i == 1 else "Pomega" if i == L - 1
               else "P1" if i == 0 else "interior")
        recov = float((col == nat[i]).mean())
        chem_ok = float(np.mean([CHEM.get(c) == CHEM.get(nat[i]) for c in col]))
        kd = np.array([KD.get(c, np.nan) for c in col], dtype=float)
        vals, cnts = np.unique(col, return_counts=True)
        p = cnts / cnts.sum()
        ent20 = float(-(p * np.log2(p)).sum() / np.log2(20))
        cl = pd.Series([CHEM.get(c, "X") for c in col]).value_counts(normalize=True).values
        ent5 = float(-(cl * np.log2(cl)).sum() / np.log2(5))
        rec_rows.append(dict(complex=cid, model=m, arm=arm, pos=i + 1, L=L, group=grp,
                             native_aa=nat[i], recovery=recov, chem_match=chem_ok,
                             kd_var=float(np.nanvar(kd)), ent20=ent20, ent5=ent5))
R = pd.DataFrame(rec_rows)
R.to_csv(f"{ROOT}/outputs/skempi_if/panel_replication_positions.csv", index=False)

# models averaged first -> crystal is the independent unit (their corrected design)
C = R.groupby(["complex", "arm", "group"]).agg(
    recovery=("recovery", "mean"), chem_match=("chem_match", "mean"),
    kd_var=("kd_var", "mean"), ent20=("ent20", "mean"), ent5=("ent5", "mean")).reset_index()


def wilcoxon_vs_interior(arm, metric, greater=True):
    out = []
    piv = C[C.arm == arm].pivot(index="complex", columns="group", values=metric)
    for anchor in ["P2", "Pomega"]:
        sub = piv[[anchor, "interior"]].dropna()
        if len(sub) < 5:
            continue
        alt = "greater" if greater else "less"
        st, p = stats.wilcoxon(sub[anchor], sub["interior"], alternative=alt)
        out.append((anchor, sub[anchor].mean(), sub["interior"].mean(),
                    p, min(p * 2, 1.0), len(sub)))
    return out


hdr("CLAIM 1 (nb03 §2a) -- P2 and P-omega are better recovered than interior positions")
print("  paired Wilcoxon, models averaged first, fixed interior reference, Bonferroni x2")
for arm in ["full", "notcr"]:
    print(f"\n  arm = {arm}")
    for anchor, a_mean, i_mean, p, padj, n in wilcoxon_vs_interior(arm, "recovery"):
        verdict = "HOLDS" if padj < 0.05 else "not significant"
        print(f"    {anchor:7s} recovery {a_mean:.3f} vs interior {i_mean:.3f}  "
              f"n={n}  p_adj={padj:.3g}  -> {verdict}".replace("e-0", "e-"))

hdr("CLAIM 2 (nb07) -- P-omega is permissive on identity but constrained on chemistry")
for arm in ["full", "notcr"]:
    piv = C[C.arm == arm].pivot(index="complex", columns="group", values="recovery")
    pc = C[C.arm == arm].pivot(index="complex", columns="group", values="chem_match")
    e20 = C[C.arm == arm].pivot(index="complex", columns="group", values="ent20")
    e5 = C[C.arm == arm].pivot(index="complex", columns="group", values="ent5")
    print(f"\n  arm = {arm}")
    print(f"    {'group':10s} {'identity_rec':>13s} {'chem_match':>11s} {'gain':>7s} "
          f"{'ent20':>7s} {'ent5':>7s}")
    for g in ["P2", "Pomega", "interior"]:
        if g in piv:
            print(f"    {g:10s} {piv[g].mean():13.3f} {pc[g].mean():11.3f} "
                  f"{pc[g].mean()-piv[g].mean():7.3f} {e20[g].mean():7.3f} {e5[g].mean():7.3f}")
    for anchor, a, i, p, padj, n in wilcoxon_vs_interior(arm, "chem_match"):
        print(f"    chem_match {anchor:7s} {a:.3f} vs interior {i:.3f}  p_adj={padj:.3g}"
              f"  -> {'HOLDS' if padj < 0.05 else 'not significant'}")

hdr("CLAIM 3 (nb03 §6) -- TCR context improves recovery, per crystal")
piv = C.pivot_table(index="complex", columns="arm", values="recovery")
piv["delta"] = piv["full"] - piv["notcr"]
st, p = stats.wilcoxon(piv["full"], piv["notcr"], alternative="greater")
print(f"  mean delta (full - notcr) = {piv['delta'].mean():+.3f}   "
      f"crystals improved: {(piv['delta'] > 0).sum()}/{len(piv)}   Wilcoxon p={p:.3g}")
print(f"  -> {'HOLDS' if p < 0.05 else 'FAILS'}")
print(f"  most helped: {piv['delta'].idxmax()} {piv['delta'].max():+.3f} | "
      f"least: {piv['delta'].idxmin()} {piv['delta'].min():+.3f}")

hdr("CLAIM 4 (nb03 §6, per model) -- benefit holds for every model, not one")
for m in MODELS:
    sub = R[(R.model == m) & (R.group != "P1")].groupby(["complex", "arm"]).recovery.mean().unstack()
    if "full" in sub and "notcr" in sub:
        s = sub.dropna()
        st, p = stats.wilcoxon(s["full"], s["notcr"], alternative="greater")
        print(f"  {m:20s} delta={(s['full']-s['notcr']).mean():+.3f}  n={len(s)}  "
              f"p={p:.3g}  -> {'HOLDS' if p < 0.05 else 'FAILS'}")

hdr("CLAIM 5 (nb03 §7) -- peptide length confounds recovery (paper: pooled Pearson r=0.36)")
cell = R[R.group != "P1"].groupby(["complex", "model", "arm"]).agg(
    recovery=("recovery", "mean"), L=("L", "first")).reset_index()
for arm in ["full", "notcr"]:
    s = cell[cell.arm == arm]
    r, p = stats.pearsonr(s.L, s.recovery)
    print(f"  arm={arm:6s} pooled Pearson r={r:+.3f} (p={p:.3g}, n={len(s)})")
print("  paper reported r=+0.36 pooled; sign/magnitude compared above")

hdr("CLAIM 6 (nb03 §3) -- models succeed and fail at the SAME sites")
site = R[R.group != "P1"].pivot_table(index=["complex", "arm", "pos"],
                                      columns="model", values="recovery")
site = site.dropna()
print(f"  pairwise Spearman on per-site recovery (n={len(site)} sites):")
for i, a in enumerate(MODELS):
    for b in MODELS[i+1:]:
        if a in site and b in site:
            rho, p = stats.spearmanr(site[a], site[b])
            print(f"    {a:20s} vs {b:20s} rho={rho:+.3f} (p={p:.2g})")

hdr("CLAIM 7 (nb03 §4) -- models agree on WHICH crystals are diverse")
uniq = df.groupby(["complex", "model", "arm"]).seq.nunique().reset_index(name="n_uniq")
for arm in ["full", "notcr"]:
    piv = uniq[uniq.arm == arm].pivot(index="complex", columns="model", values="n_uniq").dropna()
    rhos = []
    for i, a in enumerate(MODELS):
        for b in MODELS[i+1:]:
            if a in piv and b in piv:
                rhos.append(stats.spearmanr(piv[a], piv[b])[0])
    print(f"  arm={arm:6s} mean pairwise Spearman across models = {np.mean(rhos):+.3f} "
          f"(n={len(piv)} crystals, {len(rhos)} pairs)")

print("\n\nwrote per-position table -> outputs/skempi_if/panel_replication_positions.csv")
