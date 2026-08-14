#!/usr/bin/env python3
"""Quality report on the sampled epitopes, over whatever phases have data.

Reports, per model and arm:
  recovery      mean per-position identity to the native epitope
  unique        distinct sequences per 10k draws (how cold the sampling is)
  top_frac      mass on the single most-sampled sequence
  P1 Met        the chain-start methionine artifact
  Cterm match   fraction matching the native C-terminal anchor

and per complex, the TCR-present vs TCR-removed divergence: mean per-position
Jensen-Shannon distance between the two arms' amino-acid distributions. That is
the number the whole two-arm design exists to produce -- 0 means deleting the
TCR changed nothing, 1 means the two arms share no preferred residue.
"""
import glob, os, sys
import numpy as np
import pandas as pd

ROOT = "/global/scratch/users/sergiomar10/if-mhc"
OUT = f"{ROOT}/outputs/skempi_if"
AA = "ACDEFGHIKLMNPQRSTVWY"
MODELS = ["esmif", "proteinmpnn", "proteinmpnn_nomhc", "ligandmpnn"]
PHASES = ["skempi/T0.1", "skempi/T0.3", "pmhc25/T0.3", "focus6am/T0.1", "focus6am/T0.3"]


def pwm(seqs, L):
    m = np.zeros((L, 20))
    for s in seqs:
        for i, c in enumerate(s[:L]):
            if c in AA:
                m[i, AA.index(c)] += 1
    return m / np.maximum(m.sum(axis=1, keepdims=True), 1)


def js_dist(p, q):
    """Jensen-Shannon distance per position, averaged."""
    m = 0.5 * (p + q)
    def kl(a, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(a > 0, a * np.log2(a / np.where(b > 0, b, 1e-12)), 0.0)
        return t.sum(axis=1)
    return float(np.sqrt(np.maximum(0.5 * kl(p, m) + 0.5 * kl(q, m), 0)).mean())


def load(phase):
    rows = []
    for m in MODELS:
        for f in glob.glob(f"{OUT}/{phase}/{m}/parts/*.csv"):
            try:
                d = pd.read_csv(f, usecols=["complex", "arm", "seq", "native", "recovery"])
            except Exception:
                continue
            if len(d):
                d["model"] = m
                rows.append(d)
    return pd.concat(rows, ignore_index=True) if rows else None


for phase in PHASES:
    if not os.path.isdir(f"{OUT}/{phase}"):
        continue
    df = load(phase)
    if df is None:
        continue
    df["seq"] = df["seq"].astype(str)
    df["native"] = df["native"].astype(str)
    print(f"\n{'='*78}\n{phase}   {len(df):,} designs over {df['complex'].nunique()} complexes\n{'='*78}")

    agg = []
    for (m, arm), g in df.groupby(["model", "arm"]):
        per = g.groupby("complex")["seq"].nunique()
        agg.append({
            "model": m, "arm": arm, "n": len(g),
            "recovery": g["recovery"].mean(),
            "uniq/10k": per.mean(),
            "P1=M": (g["seq"].str[0] == "M").mean(),
            "Cterm_ok": (g["seq"].str[-1] == g["native"].str[-1]).mean(),
        })
    a = pd.DataFrame(agg).sort_values(["model", "arm"])
    print(a.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # TCR-present vs TCR-removed divergence
    print(f"\n  TCR effect (mean per-position Jensen-Shannon distance, full vs notcr):")
    lines = []
    for m in MODELS:
        sub = df[df["model"] == m]
        ds, drec = [], []
        for cid, g in sub.groupby("complex"):
            f_, n_ = g[g["arm"] == "full"], g[g["arm"] == "notcr"]
            if f_.empty or n_.empty:
                continue
            L = len(f_["native"].iloc[0])
            ds.append(js_dist(pwm(f_["seq"], L), pwm(n_["seq"], L)))
            drec.append(f_["recovery"].mean() - n_["recovery"].mean())
        if ds:
            lines.append(f"    {m:20s} JS={np.mean(ds):.3f} (range {min(ds):.2f}-{max(ds):.2f}, "
                         f"n={len(ds)})   recovery full-minus-notcr = {np.mean(drec):+.3f}")
    print("\n".join(lines) if lines else "    (no complex has both arms yet)")

    if phase.startswith("focus6am"):
        print(f"\n  per-complex detail:")
        for (cid, m, arm), g in df.groupby(["complex", "model", "arm"]):
            vc = g["seq"].value_counts(normalize=True)
            print(f"    {cid} {m:20s} {arm:6s} native={g['native'].iloc[0]}  "
                  f"top={vc.index[0]} ({vc.iloc[0]:.2f})  rec={g['recovery'].mean():.3f}")
