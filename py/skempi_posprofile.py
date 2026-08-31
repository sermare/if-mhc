#!/usr/bin/env python3
"""Per-position identity to native, to separate real signal from terminus artifacts.

Aggregate recovery mixes two very different things: the models' handling of the
buried middle of the epitope, and their handling of the two chain termini, which
are unnatural contexts (a free N- and C-terminus sitting inside the MHC groove).
Splitting recovery by position shows which is which.
"""
import os
import collections, glob
import numpy as np
import pandas as pd

OUT = "outputs/skempi_if"
# Phase selection: which (dataset, temperature) run to analyse. Defaults to the
# T=0.1 SKEMPI phase so existing invocations keep working; set SK_DATASET /
# SK_TEMP to point the same analysis at another phase.
DATASET = os.environ.get("SK_DATASET", "skempi")
TEMP = os.environ.get("SK_TEMP", "0.1")
TAG = "t" + TEMP.replace(".", "")
SUF = f"_{DATASET}_T{TEMP}"

MODELS = ["esmif", "proteinmpnn", "proteinmpnn_nomhc", "ligandmpnn"]

rows = []
for m in MODELS:
    for f in glob.glob(f"{OUT}/{DATASET}/T{TEMP}/{m}/parts/*.csv"):
        try:
            d = pd.read_csv(f, usecols=["arm", "seq", "native"])
        except Exception:
            continue
        if len(d):
            d["model"] = m
            rows.append(d)

df = pd.concat(rows, ignore_index=True)
df["seq"] = df.seq.astype(str)
df["native"] = df.native.astype(str)
df = df[df.native.str.len() == 9]          # the 9-mers: the dominant length class

print(f"PHASE {DATASET}/T{TEMP}")
print(f"per-position identity to native, 9-mer epitopes only ({len(df):,} designs)\n")
print(f"{'model / arm':26s} " + " ".join(f"P{i+1:<5d}" for i in range(9)))
for (m, arm), g in df.groupby(["model", "arm"]):
    s, n = g.seq.values, g.native.values
    acc = [np.mean([a[i] == b[i] for a, b in zip(s, n)]) for i in range(9)]
    print(f"{m + ' / ' + arm:26s} " + " ".join(f"{x:.2f}  " for x in acc))

print()
mid = df[df.model == "proteinmpnn"]
s, n = mid.seq.values, mid.native.values
acc = [np.mean([a[i] == b[i] for a, b in zip(s, n)]) for i in range(9)]
print(f"ProteinMPNN: termini (P1,P9) mean = {np.mean([acc[0], acc[8]]):.3f}   "
      f"core (P2-P8) mean = {np.mean(acc[1:8]):.3f}")

print()
for pos, label in [(0, "P1"), (8, "P9 (C-term)")]:
    des = collections.Counter(df.seq.str[pos])
    nat = collections.Counter(df.native.str[pos])
    td, tn = sum(des.values()), sum(nat.values())
    print(f"{label:12s} designed = " + ", ".join(f"{c}:{v/td:.2f}" for c, v in des.most_common(4))
          + "  |  native = " + ", ".join(f"{c}:{v/tn:.2f}" for c, v in nat.most_common(4)))
