#!/usr/bin/env python3
"""Merge SKEMPI sweep part files into per-model tables and summaries.

Writes under outputs/skempi_if/:
  merged/<model>__<arm>.csv    every sampled epitope for that model/arm
  summary.csv                  per model/arm/complex: n, unique, recovery, entropy
  position_freqs.csv           long-format per-position AA frequencies (the
                               substrate for the TCR-present vs TCR-removed
                               comparison)
"""
import glob, os
import numpy as np
import pandas as pd

ROOT = "/global/scratch/users/sergiomar10/if-mhc"
OUT = f"{ROOT}/outputs/skempi_if"
# merge one (dataset, temperature) phase; matches the layout skempi_units.py writes
DATASET = os.environ.get("SK_DATASET", "skempi")
TEMP = os.environ.get("SK_TEMP", "0.1")
PHASE = f"{DATASET}/T{TEMP}"
POUT = f"{OUT}/{PHASE}"
AA = "ACDEFGHIKLMNPQRSTVWY"
MODELS = ["esmif", "proteinmpnn", "proteinmpnn_nomhc", "ligandmpnn"]
ARMS = ["full", "notcr"]


def main():
    os.makedirs(f"{POUT}/merged", exist_ok=True)
    summary, posrows = [], []

    for model in MODELS:
        parts = sorted(glob.glob(f"{POUT}/{model}/parts/*.csv"))
        if not parts:
            continue
        df = pd.concat((pd.read_csv(p) for p in parts), ignore_index=True)
        df["model"] = model
        for arm in ARMS:
            sub = df[df["arm"] == arm]
            if sub.empty:
                continue
            sub.to_csv(f"{POUT}/merged/{model}__{arm}.csv", index=False)
            for cid, g in sub.groupby("complex"):
                seqs = g["seq"].astype(str).tolist()
                nat = str(g["native"].iloc[0])
                L = len(nat)
                counts = np.zeros((L, 20))
                for s in seqs:
                    for i, c in enumerate(s[:L]):
                        if c in AA:
                            counts[i, AA.index(c)] += 1
                freq = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    ent = -np.nansum(np.where(freq > 0, freq * np.log2(freq), 0), axis=1)
                summary.append({
                    "dataset": DATASET, "temp": TEMP, "model": model, "arm": arm, "complex": cid, "native": nat,
                    "n": len(seqs), "n_unique": len(set(seqs)),
                    "mean_recovery": round(float(g["recovery"].mean()), 4),
                    "frac_native_seq": round(float(np.mean([s == nat for s in seqs])), 4),
                    "mean_entropy_bits": round(float(ent.mean()), 4),
                    "top_seq": max(set(seqs), key=seqs.count),
                    "top_seq_frac": round(seqs.count(max(set(seqs), key=seqs.count)) / len(seqs), 4),
                })
                for i in range(L):
                    for j, aa in enumerate(AA):
                        if freq[i, j] > 0:
                            posrows.append({"model": model, "arm": arm, "complex": cid,
                                            "position": i + 1, "native_aa": nat[i],
                                            "aa": aa, "freq": round(float(freq[i, j]), 6)})
        print(f"{model}: {len(parts)} parts -> {len(df):,} sequences")

    if summary:
        s = pd.DataFrame(summary).sort_values(["model", "arm", "complex"])
        s.to_csv(f"{POUT}/summary.csv", index=False)
        pd.DataFrame(posrows).to_csv(f"{POUT}/position_freqs.csv", index=False)
        print(f"\n{len(s)} model/arm/complex cells -> {OUT}/summary.csv")
        print(s.groupby(["model", "arm"])
               .agg(cells=("complex", "size"), seqs=("n", "sum"),
                    recovery=("mean_recovery", "mean"),
                    entropy=("mean_entropy_bits", "mean"))
               .round(3).to_string())


if __name__ == "__main__":
    main()
