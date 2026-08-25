#!/usr/bin/env python3
"""Does the TCR-context benefit (full vs mhconly, measured as distance to real IEDB HLA-A2 binders)
depend on the real binder's affinity tier? Same nearest-Hamming-distance mechanic as
plot_iedb_hamming_summary.py, but tiers are kept SEPARATE instead of pooled, so we can compare the
full-vs-mhconly gap within each of Low/Intermediate/High.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/ubuntu/if-mhc")
FIG_DIR = ROOT / "figures/fig_iedb10_tier_tcr_effect"
FIG_DIR.mkdir(exist_ok=True, parents=True)
STRUCTS = ["2P5W", "1QSF", "1QRN", "2BNR", "2GJ6", "2F53", "2F54", "3QDG", "3QEQ", "3QFJ", "3GSN",
           "1OGA", "3UTS", "5C0A", "5C0B", "5HHO", "5EU6", "2VLR", "4MJI", "5NME"]
TIERS = ["Positive-Low", "Positive-Intermediate", "Positive-High"]
TIER_SHORT = {"Positive-Low": "Low", "Positive-Intermediate": "Intermediate", "Positive-High": "High"}
AA = list("ACDEFGHIKLMNPQRSTVWY")

dataset = pd.read_csv(ROOT / "inputs/pmhc_tcr_dataset/dataset.csv")


def peptide_from_ligandmpnn_line(line):
    return line.strip().split(":")[2]


def load_designs(pdb, cond):
    peps = []
    for weights, fname in [("vanilla", f"vanilla_{pdb}.fa"), ("noMHC", f"nomhc_{pdb}.fa")]:
        path = ROOT / f"outputs/panel/{pdb}/{cond}/mpnn/seqs/{fname}"
        with open(path) as f:
            lines = f.read().splitlines()
        for i in range(0, len(lines) - 1, 2):
            if lines[i].startswith(">"):
                peps.append(lines[i + 1].strip())
    path = ROOT / f"outputs/panel/{pdb}/{cond}/esmif/seqs/{pdb}.fa"
    with open(path) as f:
        lines = f.read().splitlines()
    for i in range(0, len(lines) - 1, 2):
        if lines[i].startswith(">"):
            peps.append(lines[i + 1].strip())
    path = ROOT / f"outputs/panel/{pdb}/{cond}/ligandmpnn/seqs/{pdb}.fa"
    with open(path) as f:
        lines = f.read().splitlines()
    for i in range(0, len(lines) - 1, 2):
        if lines[i].startswith(">"):
            peps.append(peptide_from_ligandmpnn_line(lines[i + 1]))
    return peps


def encode(seqs, length):
    return np.array([[ord(c) for c in s] for s in seqs], dtype=np.uint8)


def nearest_hamming_batched(real_peps, design_peps, length, batch=5000):
    design_arr = encode(design_peps, length)
    best_dist = np.full(len(real_peps), 255, dtype=np.int16)
    for start in range(0, len(real_peps), batch):
        chunk = real_peps[start:start + batch]
        chunk_arr = encode(chunk, length)
        dist = (chunk_arr[:, None, :] != design_arr[None, :, :]).sum(axis=2)
        best_dist[start:start + len(chunk)] = dist.min(axis=1)
    return best_dist


def main():
    iedb = pd.read_csv("/home/ubuntu/pmhc/modeling/hla_a2_iedb/hla_a2_epitopes_full.csv.gz")
    iedb = iedb[iedb["qualitative_measure"].isin(TIERS)]
    rng = np.random.RandomState(0)

    records = []
    for pdb in STRUCTS:
        native = dataset.loc[dataset.pdb == pdb, "peptide"].iloc[0]
        length = len(native)
        designs_by_cond = {}
        for cond in ["mhconly", "full"]:
            designs_by_cond[cond] = list(set(p for p in load_designs(pdb, cond) if len(p) == length))

        for tier in TIERS:
            sub = iedb[(iedb.qualitative_measure == tier) & (iedb.length == length)]
            real_peps = [p for p in sub["sequence"].dropna().unique()
                         if isinstance(p, str) and len(p) == length and all(c in AA for c in p)]
            if len(real_peps) == 0:
                continue
            if len(real_peps) > 500:
                real_peps = list(rng.choice(real_peps, size=500, replace=False))
            for cond in ["mhconly", "full"]:
                dists = nearest_hamming_batched(real_peps, designs_by_cond[cond], length)
                for d in dists:
                    records.append({"pdb": pdb, "tier": tier, "condition": cond, "hamming_dist": int(d)})
        print(f"{pdb}: done", flush=True)

    df = pd.DataFrame(records)
    df.to_csv(ROOT / "outputs/analysis/iedb_tier_tcr_effect_data.csv", index=False)

    summary = df.groupby(["tier", "condition"])["hamming_dist"].agg(["mean", "sem"]).reset_index()
    print(summary)

    pivot = summary.pivot(index="tier", columns="condition", values="mean")
    pivot = pivot.loc[TIERS]
    pivot["delta_closer_with_TCR"] = pivot["mhconly"] - pivot["full"]
    print()
    print(pivot)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    width = 0.32
    x = np.arange(len(TIERS))
    for k, cond in enumerate(["mhconly", "full"]):
        means, sems = [], []
        for tier in TIERS:
            row = summary[(summary.tier == tier) & (summary.condition == cond)]
            means.append(row["mean"].values[0])
            sems.append(row["sem"].values[0])
        color = "#4C72B0" if cond == "mhconly" else "#C44E52"
        ax.bar(x + (k - 0.5) * width, means, width=width, yerr=sems, capsize=3, color=color,
               label="mhconly (no TCR)" if cond == "mhconly" else "full (+TCR)",
               edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([TIER_SHORT[t] for t in TIERS])
    ax.set_xlabel("real IEDB binder affinity tier")
    ax.set_ylabel("mean Hamming distance to nearest real IEDB binder\n(all 20 crystals pooled)")
    ax.set_title("Does the TCR-context benefit depend on real-binder affinity tier?")
    ax.legend()
    fig.tight_layout()
    out = FIG_DIR / "fig_iedb10_tier_tcr_effect.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
