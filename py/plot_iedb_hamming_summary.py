#!/usr/bin/env python3
"""ONE consolidated summary of the IEDB Hamming-distance comparison (replaces the 14 per-crystal
histogram grids + scatter plot from earlier passes). For each crystal, mean nearest-Hamming-distance
from real IEDB peptides (pooled across the 3 ordered binding-strength tiers, since tiers don't
differ much within a condition -- established earlier) to the nearest design, mhconly vs full,
plus the pooled all-crystal summary as the rightmost group.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/ubuntu/if-mhc")
FIG_DIR = ROOT / "figures/fig_iedb3_tier_hamming_hist"
FIG_DIR.mkdir(exist_ok=True, parents=True)
STRUCTS = ["2P5W", "1QSF", "1QRN", "2BNR", "2GJ6", "2F53", "2F54", "3QDG", "3QEQ", "3QFJ", "3GSN",
           "1OGA", "3UTS", "5C0A", "5C0B", "5HHO", "5EU6", "2VLR", "4MJI", "5NME"]
TIERS = ["Positive-Low", "Positive-Intermediate", "Positive-High"]
AA = list("ACDEFGHIKLMNPQRSTVWY")
COND_COLOR = {"mhconly": "#4C72B0", "full": "#C44E52"}
COND_LABEL = {"mhconly": "mhconly (no TCR)", "full": "full (+TCR)"}

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
        real_peps = []
        for tier in TIERS:
            sub = iedb[(iedb.qualitative_measure == tier) & (iedb.length == length)]
            peps = [p for p in sub["sequence"].dropna().unique() if isinstance(p, str) and len(p) == length
                    and all(c in AA for c in p)]
            if len(peps) > 500:
                peps = list(rng.choice(peps, size=500, replace=False))
            real_peps.extend(peps)

        for cond in ["mhconly", "full"]:
            design_peps = list(set(p for p in load_designs(pdb, cond) if len(p) == length))
            dists = nearest_hamming_batched(real_peps, design_peps, length)
            for d in dists:
                records.append({"pdb": pdb, "condition": cond, "hamming_dist": int(d)})
        print(f"{pdb}: done", flush=True)

    df = pd.DataFrame(records)
    df.to_csv(ROOT / "outputs/analysis/iedb_hamming_summary_data.csv", index=False)

    summary = df.groupby(["pdb", "condition"])["hamming_dist"].agg(["mean", "sem"]).reset_index()
    pooled = df.groupby("condition")["hamming_dist"].agg(["mean", "sem"]).reset_index()
    pooled["pdb"] = "ALL (pooled)"

    fig, ax = plt.subplots(figsize=(3 + 1.1 * (len(STRUCTS) + 1), 6))
    x_labels = STRUCTS + ["ALL\n(pooled)"]
    width = 0.32
    for k, cond in enumerate(["mhconly", "full"]):
        means, sems = [], []
        for pdb in STRUCTS:
            row = summary[(summary.pdb == pdb) & (summary.condition == cond)]
            means.append(row["mean"].values[0] if len(row) else np.nan)
            sems.append(row["sem"].values[0] if len(row) else np.nan)
        prow = pooled[pooled.condition == cond]
        means.append(prow["mean"].values[0]); sems.append(prow["sem"].values[0])
        xpos = np.arange(len(x_labels)) + (k - 0.5) * width
        ax.bar(xpos, means, width=width, yerr=sems, capsize=3, color=COND_COLOR[cond],
               label=COND_LABEL[cond], edgecolor="black", linewidth=0.5)

    ax.axvline(len(STRUCTS) - 0.5, color="gray", linestyle=":", linewidth=1)
    ax.set_xticks(range(len(x_labels))); ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_ylabel("mean Hamming distance to nearest real IEDB binder\n(pooled across Low/Intermediate/High tiers)")
    ax.set_title("Do TCR-conditioned designs land closer to real HLA-A2 binders? -- all crystals + pooled")
    ax.legend(loc="best")
    fig.tight_layout()
    out = FIG_DIR / "fig_iedb3_hamming_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
    print(summary)
    print(pooled)


if __name__ == "__main__":
    main()
