#!/usr/bin/env python3
"""One consolidated summary of native-peptide recovery across the whole 20-structure panel:
violin per crystal (x-axis = crystal), each violin pooling per-position recovery rates across the
4 models (up to 10 positions x 4 models per crystal), with individual (position, model) points
overlaid, colored by model. Two rows, same crystal order and shared y-axis, so TCR-present vs
TCR-absent is a direct top-vs-bottom comparison per crystal:
  - top    = full    (MHC + TCR input)
  - bottom = mhconly (MHC only, NO TCR input)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/ubuntu/if-mhc")
FIG_DIR = ROOT / "figures/fig_iedb7_crystal_summary"
FIG_DIR.mkdir(exist_ok=True, parents=True)
MODELS = ["vanilla", "noMHC", "ESM-IF1", "LigandMPNN"]
MODEL_COLOR = {"vanilla": "#0072B2", "noMHC": "#E69F00", "ESM-IF1": "#009E73", "LigandMPNN": "#CC79A7"}
MODEL_LABEL = {"vanilla": "ProteinMPNN", "noMHC": "noMHC ProteinMPNN (No MHC)", "ESM-IF1": "ESM-IF1",
               "LigandMPNN": "LigandMPNN"}
STRUCTS = ["2P5W", "1QSF", "1QRN", "2BNR", "2GJ6", "2F53", "2F54", "3QDG", "3QEQ", "3QFJ", "3GSN",
           "1OGA", "3UTS", "5C0A", "5C0B", "5HHO", "5EU6", "2VLR", "4MJI", "5NME"]
CONDITIONS = ["full", "mhconly"]
COND_LABEL = {"full": "full (MHC + TCR)", "mhconly": "mhconly (NO TCR input)"}

dataset = pd.read_csv(ROOT / "inputs/pmhc_tcr_dataset/dataset.csv")
natives = {pdb: dataset.loc[dataset.pdb == pdb, "peptide"].iloc[0] for pdb in STRUCTS}


def peptide_from_ligandmpnn_line(line):
    return line.strip().split(":")[2]


def load_designs(pdb, cond):
    rows = []
    for weights, fname in [("vanilla", f"vanilla_{pdb}.fa"), ("noMHC", f"nomhc_{pdb}.fa")]:
        path = ROOT / f"outputs/panel/{pdb}/{cond}/mpnn/seqs/{fname}"
        with open(path) as f:
            lines = f.read().splitlines()
        for i in range(0, len(lines) - 1, 2):
            if lines[i].startswith(">"):
                rows.append({"peptide": lines[i + 1].strip(), "model": weights})
    path = ROOT / f"outputs/panel/{pdb}/{cond}/esmif/seqs/{pdb}.fa"
    with open(path) as f:
        lines = f.read().splitlines()
    for i in range(0, len(lines) - 1, 2):
        if lines[i].startswith(">"):
            rows.append({"peptide": lines[i + 1].strip(), "model": "ESM-IF1"})
    path = ROOT / f"outputs/panel/{pdb}/{cond}/ligandmpnn/seqs/{pdb}.fa"
    with open(path) as f:
        lines = f.read().splitlines()
    for i in range(0, len(lines) - 1, 2):
        if lines[i].startswith(">"):
            rows.append({"peptide": peptide_from_ligandmpnn_line(lines[i + 1]), "model": "LigandMPNN"})
    return pd.DataFrame(rows)


def per_position_recovery(pdb, cond):
    native = natives[pdb]
    length = len(native)
    df = load_designs(pdb, cond)
    mat = np.full((length, len(MODELS)), np.nan)
    for j, model in enumerate(MODELS):
        peps = [p for p in df.loc[df.model == model, "peptide"] if len(p) == length]
        if not peps:
            continue
        hits = np.zeros(length)
        for p in peps:
            for pos in range(length):
                if p[pos] == native[pos]:
                    hits[pos] += 1
        mat[:, j] = hits / len(peps)
    return mat


def main():
    mats = {(pdb, cond): per_position_recovery(pdb, cond) for pdb in STRUCTS for cond in CONDITIONS}

    fig, axes = plt.subplots(2, 1, figsize=(2 + 1.0 * len(STRUCTS), 11), sharex=True, sharey=True)
    rng_jitter = np.random.RandomState(0)

    for ax, cond in zip(axes, CONDITIONS):
        data = []
        for pdb in STRUCTS:
            mat = mats[(pdb, cond)]
            data.append(mat[~np.isnan(mat)])
        parts = ax.violinplot(data, positions=range(1, len(STRUCTS) + 1), showmeans=True, widths=0.8)
        for pc in parts["bodies"]:
            pc.set_facecolor("#4C72B0" if cond == "full" else "#C44E52")
            pc.set_alpha(0.30)
        for key in ["cmeans", "cmaxes", "cmins", "cbars"]:
            if key in parts:
                parts[key].set_color("#333333")

        for i, pdb in enumerate(STRUCTS):
            mat = mats[(pdb, cond)]
            length = mat.shape[0]
            for j, model in enumerate(MODELS):
                vals = mat[:, j]
                vals = vals[~np.isnan(vals)]
                xs = (i + 1) + rng_jitter.uniform(-0.3, 0.3, size=len(vals))
                ax.scatter(xs, vals, s=16, alpha=0.75, color=MODEL_COLOR[model],
                           edgecolors="black", linewidths=0.3, zorder=5)

        ax.set_ylabel("recovery of native residue\n(pooled across positions x models)")
        ax.set_title(f"{COND_LABEL[cond]}  (n={len(MODELS)} models x up to 10 positions per crystal)",
                      fontsize=11)
        ax.set_ylim(-0.05, 1.05)

    axes[-1].set_xticks(range(1, len(STRUCTS) + 1))
    axes[-1].set_xticklabels([f"{natives[pdb]}\n({pdb})" for pdb in STRUCTS], fontsize=8, rotation=45,
                              ha="right")
    axes[-1].set_xlabel("crystal (peptide)")

    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=MODEL_COLOR[m],
                          markeredgecolor="black", markersize=7, label=MODEL_LABEL[m]) for m in MODELS]
    fig.legend(handles=handles, title="model", loc="upper right", bbox_to_anchor=(0.995, 0.995),
               fontsize=8)
    fig.suptitle(f"Panel-wide native recovery summary, all {len(STRUCTS)} crystals\n"
                 "top = full (MHC+TCR), bottom = mhconly (NO TCR) -- same crystal order, shared y-axis",
                 y=1.0, fontsize=13)
    fig.tight_layout()
    out = FIG_DIR / "fig_iedb7_crystal_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
