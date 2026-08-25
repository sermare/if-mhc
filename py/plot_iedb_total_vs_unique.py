#!/usr/bin/env python3
"""Total designs vs. unique peptides for the panel-wide structures, same visual convention as
fig_if10_total_vs_unique: light fill = total designs, dark overlay = unique peptides among them,
black text = exact unique count. One figure per condition (mhconly, full), x-axis = crystal,
grouped bars per model.
"""
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/ubuntu/if-mhc")
FIG_DIR = ROOT / "figures/fig_iedb5_total_vs_unique"
FIG_DIR.mkdir(exist_ok=True, parents=True)
MODELS = ["vanilla", "noMHC", "ESM-IF1", "LigandMPNN"]
MODEL_COLOR = {"vanilla": "#0072B2", "noMHC": "#E69F00", "ESM-IF1": "#009E73", "LigandMPNN": "#CC79A7"}
MODEL_LABEL = {"vanilla": "ProteinMPNN", "noMHC": "noMHC ProteinMPNN (No MHC)", "ESM-IF1": "ESM-IF1",
               "LigandMPNN": "LigandMPNN"}
STRUCTS = ["2P5W", "1QSF", "1QRN", "2BNR", "2GJ6", "2F53", "2F54", "3QDG", "3QEQ", "3QFJ", "3GSN",
           "1OGA", "3UTS", "5C0A", "5C0B", "5HHO", "5EU6", "2VLR", "4MJI", "5NME"]

dataset = pd.read_csv(ROOT / "inputs/pmhc_tcr_dataset/dataset.csv")


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


def main():
    for cond, cond_label in [("mhconly", "mhconly (NO TCR input)"), ("full", "full (MHC + TCR)")]:
        fig, ax = plt.subplots(figsize=(3 + 1.3 * len(STRUCTS), 6.5))
        x = 0
        xticks, xlabels = [], []
        for pdb in STRUCTS:
            df = load_designs(pdb, cond)
            group_center = x + (len(MODELS) - 1) / 2
            for model in MODELS:
                peps = df.loc[df.model == model, "peptide"].tolist()
                total = len(peps)
                unique = len(set(peps))
                hue = MODEL_COLOR[model]
                ax.bar(x, total, width=0.8, color=hue, alpha=0.30, edgecolor=hue, linewidth=0.8)
                ax.bar(x, unique, width=0.8, color=hue, alpha=1.0)
                label_y = max(unique, total * 0.02) + total * 0.02
                ax.text(x, label_y, f"{unique:,}", ha="center", va="bottom", fontsize=7,
                        color="black", rotation=90)
                x += 1
            xticks.append(group_center)
            xlabels.append(pdb)
            x += 1.2  # gap between crystal groups

        ax.set_xticks(xticks); ax.set_xticklabels(xlabels, rotation=45, ha="right")
        ax.set_ylabel("designs")
        ax.set_title(f"Total designs vs. unique peptides -- {cond_label}\nall {len(STRUCTS)} panel crystals, T=0.1")
        legend_handles = [plt.matplotlib.patches.Patch(color=MODEL_COLOR[m], label=MODEL_LABEL[m]) for m in MODELS]
        legend_handles.append(plt.matplotlib.patches.Patch(facecolor="gray", alpha=0.30, edgecolor="gray",
                                                             label="total designs (light)"))
        legend_handles.append(plt.matplotlib.patches.Patch(facecolor="gray", alpha=1.0, label="unique peptides (dark)"))
        ax.legend(handles=legend_handles, loc="upper right", ncol=2, fontsize=8)
        fig.tight_layout()
        out = FIG_DIR / f"fig_iedb5_total_vs_unique_{cond}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
