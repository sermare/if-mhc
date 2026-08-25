#!/usr/bin/env python3
"""One consolidated matrix per condition: rows = crystals (7), columns = models (4), cell = mean
recovery vs. native peptide across all positions. No delta column -- just mhconly and full side by
side as two separate heatmaps, easy to scan across crystals at a glance."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/ubuntu/if-mhc")
FIG_DIR = ROOT / "figures/fig_iedb1_recovery_matrix"
MODELS = ["vanilla", "noMHC", "ESM-IF1", "LigandMPNN"]
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


def mean_recovery_per_model(pdb, cond, native, length):
    df = load_designs(pdb, cond)
    out = np.zeros(len(MODELS))
    for j, model in enumerate(MODELS):
        peps = [p for p in df.loc[df.model == model, "peptide"] if len(p) == length]
        if not peps:
            out[j] = np.nan
            continue
        matches = sum(sum(a == b for a, b in zip(p, native)) for p in peps)
        out[j] = matches / (len(peps) * length)
    return out


def main():
    for cond, cond_label in [("mhconly", "mhconly (NO TCR input)"), ("full", "full (MHC + TCR)")]:
        mat = np.zeros((len(STRUCTS), len(MODELS)))
        natives = []
        for i, pdb in enumerate(STRUCTS):
            native = dataset.loc[dataset.pdb == pdb, "peptide"].iloc[0]
            natives.append(native)
            mat[i, :] = mean_recovery_per_model(pdb, cond, native, len(native))

        fig, ax = plt.subplots(figsize=(7, 6.5))
        im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=mat.max(), aspect="auto")
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels([MODEL_LABEL[m] for m in MODELS], rotation=30, ha="right")
        ax.set_yticks(range(len(STRUCTS)))
        ax.set_yticklabels([f"{pdb} ({nat})" for pdb, nat in zip(STRUCTS, natives)])
        for i in range(len(STRUCTS)):
            for j in range(len(MODELS)):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        color="white" if mat[i, j] < mat.max() * 0.6 else "black", fontsize=9)
        ax.set_xlabel("model")
        ax.set_title(f"Mean recovery vs. native peptide -- {cond_label}\n(all positions averaged, all 7 crystals)")
        fig.colorbar(im, ax=ax, fraction=0.046, label="mean recovery")
        fig.tight_layout()
        out = FIG_DIR / f"fig_iedb1_recovery_crystal_by_model_{cond}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
