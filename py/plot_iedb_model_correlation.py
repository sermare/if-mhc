#!/usr/bin/env python3
"""Correlation between models' per-position recovery patterns, pooled across all 7 crystals x 9
positions (63 data points per model), separately for mhconly (no TCR) and full (with TCR). Answers:
do models succeed/fail on the same positions and crystals, or are they capturing different things?
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/ubuntu/if-mhc")
FIG_DIR = ROOT / "figures/fig_iedb4_model_correlation"
FIG_DIR.mkdir(exist_ok=True, parents=True)
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


def per_position_recovery(pdb, cond, native, length):
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
    for cond, cond_label in [("mhconly", "mhconly (NO TCR input)"), ("full", "full (MHC + TCR)")]:
        all_rows = []
        for pdb in STRUCTS:
            native = dataset.loc[dataset.pdb == pdb, "peptide"].iloc[0]
            mat = per_position_recovery(pdb, cond, native, len(native))
            all_rows.append(mat)
        pooled = np.vstack(all_rows)  # (63, 4)
        print(f"{cond}: pooled shape={pooled.shape}")

        corr = np.corrcoef(pooled.T)
        fig, ax = plt.subplots(figsize=(6, 5.5))
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        model_labels = [MODEL_LABEL[m] for m in MODELS]
        ax.set_xticks(range(len(MODELS))); ax.set_xticklabels(model_labels, rotation=30, ha="right")
        ax.set_yticks(range(len(MODELS))); ax.set_yticklabels(model_labels)
        for i in range(len(MODELS)):
            for j in range(len(MODELS)):
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                        color="white" if abs(corr[i, j]) > 0.6 else "black", fontsize=11)
        ax.set_title(f"Model agreement on per-position recovery -- {cond_label}\n"
                     f"(Pearson r, pooled across 7 crystals x 9 positions, n={pooled.shape[0]})")
        fig.colorbar(im, ax=ax, fraction=0.046, label="Pearson r")
        fig.tight_layout()
        out = FIG_DIR / f"fig_iedb4_model_correlation_{cond}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"wrote {out}")
        print(pd.DataFrame(corr, index=MODELS, columns=MODELS).round(2))
        print()


if __name__ == "__main__":
    main()
