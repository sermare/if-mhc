#!/usr/bin/env python3
"""Per-residue version of the crystal x model recovery matrix: same 7-crystal x 4-model outer grid,
but each cell is now a 9-position recovery strip instead of one averaged number."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
        length = 9  # all 7 crystals in this panel are 9-mers
        all_mats = {}
        for pdb in STRUCTS:
            native = dataset.loc[dataset.pdb == pdb, "peptide"].iloc[0]
            all_mats[pdb] = (native, per_position_recovery(pdb, cond, native, len(native)))

        fig = plt.figure(figsize=(1.6 * len(MODELS) + 2.2, 1.7 * len(STRUCTS) + 1))
        outer = gridspec.GridSpec(len(STRUCTS), len(MODELS), figure=fig, wspace=0.15, hspace=0.35)

        for i, pdb in enumerate(STRUCTS):
            native, mat = all_mats[pdb]
            for j, model in enumerate(MODELS):
                ax = fig.add_subplot(outer[i, j])
                col = mat[:, j].reshape(-1, 1)
                im = ax.imshow(col, cmap="viridis", vmin=0, vmax=1, aspect="auto")
                ax.set_xticks([])
                if j == 0:
                    ax.set_yticks(range(len(native)))
                    ax.set_yticklabels(range(1, len(native) + 1), fontsize=6)
                    ax.set_ylabel(f"{pdb}\n{native}", fontsize=8, rotation=0, ha="right", va="center",
                                  labelpad=30)
                else:
                    ax.set_yticks([])
                if i == 0:
                    ax.set_title(MODEL_LABEL[model], fontsize=9)
                if i == len(STRUCTS) - 1:
                    ax.set_xlabel("")

        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, label="recovery of native residue")
        fig.suptitle(f"Per-residue recovery, {cond_label} -- rows=crystal, columns=model\n"
                     "(each cell: positions 1-9 top-to-bottom)", y=1.0, fontsize=13)
        out = FIG_DIR / f"fig_iedb1_recovery_crystal_by_model_per_residue_{cond}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
