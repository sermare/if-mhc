#!/usr/bin/env python3
"""Position x amino-acid frequency matrix (raw usage, not confusion against native), one heatmap
per model, both structures, at a given temperature. Complements fig_if5 (which is native-AA vs
designed-AA confusion) with the simpler "how often is each AA used at each position" view.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = "/home/ubuntu/if-mhc/"
NATIVE = {"3HG1": "ELAGIGILTV", "2P5E": "SLLMWITQC"}
STRUCT_LABEL = {"3HG1": "3HG1 / MEL5", "2P5E": "2P5E / NY-ESO-1"}
MODELS = ["vanilla", "noMHC", "ESM-IF1", "LigandMPNN"]
AA = list("ACDEFGHIKLMNPQRSTVWY")

SOURCES = {
    ("3HG1", 0.1): {
        "vanilla": f"{ROOT}outputs/mpnn_3hg1_100k/archive_T01_partial/vanilla_3HG1_T01_partial_26993.fa",
        "noMHC": f"{ROOT}outputs/mpnn_3hg1_100k/archive_T01_partial/nomhc_3HG1_T01_partial_26528.fa",
        "ESM-IF1": f"{ROOT}outputs/esmif_3hg1_pilot/seqs/3HG1.fa",
        "LigandMPNN": f"{ROOT}outputs/ligandmpnn_3hg1_pilot/seqs/3HG1.fa",
    },
    ("3HG1", 0.3): {
        "vanilla": f"{ROOT}outputs/mpnn_3hg1_T03_50k/run_vanilla/seqs/3HG1.fa",
        "noMHC": f"{ROOT}outputs/mpnn_3hg1_T03_50k/run_nomhc/seqs/3HG1.fa",
        "ESM-IF1": f"{ROOT}outputs/esmif_3hg1_T03_20k/seqs/3HG1.fa",
        "LigandMPNN": f"{ROOT}outputs/ligandmpnn_3hg1_T03_20k/seqs/3HG1.fa",
    },
    ("2P5E", 0.1): {
        "vanilla": f"{ROOT}outputs/mpnn_2p5e_T01_20k/seqs/vanilla_2P5E.fa",
        "noMHC": f"{ROOT}outputs/mpnn_2p5e_T01_20k/seqs/nomhc_2P5E.fa",
        "ESM-IF1": f"{ROOT}outputs/esmif_2p5e_pilot/seqs/2P5E.fa",
        "LigandMPNN": f"{ROOT}outputs/ligandmpnn_2p5e_pilot/seqs/2P5E.fa",
    },
    ("2P5E", 0.3): {
        "vanilla": f"{ROOT}outputs/mpnn_2p5e_T03_20k/seqs/vanilla_2P5E.fa",
        "noMHC": f"{ROOT}outputs/mpnn_2p5e_T03_20k/seqs/nomhc_2P5E.fa",
        "ESM-IF1": f"{ROOT}outputs/esmif_2p5e_T03_20k/seqs/2P5E.fa",
        "LigandMPNN": f"{ROOT}outputs/ligandmpnn_2p5e_T03_20k/seqs/2P5E.fa",
    },
}


def peptide_from_ligandmpnn_line(line):
    return line.strip().split(":")[2]


def load_peptides(path, tool):
    seqs = []
    with open(path) as f:
        lines = f.read().splitlines()
    for i in range(0, len(lines) - 1, 2):
        header, seq = lines[i], lines[i + 1]
        if not header.startswith(">"):
            continue
        seqs.append(peptide_from_ligandmpnn_line(seq) if tool == "LigandMPNN" else seq.strip())
    return seqs


def freq_matrix(peps, length):
    mat = pd.DataFrame(0, index=range(1, length + 1), columns=AA, dtype=float)
    for p in peps:
        if len(p) != length:
            continue
        for pos, aa in enumerate(p, start=1):
            if aa in AA:
                mat.loc[pos, aa] += 1
    row_sums = mat.sum(axis=1).replace(0, np.nan)
    return mat.div(row_sums, axis=0)


def build(temp):
    fig, axes = plt.subplots(2, 4, figsize=(19, 11))
    fig.subplots_adjust(hspace=0.7, top=0.90)
    for row, struct in enumerate(["3HG1", "2P5E"]):
        native = NATIVE[struct]
        length = len(native)
        for col, model in enumerate(MODELS):
            peps = load_peptides(SOURCES[(struct, temp)][model], model)
            mat = freq_matrix(peps, length)
            ax = axes[row, col]
            im = ax.imshow(mat.values.T, cmap="viridis", vmin=0, vmax=1, aspect="auto")
            ax.set_yticks(range(len(AA))); ax.set_yticklabels(AA, fontsize=7)
            ax.set_xticks(range(length)); ax.set_xticklabels(range(1, length + 1), fontsize=7)
            ax.set_title(f"{model}  (n={len(peps):,})", fontsize=10)
            ax.set_xlabel("position", fontsize=8)
            if col == 0:
                ax.set_ylabel("amino acid", fontsize=8)
        fig.colorbar(im, ax=axes[row, :].tolist(), fraction=0.02, pad=0.01,
                     label="P(amino acid | position)")
    for row, struct in enumerate(["3HG1", "2P5E"]):
        top_y = axes[row, 0].get_position().y1
        fig.text(0.42, top_y + 0.075, f"{STRUCT_LABEL[struct]} -- position x amino-acid usage "
                  f"(T={temp}, all raw designs)  --  native/index peptide: {NATIVE[struct]}",
                  ha="center", fontsize=12, transform=fig.transFigure)
    suffix = "" if temp == 0.1 else f"_T{str(temp).replace('0.', '')}"
    out = f"{ROOT}figures/fig_if12_position_aa_matrix/fig_if12_position_aa_matrix{suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    temps = [float(t) for t in sys.argv[1:]] or [0.1]
    for t in temps:
        build(t)
