#!/usr/bin/env python3
"""Sequence logos for the panel-wide designs: one combined figure per condition (full, mhconly),
grid = 20 crystals (rows) x 4 models (cols), each subplot a logomaker sequence logo built from that
crystal's UNIQUE designs (same convention as fig_if4_logos). The x-axis of every logo is labeled with
the crystal's own native/index peptide (one letter per position) instead of plain position numbers,
so the reader can directly compare each column's dominant letter against what the model was
"supposed" to recover.
"""
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logomaker

ROOT = Path("/home/ubuntu/if-mhc")
FIG_DIR = ROOT / "figures/fig_iedb8_logos"
FIG_DIR.mkdir(exist_ok=True, parents=True)
AA = list("ACDEFGHIKLMNPQRSTVWY")
MODELS = ["vanilla", "noMHC", "ESM-IF1", "LigandMPNN"]
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


def load_peptides(pdb, cond, model):
    if model in ("vanilla", "noMHC"):
        fname = "vanilla_" + pdb + ".fa" if model == "vanilla" else "nomhc_" + pdb + ".fa"
        path = ROOT / f"outputs/panel/{pdb}/{cond}/mpnn/seqs/{fname}"
        tool = "mpnn"
    elif model == "ESM-IF1":
        path = ROOT / f"outputs/panel/{pdb}/{cond}/esmif/seqs/{pdb}.fa"
        tool = "esmif"
    else:
        path = ROOT / f"outputs/panel/{pdb}/{cond}/ligandmpnn/seqs/{pdb}.fa"
        tool = "ligandmpnn"
    with open(path) as f:
        lines = f.read().splitlines()
    seqs = []
    for i in range(0, len(lines) - 1, 2):
        if not lines[i].startswith(">"):
            continue
        seq = lines[i + 1]
        seqs.append(peptide_from_ligandmpnn_line(seq) if tool == "ligandmpnn" else seq.strip())
    return seqs


def counts_matrix(seqs, length):
    mat = pd.DataFrame(0.0, index=range(1, length + 1), columns=AA)
    for seq in seqs:
        if len(seq) != length:
            continue
        for pos, aa in enumerate(seq, start=1):
            if aa in AA:
                mat.loc[pos, aa] += 1.0
    return mat


def main():
    for cond in CONDITIONS:
        fig, axes = plt.subplots(len(STRUCTS), len(MODELS),
                                  figsize=(3.2 * len(MODELS), 2.1 * len(STRUCTS)))
        for row, pdb in enumerate(STRUCTS):
            native = natives[pdb]
            length = len(native)
            for col, model in enumerate(MODELS):
                ax = axes[row, col]
                peps = load_peptides(pdb, cond, model)
                uniq = sorted(set(p for p in peps if len(p) == length))
                mat = counts_matrix(uniq, length)
                logomaker.Logo(mat, ax=ax, color_scheme="chemistry")
                ax.set_xticks(range(1, length + 1))
                ax.set_xticklabels(list(native), fontsize=8)
                ax.set_yticks([])
                if row == 0:
                    ax.set_title(MODEL_LABEL[model], fontsize=11)
                if col == 0:
                    ax.set_ylabel(f"{native}\n({pdb})", fontsize=9, rotation=0, ha="right", va="center",
                                  labelpad=35)
                ax.text(0.98, 0.95, f"n={len(uniq):,}", transform=ax.transAxes, ha="right", va="top",
                        fontsize=7, color="#555555")
        fig.suptitle(f"Sequence logos, panel-wide unique designs -- {COND_LABEL[cond]}\n"
                     "x-axis = crystal's own native/index peptide (letter per position)",
                     y=1.0, fontsize=14)
        fig.tight_layout(rect=[0.06, 0, 1, 0.985])
        out = FIG_DIR / f"fig_iedb8_logos_{cond}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
