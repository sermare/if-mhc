#!/usr/bin/env python3
"""Total designs vs. unique peptides, all 4 models x 2 temperatures, both structures.

Bullet-bar form: light fill = total designs generated, dark overlay = unique peptides among
them, black text = exact unique count. One hue per model (fixed categorical order), light/dark
variants of that hue encode the total/unique magnitude split within each bar.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = "/home/ubuntu/if-mhc"

# (structure, native_peptide, temp) -> {model: fasta_path}
SOURCES = {
    ("3HG1", 0.1): {
        "vanilla": f"{ROOT}/outputs/mpnn_3hg1_100k/archive_T01_partial/vanilla_3HG1_T01_partial_26993.fa",
        "noMHC": f"{ROOT}/outputs/mpnn_3hg1_100k/archive_T01_partial/nomhc_3HG1_T01_partial_26528.fa",
        "ESM-IF1": f"{ROOT}/outputs/esmif_3hg1_pilot/seqs/3HG1.fa",
        "LigandMPNN": f"{ROOT}/outputs/ligandmpnn_3hg1_pilot/seqs/3HG1.fa",
    },
    ("3HG1", 0.3): {
        "vanilla": f"{ROOT}/outputs/mpnn_3hg1_T03_50k/run_vanilla/seqs/3HG1.fa",
        "noMHC": f"{ROOT}/outputs/mpnn_3hg1_T03_50k/run_nomhc/seqs/3HG1.fa",
        "ESM-IF1": f"{ROOT}/outputs/esmif_3hg1_T03_20k/seqs/3HG1.fa",
        "LigandMPNN": f"{ROOT}/outputs/ligandmpnn_3hg1_T03_20k/seqs/3HG1.fa",
    },
    ("2P5E", 0.1): {
        "vanilla": f"{ROOT}/outputs/mpnn_2p5e_T01_20k/seqs/vanilla_2P5E.fa",
        "noMHC": f"{ROOT}/outputs/mpnn_2p5e_T01_20k/seqs/nomhc_2P5E.fa",
        "ESM-IF1": f"{ROOT}/outputs/esmif_2p5e_pilot/seqs/2P5E.fa",
        "LigandMPNN": f"{ROOT}/outputs/ligandmpnn_2p5e_pilot/seqs/2P5E.fa",
    },
    ("2P5E", 0.3): {
        "vanilla": f"{ROOT}/outputs/mpnn_2p5e_T03_20k/seqs/vanilla_2P5E.fa",
        "noMHC": f"{ROOT}/outputs/mpnn_2p5e_T03_20k/seqs/nomhc_2P5E.fa",
        "ESM-IF1": f"{ROOT}/outputs/esmif_2p5e_T03_20k/seqs/2P5E.fa",
        "LigandMPNN": f"{ROOT}/outputs/ligandmpnn_2p5e_T03_20k/seqs/2P5E.fa",
    },
}

NATIVE = {"3HG1": "ELAGIGILTV", "2P5E": "SLLMWITQC"}
STRUCT_LABEL = {"3HG1": "3HG1 / MEL5", "2P5E": "2P5E / NY-ESO-1"}
MODELS = ["vanilla", "noMHC", "ESM-IF1", "LigandMPNN"]
# Okabe-Ito colorblind-safe categorical set, fixed order (one hue per model, never cycled/reused)
HUE = {"vanilla": "#0072B2", "noMHC": "#E69F00", "ESM-IF1": "#009E73", "LigandMPNN": "#CC79A7"}


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


def totals_and_uniques():
    data = {}
    for (struct, temp), models in SOURCES.items():
        for model, path in models.items():
            peps = load_peptides(path, model)
            data[(struct, temp, model)] = (len(peps), len(set(peps)))
    return data


def main():
    data = totals_and_uniques()
    structures = ["3HG1", "2P5E"]
    temps = [0.1, 0.3]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharey=False)

    for ax, struct in zip(axes, structures):
        xpos, xlabels = [], []
        x = 0
        for temp in temps:
            for model in MODELS:
                total, unique = data[(struct, temp, model)]
                hue = HUE[model]
                # light fill = total
                ax.bar(x, total, width=0.7, color=hue, alpha=0.30, edgecolor=hue, linewidth=0.8)
                # dark overlay = unique, anchored at the same baseline
                ax.bar(x, unique, width=0.7, color=hue, alpha=1.0)
                # black annotation with the exact unique count, just above the dark segment
                label_y = max(unique, total * 0.02) + total * 0.015
                ax.text(x, label_y, f"{unique:,}", ha="center", va="bottom",
                        fontsize=8.5, color="black")
                xpos.append(x)
                xlabels.append(f"{model}\nT={temp}")
                x += 1
            x += 0.8  # gap between temperature groups

        ax.set_xticks(xpos)
        ax.set_xticklabels(xlabels, fontsize=8)
        ax.set_ylabel("designs")
        ax.set_title(f"{STRUCT_LABEL[struct]}\nnative peptide: {NATIVE[struct]}", fontsize=11)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    legend_handles = [mpatches.Patch(color=HUE[m], label=m) for m in MODELS]
    legend_handles.append(mpatches.Patch(facecolor="gray", alpha=0.30, edgecolor="gray",
                                          label="total designs (light)"))
    legend_handles.append(mpatches.Patch(facecolor="gray", alpha=1.0, label="unique peptides (dark)"))
    fig.legend(handles=legend_handles, loc="upper center", ncol=6, fontsize=8.5,
               bbox_to_anchor=(0.5, 1.04), frameon=False)
    fig.suptitle("Total designs vs. unique peptides, all 4 models x T=0.1/0.3", y=1.13, fontsize=13)
    fig.tight_layout()
    out = f"{ROOT}/figures/fig_if10_total_vs_unique/fig_if10_total_vs_unique.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"wrote {out}")
    for k, v in sorted(data.items()):
        print(k, v)


if __name__ == "__main__":
    main()
