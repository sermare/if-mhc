#!/usr/bin/env python3
"""Regenerate fig_if4 (logos), fig_if5 (confusion), fig_if8 (model-compare JS divergence) with the
native/index peptide added to every panel title. Run under esmcba env (needs pyarrow for the 3HG1
NGS parquet read via ong229_ranking_lib).

fig_if4 matches the original convention exactly (T=0.3, deduplicated/unique designs; NGS = terminal
round, nonzero count) -- verified: unique counts here (2459/554/1624/973 for 3HG1, 2001/672/478/1386
for 2P5E) match the original panel n's to within 1. fig_if5/fig_if8 are freshly computed at T=0.1 on
RAW (non-deduplicated) designs, count-weighted for NGS -- the original script no longer exists, so
these are clearly-labeled fresh builds rather than bit-for-bit reproductions.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logomaker
from scipy.spatial.distance import jensenshannon

ROOT = "/home/ubuntu/if-mhc"
sys.path.append("/home/ubuntu/pmhc/modeling/ONG229/py")

AA = list("ACDEFGHIKLMNPQRSTVWY")
NATIVE = {"3HG1": "ELAGIGILTV", "2P5E": "SLLMWITQC"}
STRUCT_LABEL = {"3HG1": "3HG1 / MEL5", "2P5E": "2P5E / NY-ESO-1"}
MODELS = ["vanilla", "noMHC", "ESM-IF1", "LigandMPNN"]

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


def load_ngs(struct):
    if struct == "3HG1":
        import ong229_ranking_lib as lib
        ngs = lib.load_tab_data()["MART1_10mer__CAB60174_G01"]
        col = "R3"
    else:
        ngs = pd.read_csv("/home/ubuntu/pmhc/modeling/work/full_5round/ONG229_1G4c58c61_peptide_counts.csv")
        col = "R4"
    terminal = ngs[ngs[col] > 0][["Peptide", col]].rename(columns={col: "count"})
    return terminal, col


def counts_matrix(peptides_with_weights, length):
    """peptides_with_weights: list of (seq, weight). Returns position x AA count matrix."""
    mat = pd.DataFrame(0.0, index=range(1, length + 1), columns=AA)
    for seq, w in peptides_with_weights:
        if len(seq) != length:
            continue
        for pos, aa in enumerate(seq, start=1):
            if aa in AA:
                mat.loc[pos, aa] += w
    return mat


def build_fig_if4():
    fig, axes = plt.subplots(2, 5, figsize=(20, 11))
    fig.subplots_adjust(hspace=0.75, top=0.90)
    for row, struct in enumerate(["3HG1", "2P5E"]):
        native = NATIVE[struct]
        length = len(native)
        ngs_terminal, round_col = load_ngs(struct)
        ngs_peps = [(p, c) for p, c in zip(ngs_terminal["Peptide"], ngs_terminal["count"]) if len(p) == length]
        ax = axes[row, 0]
        mat = counts_matrix(ngs_peps, length)
        logomaker.Logo(mat, ax=ax, color_scheme="chemistry")
        ax.set_title(f"real NGS ({round_col})\nn={len(ngs_peps):,}", fontsize=10)
        ax.set_ylabel("bits" if row == 0 else "")

        for col, model in enumerate(MODELS, start=1):
            peps = load_peptides(SOURCES[(struct, 0.3)][model], model)
            uniq = sorted(set(p for p in peps if len(p) == length))
            ax = axes[row, col]
            mat = counts_matrix([(p, 1.0) for p in uniq], length)
            logomaker.Logo(mat, ax=ax, color_scheme="chemistry")
            ax.set_title(f"{model}\nn={len(uniq):,}", fontsize=10)
    for row, struct in enumerate(["3HG1", "2P5E"]):
        top_y = axes[row, 0].get_position().y1
        fig.text(0.5, top_y + 0.075, f"{STRUCT_LABEL[struct]} (T=0.3 unique designs)  --  "
                  f"native/index peptide: {NATIVE[struct]}",
                  ha="center", fontsize=13, transform=fig.transFigure)
    out = f"{ROOT}/figures/fig_if4_logos/fig_if4_logos_combined.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def build_fig_if5():
    fig, axes = plt.subplots(2, 4, figsize=(18, 11))
    fig.subplots_adjust(hspace=0.7, top=0.90)
    for row, struct in enumerate(["3HG1", "2P5E"]):
        native = NATIVE[struct]
        length = len(native)
        for col, model in enumerate(MODELS):
            peps = load_peptides(SOURCES[(struct, 0.1)][model], model)
            peps = [p for p in peps if len(p) == length]
            conf = pd.DataFrame(0.0, index=list(native), columns=AA)
            for p in peps:
                for pos, (native_aa, designed_aa) in enumerate(zip(native, p)):
                    if designed_aa in AA:
                        conf.loc[native_aa, designed_aa] += 1
            conf_norm = conf.div(conf.sum(axis=1).replace(0, np.nan), axis=0)
            ax = axes[row, col]
            im = ax.imshow(conf_norm.values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(len(AA))); ax.set_xticklabels(AA, fontsize=7)
            ax.set_yticks(range(len(native))); ax.set_yticklabels(list(native), fontsize=7)
            ax.set_title(model, fontsize=10)
            ax.set_xlabel("designed AA", fontsize=8)
            if col == 0:
                ax.set_ylabel("native AA (by position)", fontsize=8)
        fig.colorbar(im, ax=axes[row, :].tolist(), fraction=0.02, pad=0.01,
                     label="P(designed AA | native AA)")
    for row, struct in enumerate(["3HG1", "2P5E"]):
        top_y = axes[row, 0].get_position().y1
        fig.text(0.42, top_y + 0.075, f"{STRUCT_LABEL[struct]} -- per-position confusion "
                  f"(T=0.1, row-normalized)  --  native/index peptide: {NATIVE[struct]}",
                  ha="center", fontsize=12, transform=fig.transFigure)
    out = f"{ROOT}/figures/fig_if5_confusion/fig_if5_confusion_combined.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def aa_freq_vector(peps_with_weights, length):
    mat = counts_matrix(peps_with_weights, length)
    total = mat.values.sum()
    return mat.values / total if total > 0 else mat.values


def build_fig_if8(temp=0.1):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for ax, struct in zip(axes, ["3HG1", "2P5E"]):
        native = NATIVE[struct]
        length = len(native)
        sources = {}
        for model in MODELS:
            peps = load_peptides(SOURCES[(struct, temp)][model], model)
            sources[model] = [(p, 1.0) for p in peps if len(p) == length]
        ngs_terminal, round_col = load_ngs(struct)
        sources[f"real NGS ({round_col})"] = [
            (p, c) for p, c in zip(ngs_terminal["Peptide"], ngs_terminal["count"]) if len(p) == length
        ]
        names = list(sources.keys())
        vecs = {n: aa_freq_vector(sources[n], length) for n in names}
        n = len(names)
        js = np.zeros((n, n))
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                per_pos = [jensenshannon(vecs[a][p], vecs[b][p], base=2) for p in range(length)]
                js[i, j] = np.nanmean(per_pos)
        im = ax.imshow(js, cmap="inferno", vmin=0, vmax=js.max())
        ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=8)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{js[i, j]:.2f}", ha="center", va="center",
                        color="white" if js[i, j] < js.max() * 0.6 else "black", fontsize=8)
        ax.set_title(f"{STRUCT_LABEL[struct]}\nnative/index peptide: {native}", fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mean per-position JS divergence")
    fig.suptitle(f"Mean per-position Jensen-Shannon divergence between design sources (T={temp}, raw designs)",
                 fontsize=12)
    fig.tight_layout()
    suffix = "" if temp == 0.1 else f"_T{str(temp).replace('0.', '')}"
    out = f"{ROOT}/figures/fig_if8_modelcompare/fig_if8_modelcompare{suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    import sys as _sys
    targets = _sys.argv[1:] or ["4", "5", "8"]
    if "4" in targets:
        build_fig_if4()
    if "5" in targets:
        build_fig_if5()
    if "8" in targets:
        build_fig_if8(0.1)
    if "8T3" in targets:
        build_fig_if8(0.3)
