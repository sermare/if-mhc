#!/usr/bin/env python3
"""Rebuild fig_if6 (sequence-space PCA). Row order: (1) colored by round, (2) colored by NGS median
count, (3) colored by NGS average count -- both count stats computed over nonzero rounds per unique
peptide. Same PCA axes reused across all three rows per structure (fit once on the round-labeled
combined sample, applied to both the per-appearance round row and the deduplicated unique-peptide
rows) so the maps are directly comparable, not independently-scaled projections. The "colored by
generative peptide" (which model produced it) view comes last, as a separate per-structure file
(fig_if6_pca_design_overlay_{struct}.png), transposed to T=0.1/T=0.3 rows x one-column-per-model.

Round-colored row reproduces the paper's documented method (Methods S2.6): per round column, every
peptide with nonzero count in that round, capped at 30,000 (random subsample beyond that), rounds
concatenated with a round label -- so a peptide present in multiple rounds contributes one point per
round it's present in (see conversation: this was flagged as a real ambiguity, not a bug to silently
fix here). The two new rows use each unique peptide exactly once, sidestepping that ambiguity by
construction.
"""
import sys
sys.path.append("/home/ubuntu/pmhc/modeling/ONG229/py")
import ong229_ranking_lib as lib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

ROOT = "/home/ubuntu/if-mhc/"
NATIVE = {"3HG1": "ELAGIGILTV", "2P5E": "SLLMWITQC"}
STRUCT_LABEL = {"3HG1": "3HG1 / MEL5", "2P5E": "2P5E / NY-ESO-1"}
AA = list("ACDEFGHIKLMNPQRSTVWY")
AA_IDX = {a: i for i, a in enumerate(AA)}
CAP = 30000
RNG = np.random.RandomState(0)
MODELS = ["vanilla", "noMHC", "ESM-IF1", "LigandMPNN"]
MODEL_COLOR = {"vanilla": "#4C72B0", "noMHC": "#DD8452", "ESM-IF1": "#55A868", "LigandMPNN": "#C44E52"}

DESIGN_SOURCES = {
    ("3HG1", 0.1): {
        "vanilla": f"{ROOT}outputs/mpnn_3hg1_100k/archive_T01_partial/vanilla_3HG1_T01_partial_26993.fa",
        "noMHC": f"{ROOT}outputs/mpnn_3hg1_100k/archive_T01_partial/nomhc_3HG1_T01_partial_26528.fa",
        "ESM-IF1": f"{ROOT}outputs/esmif_3hg1_pilot/seqs/3HG1.fa",
        "LigandMPNN": f"{ROOT}outputs/ligandmpnn_3hg1_pilot/seqs/3HG1.fa",
    },
    ("2P5E", 0.1): {
        "vanilla": f"{ROOT}outputs/mpnn_2p5e_T01_20k/seqs/vanilla_2P5E.fa",
        "noMHC": f"{ROOT}outputs/mpnn_2p5e_T01_20k/seqs/nomhc_2P5E.fa",
        "ESM-IF1": f"{ROOT}outputs/esmif_2p5e_pilot/seqs/2P5E.fa",
        "LigandMPNN": f"{ROOT}outputs/ligandmpnn_2p5e_pilot/seqs/2P5E.fa",
    },
    ("3HG1", 0.3): {
        "vanilla": f"{ROOT}outputs/mpnn_3hg1_T03_50k/run_vanilla/seqs/3HG1.fa",
        "noMHC": f"{ROOT}outputs/mpnn_3hg1_T03_50k/run_nomhc/seqs/3HG1.fa",
        "ESM-IF1": f"{ROOT}outputs/esmif_3hg1_T03_20k/seqs/3HG1.fa",
        "LigandMPNN": f"{ROOT}outputs/ligandmpnn_3hg1_T03_20k/seqs/3HG1.fa",
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


def load_design_peptides(path, tool):
    seqs = []
    with open(path) as f:
        lines = f.read().splitlines()
    for i in range(0, len(lines) - 1, 2):
        if not lines[i].startswith(">"):
            continue
        seqs.append(peptide_from_ligandmpnn_line(lines[i + 1]) if tool == "LigandMPNN" else lines[i + 1].strip())
    return seqs


def one_hot(seqs, length):
    X = np.zeros((len(seqs), length * len(AA)), dtype=np.float32)
    for i, s in enumerate(seqs):
        for pos, aa in enumerate(s):
            j = AA_IDX.get(aa)
            if j is not None:
                X[i, pos * len(AA) + j] = 1.0
    return X


def build_round_labeled_sample(ngs_df, round_cols, length):
    frames = []
    for col in round_cols:
        sub = ngs_df[(ngs_df[col] > 0) & (ngs_df["Peptide"].str.len() == length)]
        if len(sub) > CAP:
            sub = sub.sample(CAP, random_state=RNG)
        frames.append(pd.DataFrame({"Peptide": sub["Peptide"].values, "round": col}))
    return pd.concat(frames, ignore_index=True)


def main():
    specs = [
        ("3HG1", "MART1_10mer__CAB60174_G01", ["R0", "R1", "R2", "R3"], None),
        ("2P5E", None, ["R0", "R1", "R2", "R3", "R4"],
         "/home/ubuntu/pmhc/modeling/work/full_5round/ONG229_1G4c58c61_peptide_counts.csv"),
    ]
    tab_data = lib.load_tab_data()

    # rows = the 2 structures, columns = round / NGS median / NGS average / Hamming-to-native / generative peptide (model)
    fig, axes = plt.subplots(2, 5, figsize=(32, 11))
    round_cmap = plt.get_cmap("viridis")

    for row_idx, (struct, tab_key, round_cols, csv_path) in enumerate(specs):
        native = NATIVE[struct]
        length = len(native)
        ngs_df = tab_data[tab_key] if tab_key else pd.read_csv(csv_path)

        labeled = build_round_labeled_sample(ngs_df, round_cols, length)
        X = one_hot(labeled["Peptide"].tolist(), length)
        pca = PCA(n_components=2, random_state=0)
        pcs = pca.fit_transform(X)
        labeled["PC1"], labeled["PC2"] = pcs[:, 0], pcs[:, 1]
        var1, var2 = pca.explained_variance_ratio_[:2] * 100
        pc1_label, pc2_label = f"PC1 ({var1:.1f}% var)", f"PC2 ({var2:.1f}% var)"

        ax = axes[row_idx, 0]
        for i, rc in enumerate(round_cols):
            m = labeled["round"] == rc
            ax.scatter(labeled.loc[m, "PC1"], labeled.loc[m, "PC2"], s=4, alpha=0.15,
                       color=round_cmap(i / max(len(round_cols) - 1, 1)), label=rc)
        ax.set_title(f"{STRUCT_LABEL[struct]} -- colored by round", fontsize=11)
        ax.legend(fontsize=8, markerscale=3, loc="best")
        ax.set_ylabel(f"native/index peptide: {native}\n{pc2_label}", fontsize=9)
        ax.set_xlabel(pc1_label)

        # deduplicated unique-peptide view, same fitted PCA space
        uniq = ngs_df[ngs_df["Peptide"].str.len() == length].drop_duplicates("Peptide").copy()
        uniq = uniq[uniq["Peptide"].isin(set(labeled["Peptide"]))]  # keep it to the same sampled universe
        # both stats computed over rounds where the peptide is actually present -- over ALL rounds
        # (including zeros) they're trivially near-0 for most peptides here, since most are only
        # nonzero in 1-2 of the round columns; that washed out all contrast in an earlier pass.
        uniq["median_count"] = uniq[round_cols].apply(
            lambda row: row[row > 0].median() if (row > 0).any() else 0.0, axis=1)
        uniq["average_count"] = uniq[round_cols].apply(
            lambda row: row[row > 0].mean() if (row > 0).any() else 0.0, axis=1)
        Xu = one_hot(uniq["Peptide"].tolist(), length)
        pcu = pca.transform(Xu)
        uniq["PC1"], uniq["PC2"] = pcu[:, 0], pcu[:, 1]

        log_med = np.log1p(uniq["median_count"])
        log_avg = np.log1p(uniq["average_count"])

        ax = axes[row_idx, 1]
        nonzero_med = log_med[log_med > 0]
        vmax = np.percentile(nonzero_med, 99) if len(nonzero_med) else 1.0
        sca = ax.scatter(uniq["PC1"], uniq["PC2"], s=5, alpha=0.6,
                          c=log_med, cmap="magma", vmin=0, vmax=vmax)
        fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.04,
                     label="log1p(median count, nonzero rounds)\n[clipped at p99 of nonzero]")
        ax.set_title(f"{STRUCT_LABEL[struct]} -- colored by NGS median count", fontsize=11)
        ax.set_xlabel(pc1_label)

        # which POSITION drives PC1 the most: reshape PC1's loading vector from (length*20,) back to
        # (length, 20) and sum the squared loadings per position -- the position with the largest
        # total contributes most to spreading points along PC1, i.e. is the best candidate for
        # explaining the discrete streaks (since PC1 is the dominant/first axis).
        pc1_loadings = pca.components_[0].reshape(length, len(AA))
        position_importance = (pc1_loadings ** 2).sum(axis=1)
        best_pos = int(position_importance.argmax())  # 0-indexed
        print(f"[{struct}] most PC1-explanatory position: {best_pos + 1} (1-indexed), "
              f"native residue there = {native[best_pos]}, "
              f"per-position PC1 loading-energy = {[f'{v:.3f}' for v in position_importance]}")

        residue_at_pos = uniq["Peptide"].str[best_pos]
        present_aas = [a for a in AA if a in set(residue_at_pos)]
        aa_color = {a: plt.get_cmap("tab20")(i / 20) for i, a in enumerate(AA)}
        ax = axes[row_idx, 2]
        for a in present_aas:
            m = residue_at_pos == a
            ax.scatter(uniq.loc[m, "PC1"], uniq.loc[m, "PC2"], s=5, alpha=0.6,
                       color=aa_color[a], label=f"{a} (n={m.sum():,})")
        ax.set_title(f"{STRUCT_LABEL[struct]} -- colored by residue at position {best_pos + 1} "
                     f"(top PC1 driver)", fontsize=11)
        ax.legend(fontsize=6.5, markerscale=1.5, loc="best", ncol=2)
        ax.set_xlabel(pc1_label)

        print(f"[{struct}] round-labeled sample rows: {len(labeled):,} (dup peptides across rounds included); "
              f"unique peptides in count columns: {len(uniq):,}")

        # column 4: Hamming distance from each real unique peptide to the native/index peptide
        uniq["hamming_to_native"] = uniq["Peptide"].apply(
            lambda p: sum(a != b for a, b in zip(p, native)))
        ax = axes[row_idx, 3]
        sca = ax.scatter(uniq["PC1"], uniq["PC2"], s=5, alpha=0.6,
                          c=uniq["hamming_to_native"], cmap="magma_r",
                          vmin=0, vmax=uniq["hamming_to_native"].max())
        fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.04,
                     label=f"Hamming distance to native ({native})")
        ax.set_title(f"{STRUCT_LABEL[struct]} -- colored by Hamming distance to native", fontsize=11)
        ax.set_xlabel(pc1_label)

        # column 5: generative peptide (which model produced it) -- real terminal-round background
        # in light gray, all 4 models' unique T=0.1 designs overlaid in one panel, same PCA space
        ax = axes[row_idx, 4]
        terminal_round = round_cols[-1]
        bg = labeled[labeled["round"] == terminal_round]
        ax.scatter(bg["PC1"], bg["PC2"], s=4, alpha=0.15, color="lightgray",
                   label=f"real NGS ({terminal_round})")
        for model in MODELS:
            peps = list(set(p for p in load_design_peptides(DESIGN_SOURCES[(struct, 0.1)][model], model)
                             if len(p) == length))
            Xd = one_hot(peps, length)
            pcd = pca.transform(Xd)
            ax.scatter(pcd[:, 0], pcd[:, 1], s=10, alpha=0.7, color=MODEL_COLOR[model],
                       label=f"{model} (n={len(peps):,})")
        ax.set_title(f"{STRUCT_LABEL[struct]} -- colored by generative peptide (T=0.1)", fontsize=11)
        ax.legend(fontsize=7, markerscale=1.5, loc="best")
        ax.set_xlabel(pc1_label)

        # new figure: one panel PER ROUND (not overlaid), each colored by that round's own actual
        # count (continuous, not just round-membership) -- same fitted PCA space reused
        fig3, axes3 = plt.subplots(1, len(round_cols), figsize=(6 * len(round_cols), 5.5), sharex=True, sharey=True)
        for i, rc in enumerate(round_cols):
            sub = ngs_df[(ngs_df[rc] > 0) & (ngs_df["Peptide"].str.len() == length)]
            if len(sub) > CAP:
                sub = sub.sample(CAP, random_state=RNG)
            Xr = one_hot(sub["Peptide"].tolist(), length)
            pcr = pca.transform(Xr)
            log_c = np.log1p(sub[rc].values)
            ax3 = axes3[i]
            vmax = np.percentile(log_c, 99)
            sca3 = ax3.scatter(pcr[:, 0], pcr[:, 1], s=5, alpha=0.6, c=log_c, cmap="magma",
                                vmin=0, vmax=vmax)
            fig3.colorbar(sca3, ax=ax3, fraction=0.046, pad=0.04, label=f"log1p({rc} count)\n[p99-clipped]")
            ax3.set_title(f"{rc} (n={len(sub):,})", fontsize=10)
            ax3.set_xlabel(pc1_label)
            if i == 0:
                ax3.set_ylabel(pc2_label)
        fig3.suptitle(f"{STRUCT_LABEL[struct]} -- one panel per round, colored by that round's own count "
                      f"(native/index peptide: {native})", fontsize=13)
        fig3.tight_layout(rect=[0, 0, 1, 0.94])
        out3 = f"{ROOT}figures/fig_if6_pca/fig_if6_pca_per_round_count_{struct}.png"
        fig3.savefig(out3, dpi=150, bbox_inches="tight")
        plt.close(fig3)
        print(f"wrote {out3}")

    fig.tight_layout()
    out = f"{ROOT}figures/fig_if6_pca/fig_if6_pca_secondfinal.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
