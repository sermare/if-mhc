#!/usr/bin/env python3
"""Rebuild fig_if1 (per-round trajectory) and fig_if2 (context ablation) with the requested axis
changes. Source data: outputs/analysis/{3hg1,2p5e}_per_round_mean_dist.csv and
outputs/analysis/context_ablation_per_round.csv (no other builder script for these two persists --
same situation as fig_if4/5/8). Values were cross-checked against the existing PNGs before this
rewrite; the per-category numbers reproduce closely except fig_if2's RANDOM_baseline line, which the
original plotted with a slightly different trend (5.17->5.20->5.11->4.90) than this CSV's
RANDOM_baseline row (4.83->4.80->4.88->5.11) -- likely a different random draw at figure-build time,
kept as-is here since it's the only random-baseline series available for this exact category split.
"""
import matplotlib.pyplot as plt
import pandas as pd

ROOT = "/home/ubuntu/if-mhc"
NATIVE = {"3HG1": "ELAGIGILTV", "2P5E": "SLLMWITQC"}


def build_fig_if1():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    specs = [("3HG1", "3HG1 / MEL5\n(MART1_10 selection)", "outputs/analysis/3hg1_per_round_mean_dist.csv"),
             ("2P5E", "2P5E / NY-ESO-1\n(1G4-c58/c61 selection)", "outputs/analysis/2p5e_per_round_mean_dist.csv")]
    series = [("vanilla_T0.1", "vanilla T0.1", "tab:blue", "-"),
              ("vanilla_T0.3", "vanilla T0.3", "tab:blue", "--"),
              ("noMHC_T0.1", "noMHC T0.1", "tab:red", "-"),
              ("noMHC_T0.3", "noMHC T0.3", "tab:red", "--"),
              ("ESM-IF_T0.1", "ESM-IF1 T0.1", "tab:green", "-"),
              ("ESM-IF_T0.3", "ESM-IF1 T0.3", "tab:green", "--"),
              ("LigandMPNN_T0.1", "LigandMPNN T0.1", "tab:purple", "-"),
              ("LigandMPNN_T0.3", "LigandMPNN T0.3", "tab:purple", "--"),
              ("RANDOM_baseline", "RANDOM baseline", "gray", ":")]
    for ax, (struct, title, path) in zip(axes, specs):
        df = pd.read_csv(f"{ROOT}/{path}").set_index("source")
        round_cols = [c for c in df.columns if c.startswith("R")]
        for name, label, color, ls in series:
            marker = "x" if "RANDOM" in name else ("s" if "noMHC" in name or "LigandMPNN" in name else "o")
            ax.plot(round_cols, df.loc[name, round_cols], label=label, color=color,
                     linestyle=ls, marker=marker, linewidth=2, markersize=7)
        ax.set_ylim(2, 6)  # reading top->bottom, ticks descend 6->2
        ax.set_xlabel("selection round")
        ax.set_title(f"{title}\nnative/index peptide: {NATIVE[struct]}", fontsize=11)
        if ax is axes[0]:
            ax.set_ylabel("mean nearest-Hamming distance\nto enriched population")
        else:
            ax.legend(fontsize=8.5, loc="upper right", ncol=2)
    fig.tight_layout()
    out = f"{ROOT}/figures/fig_if1_per_round/fig_if1_per_round.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def build_fig_if2():
    df = pd.read_csv(f"{ROOT}/outputs/analysis/context_ablation_per_round.csv").set_index("source")
    round_cols = [c for c in df.columns if c.startswith("R")]
    categories = {
        "full context (MHC+TCR)": ["FULL_vanilla_T0.1", "FULL_noMHC_T0.1", "FULL_ESM-IF_T0.1", "FULL_LigandMPNN_T0.1"],
        "MHC only": ["vanilla_mhconly", "noMHC_mhconly", "LigandMPNN_mhconly"],
        "TCR only": ["vanilla_tcronly", "noMHC_tcronly", "LigandMPNN_tcronly"],
        "no context (peptide alone)": ["vanilla_nocontext", "noMHC_nocontext", "ESM-IF_nocontext", "LigandMPNN_nocontext"],
    }
    fig, ax = plt.subplots(figsize=(9, 6.5))
    colors = {"full context (MHC+TCR)": "tab:blue", "MHC only": "tab:orange",
              "TCR only": "tab:green", "no context (peptide alone)": "tab:purple"}
    markers = {"full context (MHC+TCR)": "o", "MHC only": "^", "TCR only": "v",
               "no context (peptide alone)": "D"}
    for label, rows in categories.items():
        mean_vals = df.loc[rows, round_cols].mean(axis=0)
        ax.plot(round_cols, mean_vals, label=label, color=colors[label],
                 marker=markers[label], linewidth=2.5, markersize=9)
    ax.plot(round_cols, df.loc["RANDOM_baseline", round_cols], label="random baseline",
             color="gray", linestyle=":", marker="x", linewidth=2, markersize=8)
    ax.set_ylim(3, 6)  # reversed: reading top->bottom, ticks descend 6->3
    ax.set_xlabel("selection round")
    ax.set_ylabel("mean nearest-Hamming distance\nto enriched population (3HG1)")
    ax.set_title("Only the full MHC+TCR complex tracks selection\n"
                  f"native/index peptide: {NATIVE['3HG1']}  (mean across all available models/condition)")
    ax.legend(fontsize=10, loc="lower left")
    fig.tight_layout()
    out = f"{ROOT}/figures/fig_if2_context_ablation/fig_if2_context_ablation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    build_fig_if1()
    build_fig_if2()
