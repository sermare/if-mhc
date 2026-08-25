#!/usr/bin/env python3
"""What drives unique-peptide diversity across the panel? Two tests:
1. Do models agree on WHICH crystals are more/less diverse (correlate unique-counts across models,
   same mechanic as fig_iedb4's recovery-agreement matrix) -- a shared structural driver would show
   up as positive correlation; if diversity is basically idiosyncratic per model, correlations
   should be weak/near-zero.
2. Is diversity mechanistically the flip side of recovery -- crystals/models with more "locked"
   (highly-recovered, low-entropy) positions should have fewer unique peptides, since the model is
   confidently repeating the same residues at most positions.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

ROOT = Path("/home/ubuntu/if-mhc")
FIG_DIR = ROOT / "figures/fig_iedb6_diversity_drivers"
FIG_DIR.mkdir(exist_ok=True, parents=True)
MODELS = ["vanilla", "noMHC", "ESM-IF1", "LigandMPNN"]
MODEL_LABEL = {"vanilla": "ProteinMPNN", "noMHC": "noMHC ProteinMPNN (No MHC)", "ESM-IF1": "ESM-IF1",
               "LigandMPNN": "LigandMPNN"}
STRUCTS = ["2P5W", "1QSF", "1QRN", "2BNR", "2GJ6", "2F53", "2F54", "3QDG", "3QEQ", "3QFJ", "3GSN",
           "1OGA", "3UTS", "5C0A", "5C0B", "5HHO", "5EU6", "2VLR", "4MJI", "5NME",
           "1BD2", "1LP9", "1MI5", "1QSE", "2AK4", "2BNQ", "2E7L", "2J8U", "2JCC", "2OI9",
           "2PYE", "2UWE", "3C60", "3D3V", "3H9S", "3PWP", "3QDJ", "3QIB", "4FTV", "4JFD",
           "4JFE", "4JFF", "4L3E", "4MNQ", "4OZG", "4P23", "4P5T", "5E9D", "6AM5", "6AMU"]
# same uniform sample size the notebooks use, so unique counts are comparable across cells
N_DESIGNS = 9984

dataset = pd.read_csv(ROOT / "inputs/pmhc_tcr_dataset/dataset.csv")


def peptide_from_ligandmpnn_line(line):
    return line.strip().split(":")[2]


def load_designs(pdb, cond):
    rows = []
    for weights, fname in [("vanilla", f"vanilla_{pdb}.fa"), ("noMHC", f"nomhc_{pdb}.fa")]:
        path = ROOT / f"outputs/panel/{pdb}/{cond}/mpnn/seqs/{fname}"
        lines = path.read_text().splitlines() if path.exists() else []
        lines = lines[:2 * N_DESIGNS + 0]
        for i in range(0, len(lines) - 1, 2):
            if lines[i].startswith(">"):
                rows.append({"peptide": lines[i + 1].strip(), "model": weights})
    path = ROOT / f"outputs/panel/{pdb}/{cond}/esmif/seqs/{pdb}.fa"
    lines = path.read_text().splitlines() if path.exists() else []
    lines = lines[:2 * N_DESIGNS + 0]
    for i in range(0, len(lines) - 1, 2):
        if lines[i].startswith(">"):
            rows.append({"peptide": lines[i + 1].strip(), "model": "ESM-IF1"})
    path = ROOT / f"outputs/panel/{pdb}/{cond}/ligandmpnn/seqs/{pdb}.fa"
    lines = path.read_text().splitlines() if path.exists() else []
    lines = lines[:2 * N_DESIGNS + 0]
    for i in range(0, len(lines) - 1, 2):
        if lines[i].startswith(">"):
            rows.append({"peptide": peptide_from_ligandmpnn_line(lines[i + 1]), "model": "LigandMPNN"})
    return pd.DataFrame(rows)


def main():
    records = []
    for cond in ["mhconly", "full"]:
        for pdb in STRUCTS:
            native = dataset.loc[dataset.pdb == pdb, "peptide"].iloc[0]
            length = len(native)
            df = load_designs(pdb, cond)
            for model in MODELS:
                peps = [p for p in df.loc[df.model == model, "peptide"] if len(p) == length]
                if not peps:
                    continue
                total = len(peps)
                unique = len(set(peps))
                # per-position recovery for THIS (pdb, cond, model)
                hits = np.zeros(length)
                for p in peps:
                    for pos in range(length):
                        if p[pos] == native[pos]:
                            hits[pos] += 1
                recovery = hits / total
                mean_recovery = recovery.mean()
                n_locked = (recovery > 0.9).sum()  # positions essentially fixed to native
                records.append({"pdb": pdb, "condition": cond, "model": model, "total": total,
                                "unique": unique, "unique_frac": unique / total,
                                "mean_recovery": mean_recovery, "n_locked": n_locked, "length": length})

    df = pd.DataFrame(records)
    df.to_csv(ROOT / "outputs/analysis/diversity_drivers_data.csv", index=False)

    # --- Test 1: do models agree on which crystals are diverse? ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, cond in zip(axes, ["mhconly", "full"]):
        sub = df[df.condition == cond]
        pivot = sub.pivot(index="pdb", columns="model", values="unique_frac")
        corr = pivot.corr()
        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
        model_labels = [MODEL_LABEL[m] for m in MODELS]
        ax.set_xticks(range(len(MODELS))); ax.set_xticklabels(model_labels, rotation=30, ha="right")
        ax.set_yticks(range(len(MODELS))); ax.set_yticklabels(model_labels)
        for i in range(len(MODELS)):
            for j in range(len(MODELS)):
                v = corr.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if abs(v) > 0.6 else "black", fontsize=10)
        ax.set_title(f"Do models agree on WHICH crystals are diverse? -- {cond}\n"
                     f"(correlation of unique-fraction across {len(STRUCTS)} crystals)")
        fig.colorbar(im, ax=ax, fraction=0.046, label="Pearson r")
    fig.tight_layout()
    out1 = FIG_DIR / "fig_iedb6_diversity_model_agreement.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"wrote {out1}")

    # --- Test 2: is diversity just the flip side of recovery? ---
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    colors = {"mhconly": "#4C72B0", "full": "#C44E52"}
    for cond in ["mhconly", "full"]:
        sub = df[df.condition == cond]
        ax.scatter(sub["mean_recovery"], np.log10(sub["unique"]), s=30, alpha=0.6,
                  color=colors[cond], label=cond)
    r, p = pearsonr(df["mean_recovery"], np.log10(df["unique"]))
    ax.set_xlabel("mean recovery across positions")
    ax.set_ylabel("log10(unique peptide count)")
    ax.set_title(f"Diversity vs. recovery, all crystals x models x conditions\n"
                 f"(Pearson r={r:.2f} on log10(unique) vs mean recovery, n={len(df)})")
    ax.legend()
    fig.tight_layout()
    out2 = FIG_DIR / "fig_iedb6_diversity_vs_recovery.png"
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"wrote {out2}")

    print(f"\noverall Pearson r (mean_recovery vs log10(unique)): {r:.3f} (p={p:.2e})")
    print("\nper-model summary (unique_frac, mean_recovery):")
    print(df.groupby("model")[["unique_frac", "mean_recovery"]].mean())
    print("\nper-crystal summary (unique_frac averaged across models):")
    print(df.groupby("pdb")["unique_frac"].mean().sort_values())


if __name__ == "__main__":
    main()
