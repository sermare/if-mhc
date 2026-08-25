#!/usr/bin/env python3
"""Per-position version of the diversity-vs-recovery relationship: for every (crystal, model,
condition, position), recovery of the native residue AT that position vs. the number of distinct
amino acids actually used AT that position (and Shannon entropy, which also captures how evenly
spread those amino acids are -- unique count alone can't distinguish "5 AAs, one at 95%" from
"5 AAs, evenly split")."""
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
MODEL_COLOR = {"vanilla": "#0072B2", "noMHC": "#E69F00", "ESM-IF1": "#009E73", "LigandMPNN": "#CC79A7"}
MODEL_LABEL = {"vanilla": "ProteinMPNN", "noMHC": "noMHC ProteinMPNN (No MHC)", "ESM-IF1": "ESM-IF1",
               "LigandMPNN": "LigandMPNN"}
STRUCTS = ["2P5W", "1QSF", "1QRN", "2BNR", "2GJ6", "2F53", "2F54", "3QDG", "3QEQ", "3QFJ", "3GSN",
           "1OGA", "3UTS", "5C0A", "5C0B", "5HHO", "5EU6", "2VLR", "4MJI", "5NME"]
AA = list("ACDEFGHIKLMNPQRSTVWY")

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


def shannon_entropy(counts):
    counts = np.array(counts, dtype=float)
    counts = counts[counts > 0]
    p = counts / counts.sum()
    return -(p * np.log2(p)).sum()


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
                for pos in range(length):
                    col = [p[pos] for p in peps]
                    counts = pd.Series(col).value_counts()
                    n_unique_aa = counts.shape[0]
                    entropy = shannon_entropy(counts.values)
                    recovery = sum(c == native[pos] for c in col) / len(col)
                    records.append({"pdb": pdb, "condition": cond, "model": model, "position": pos + 1,
                                    "recovery": recovery, "n_unique_aa": n_unique_aa, "entropy": entropy})
        print(f"{cond}: done", flush=True)

    df = pd.DataFrame(records)
    df.to_csv(ROOT / "outputs/analysis/per_position_diversity_data.csv", index=False)

    r_unique, p_unique = pearsonr(df["recovery"], df["n_unique_aa"])
    r_ent, p_ent = pearsonr(df["recovery"], df["entropy"])
    print(f"\nrecovery vs n_unique_aa: r={r_unique:.3f} (p={p_unique:.2e})")
    print(f"recovery vs entropy:    r={r_ent:.3f} (p={p_ent:.2e})")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, ycol, ylabel, r, p in [
        (axes[0], "n_unique_aa", "number of distinct amino acids used at this position", r_unique, p_unique),
        (axes[1], "entropy", "Shannon entropy at this position (bits)", r_ent, p_ent),
    ]:
        for model in MODELS:
            sub = df[df.model == model]
            ax.scatter(sub["recovery"], sub[ycol], s=14, alpha=0.5, color=MODEL_COLOR[model],
                       label=MODEL_LABEL[model])
        ax.set_xlabel("recovery of native residue at this position")
        ax.set_ylabel(ylabel)
        ax.set_title(f"r={r:.2f} (p={p:.1e}), n={len(df)}")
        ax.legend(fontsize=8)
    fig.suptitle("Per-position: recovery vs. amino-acid diversity at that SAME position\n"
                 f"(all {len(STRUCTS)} crystals x 4 models x 2 conditions x up to 10 positions)",
                 y=1.03, fontsize=13)
    fig.tight_layout()
    out = FIG_DIR / "fig_iedb6_per_position_diversity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
