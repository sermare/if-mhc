#!/usr/bin/env python3
"""Where does TCR-context input actually help native-peptide recovery? Bins each panel position
into N-anchor (P1-P2), C-anchor (P-omega), or middle (TCR-facing) by relative position, then plots
recovery(full) - recovery(mhconly) per region per model. Reads the per-position/per-crystal CSVs
already written by analyze_diversity_drivers.py / analyze_per_position_diversity.py.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/ubuntu/if-mhc")
FIG_DIR = ROOT / "figures/fig_iedb9_tcr_effect_by_region"
FIG_DIR.mkdir(exist_ok=True, parents=True)
MODELS = ["vanilla", "noMHC", "ESM-IF1", "LigandMPNN"]
MODEL_COLOR = {"vanilla": "#0072B2", "noMHC": "#E69F00", "ESM-IF1": "#009E73", "LigandMPNN": "#CC79A7"}
MODEL_LABEL = {"vanilla": "ProteinMPNN", "noMHC": "noMHC ProteinMPNN (No MHC)", "ESM-IF1": "ESM-IF1",
               "LigandMPNN": "LigandMPNN"}
REGIONS = ["N-anchor\n(P1-P2)", "middle\n(TCR-facing)", "C-anchor\n(P-omega)"]


def rel_bin(row):
    frac = (row["position"] - 1) / (row["length"] - 1) if row["length"] > 1 else 0
    if frac <= 0.15:
        return "N-anchor\n(P1-P2)"
    elif frac >= 0.85:
        return "C-anchor\n(P-omega)"
    else:
        return "middle\n(TCR-facing)"


def main():
    dd = pd.read_csv(ROOT / "outputs/analysis/diversity_drivers_data.csv")
    pp = pd.read_csv(ROOT / "outputs/analysis/per_position_diversity_data.csv")
    pp["length"] = pp["pdb"].map(dict(zip(dd["pdb"], dd["length"])))
    pp["region"] = pp.apply(rel_bin, axis=1)

    tab = pp.groupby(["region", "model", "condition"])["recovery"].mean().unstack("condition")
    tab["delta"] = tab["full"] - tab["mhconly"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    width = 0.2
    x = np.arange(len(REGIONS))
    for i, model in enumerate(MODELS):
        deltas = [tab.loc[(r, model), "delta"] for r in REGIONS]
        ax.bar(x + (i - 1.5) * width, deltas, width, label=MODEL_LABEL[model], color=MODEL_COLOR[model])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(REGIONS)
    ax.set_ylabel("recovery(full) - recovery(mhconly)\n[TCR-context benefit]")
    ax.set_title("Where does TCR input actually help recovery?\n"
                 "Benefit is concentrated at TCR-facing middle positions, near-zero at the N-terminal anchor")
    ax.legend(title="model", fontsize=9)
    fig.tight_layout()
    out = FIG_DIR / "fig_iedb9_tcr_effect_by_region.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
    print(tab.round(3))


if __name__ == "__main__":
    main()
