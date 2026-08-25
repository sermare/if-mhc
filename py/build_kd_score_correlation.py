#!/usr/bin/env python3
"""ProteinMPNN score_only (vanilla + noMHC) on the 51 real KD-tested NY-ESO-1/1G4c58c61 peptides,
scored against the 2P5E full-context (MHC+TCR) structure, correlated against measured KD.

score_only gives the model's mean per-residue negative log-likelihood for a GIVEN sequence on the
GIVEN structure (lower = model considers it more favorable/native-like) -- this is genuinely
different from the generation runs (which sample new sequences); here we're scoring fixed,
externally-measured peptides.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

ROOT = Path("/home/ubuntu/if-mhc")
NATIVE = "SLLMWITQC"


def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))


def load_scores(weights):
    scdir = ROOT / f"outputs/kd_scoring/{weights}/score_only"
    rows = []
    for f in sorted(scdir.glob("2P5E_fasta_*.npz")):
        d = np.load(f)
        rows.append({"peptide": str(d["seq_str"]), "score": float(d["score"][0]),
                     "global_score": float(d["global_score"][0])})
    return pd.DataFrame(rows)


def main():
    kin = pd.read_csv("/tmp/kd_peptides.csv")

    def parse_kd(val):
        if pd.isna(val) or val == "N.B.":
            return (val != "N.B."), np.nan
        try:
            return True, float(val)
        except ValueError:
            return None, np.nan

    parsed = kin["KD_raw"].apply(parse_kd)
    kin["is_binder"] = parsed.apply(lambda t: t[0])
    kin["kd_value"] = parsed.apply(lambda t: t[1])
    kin["hamming_to_native"] = kin["Peptide"].apply(lambda p: hamming(p, NATIVE))

    vanilla = load_scores("vanilla").rename(columns={"score": "score_vanilla"})
    nomhc = load_scores("nomhc").rename(columns={"score": "score_nomhc"})
    merged = kin.merge(vanilla[["peptide", "score_vanilla"]], left_on="Peptide", right_on="peptide") \
                .merge(nomhc[["peptide", "score_nomhc"]], left_on="Peptide", right_on="peptide")

    merged.to_csv(ROOT / "outputs/analysis/kd_score_correlation.csv", index=False)

    binders = merged[merged["is_binder"] == True].copy()
    binders["pKD"] = -np.log10(binders["kd_value"])

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for ax, score_col, label in [(axes[0], "score_vanilla", "vanilla"), (axes[1], "score_nomhc", "noMHC")]:
        sc = ax.scatter(binders[score_col], binders["pKD"], c=binders["hamming_to_native"],
                         cmap="viridis_r", s=140, edgecolors="black", linewidths=0.8, zorder=5)
        for _, row in binders.iterrows():
            ax.annotate(row["Peptide"], (row[score_col], row["pKD"]), fontsize=8,
                        xytext=(5, 5), textcoords="offset points")
        # show non-binders too, at a fixed low pKD row for visual context (not part of correlation)
        nb = merged[merged["is_binder"] == False]
        ax.scatter(nb[score_col], np.full(len(nb), binders["pKD"].min() - 0.5), marker="x",
                   color="dimgray", s=40, alpha=0.5, label="confirmed N.B. (no KD; placed for reference)")
        r, p = pearsonr(binders[score_col], binders["pKD"])
        rho, ps = spearmanr(binders[score_col], binders["pKD"])
        ax.set_title(f"{label}: Pearson r={r:.2f} (p={p:.2f}), Spearman rho={rho:.2f} (p={ps:.2f}), n={len(binders)}",
                     fontsize=10)
        ax.set_xlabel(f"ProteinMPNN score ({label}), lower = more favorable")
        ax.set_ylabel("pKD = -log10(KD, M)  [higher = stronger binder]")
        ax.legend(fontsize=7, loc="best")
        fig.colorbar(sc, ax=ax, fraction=0.046, label="Hamming distance to native (SLLMWITQC)")
    fig.suptitle("ProteinMPNN score vs. measured KD, real NY-ESO-1/1G4c58c61 peptides (2P5E full context)",
                 y=1.02, fontsize=13)
    fig.tight_layout()
    Path(ROOT / "figures/fig_if14_kd_score_correlation").mkdir(exist_ok=True, parents=True)
    out = ROOT / "figures/fig_if14_kd_score_correlation/fig_if14_kd_score_correlation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
    print(binders[["Peptide", "kd_value", "pKD", "score_vanilla", "score_nomhc", "hamming_to_native"]]
          .sort_values("pKD", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
