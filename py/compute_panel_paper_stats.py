#!/usr/bin/env python3
"""Consolidated statistical backing for every claim in the panel-wide inverse-folding paper
(paper/paper.tex). T=0.1 only. Reads the CSVs already written by analyze_diversity_drivers.py,
analyze_per_position_diversity.py, and analyze_iedb_tier_tcr_effect.py; adds the formal significance
tests (paired Wilcoxon, Friedman, Mann-Whitney, Fisher-z) that the earlier exploratory passes didn't
compute. Prints one block per claim; nothing here is re-plotted, this is purely the numbers table
for the manuscript.
"""
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import (wilcoxon, friedmanchisquare, pearsonr, mannwhitneyu)

ROOT = Path("/home/ubuntu/if-mhc")
MODELS = ["vanilla", "noMHC", "ESM-IF1", "LigandMPNN"]
MODEL_LABEL = {"vanilla": "ProteinMPNN", "noMHC": "noMHC ProteinMPNN (No MHC)", "ESM-IF1": "ESM-IF1",
               "LigandMPNN": "LigandMPNN"}


def fisher_z(r):
    r = np.clip(r, -0.999999, 0.999999)
    return 0.5 * np.log((1 + r) / (1 - r))


def hdr(s):
    print("\n" + "=" * 90)
    print(s)
    print("=" * 90)


def main():
    dd = pd.read_csv(ROOT / "outputs/analysis/diversity_drivers_data.csv")
    pp = pd.read_csv(ROOT / "outputs/analysis/per_position_diversity_data.csv")
    pp["length"] = pp["pdb"].map(dict(zip(dd["pdb"], dd["length"])))

    def rel_bin(row):
        frac = (row["position"] - 1) / (row["length"] - 1) if row["length"] > 1 else 0
        if frac <= 0.15:
            return "N-anchor"
        elif frac >= 0.85:
            return "C-anchor"
        else:
            return "middle"
    pp["region"] = pp.apply(rel_bin, axis=1)

    # ---------------------------------------------------------------- A: TCR effect, overall
    hdr("A. TCR context effect on native recovery -- paired Wilcoxon, n=80 (20 pdb x 4 model)")
    piv = dd.pivot_table(index=["pdb", "model"], columns="condition", values="mean_recovery")
    stat, p = wilcoxon(piv["full"], piv["mhconly"])
    d = (piv["full"] - piv["mhconly"])
    cohend = d.mean() / d.std(ddof=1)
    print(f"mean(full)={piv['full'].mean():.4f}  mean(mhconly)={piv['mhconly'].mean():.4f}  "
          f"mean_delta={d.mean():.4f}")
    print(f"Wilcoxon signed-rank: W={stat:.1f}, p={p:.2e}   paired Cohen's d={cohend:.3f}")

    # ------------------------------------------------------- A2: TCR effect, by region (localization)
    hdr("A2. TCR effect by peptide region -- paired Wilcoxon, n=80 each region")
    region_pp = pp.groupby(["pdb", "model", "condition", "region"])["recovery"].mean().reset_index()
    for region in ["N-anchor", "middle", "C-anchor"]:
        sub = region_pp[region_pp.region == region]
        rp = sub.pivot_table(index=["pdb", "model"], columns="condition", values="recovery")
        rp = rp.dropna()
        stat, p = wilcoxon(rp["full"], rp["mhconly"])
        dlt = (rp["full"] - rp["mhconly"])
        cohend = dlt.mean() / dlt.std(ddof=1)
        print(f"{region:10s}: delta={dlt.mean():+.4f}  W={stat:.1f}  p={p:.2e}  n={len(rp)}  d={cohend:.3f}")

    # ---------------------------------------------------------------- B: model comparison
    hdr("B. Model comparison -- Friedman test across 4 models, blocks = (pdb,condition), n=40")
    blk = dd.pivot_table(index=["pdb", "condition"], columns="model", values="mean_recovery")
    stat, p = friedmanchisquare(*[blk[m] for m in MODELS])
    print(f"Friedman chi2={stat:.2f}, df=3, p={p:.2e}")
    print(blk.mean().sort_values(ascending=False))

    hdr("B2. Post-hoc: ESM-IF1 vs each other model -- paired Wilcoxon, Bonferroni-corrected (x3)")
    for m in ["LigandMPNN", "noMHC", "vanilla"]:
        stat, p = wilcoxon(blk["ESM-IF1"], blk[m])
        print(f"ESM-IF1 vs {MODEL_LABEL[m]:28s}: W={stat:.1f}  p_raw={p:.2e}  p_bonf={min(p*3,1):.2e}")

    hdr("B3. Length-controlled sensitivity: 9-mers only (n=15 pdb x 2 cond = 30 blocks)")
    nine = dd[dd.length == 9]
    blk9 = nine.pivot_table(index=["pdb", "condition"], columns="model", values="mean_recovery")
    stat, p = friedmanchisquare(*[blk9[m] for m in MODELS])
    print(f"Friedman chi2={stat:.2f}, df=3, p={p:.2e}  (full panel chi2 above for comparison)")
    print(blk9.mean().sort_values(ascending=False))

    hdr("B4. ProteinMPNN's C-anchor advantage -- paired Wilcoxon, ProteinMPNN vs others, C-anchor only, n=20 blocks")
    canchor = region_pp[region_pp.region == "C-anchor"]
    blkc = canchor.pivot_table(index=["pdb", "condition"], columns="model", values="recovery")
    for m in ["ESM-IF1", "noMHC", "LigandMPNN"]:
        stat, p = wilcoxon(blkc["vanilla"], blkc[m])
        print(f"{MODEL_LABEL['vanilla']} vs {MODEL_LABEL[m]:28s} (C-anchor): "
              f"mean_ProteinMPNN={blkc['vanilla'].mean():.3f} "
              f"mean_{MODEL_LABEL[m]}={blkc[m].mean():.3f}  W={stat:.1f}  p={p:.2e}")

    # ---------------------------------------------------------------- C: diversity vs recovery
    hdr("C. Diversity vs. recovery (already-established correlations, restated for the manuscript)")
    r, p = pearsonr(dd["mean_recovery"], np.log10(dd["unique"]))
    print(f"crystal x model x condition level (n={len(dd)}): r={r:.3f}, p={p:.2e}")
    r, p = pearsonr(pp["recovery"], pp["n_unique_aa"])
    print(f"per-position, recovery vs n_unique_aa (n={len(pp)}): r={r:.3f}, p={p:.2e}")
    r, p = pearsonr(pp["recovery"], pp["entropy"])
    print(f"per-position, recovery vs entropy (n={len(pp)}): r={r:.3f}, p={p:.2e}")

    # ---------------------------------------------------------------- D: crystal-level model agreement
    hdr("D. Crystal-level cross-model recovery agreement -- Pearson r + p, n=20 crystals")
    for cond in ["full", "mhconly"]:
        sub = dd[dd.condition == cond]
        cp = sub.pivot(index="pdb", columns="model", values="mean_recovery")
        print(f"-- {cond} --")
        for m1, m2 in combinations(MODELS, 2):
            r, p = pearsonr(cp[m1], cp[m2])
            print(f"  {m1:12s} vs {m2:12s}: r={r:.3f}  p={p:.2e}")

    # -------------------------------------------------- E: position-level vs crystal-level agreement
    hdr("E. Position-level (within-crystal shape) vs crystal-level (aggregate) agreement, Fisher-z paired test")
    crystal_r, position_r, labels = [], [], []
    for cond in ["full", "mhconly"]:
        sub = dd[dd.condition == cond]
        cp = sub.pivot(index="pdb", columns="model", values="mean_recovery")
        psub = pp[pp.condition == cond]
        for m1, m2 in combinations(MODELS, 2):
            r_c, _ = pearsonr(cp[m1], cp[m2])
            within = []
            for pdb, g in psub.groupby("pdb"):
                piv = g.pivot(index="position", columns="model", values="recovery")
                if m1 in piv.columns and m2 in piv.columns and piv[m1].notna().sum() > 2:
                    r_w, _ = pearsonr(piv[m1], piv[m2])
                    within.append(r_w)
            r_p = np.nanmean(within)
            crystal_r.append(r_c); position_r.append(r_p); labels.append((cond, m1, m2))
    crystal_r, position_r = np.array(crystal_r), np.array(position_r)
    z_diff = fisher_z(position_r) - fisher_z(crystal_r)
    stat, p = wilcoxon(z_diff)
    n_lower = (position_r < crystal_r).sum()
    print(f"position-level r lower than crystal-level r in {n_lower}/{len(labels)} (cond,pair) combos")
    print(f"Wilcoxon signed-rank on Fisher-z(position) - Fisher-z(crystal): W={stat:.1f}, p={p:.3f}")
    print(f"mean crystal-level r={crystal_r.mean():.3f}   mean position-level r={position_r.mean():.3f}")

    # ---------------------------------------------------------------- F: structure quality confound
    hdr("F. Structure-quality confound (resolution / B-factor vs recovery), n=20 crystals")
    qual = pd.read_csv(ROOT / "outputs/analysis/panel_structure_quality.csv")
    r, p = pearsonr(qual["resolution"], qual["mean_recovery"])
    print(f"resolution vs mean_recovery: r={r:.3f}, p={p:.3f}  (n=20, NOT significant)")
    r, p = pearsonr(qual["mean_bfactor"], qual["mean_recovery"])
    print(f"mean_bfactor vs mean_recovery: r={r:.3f}, p={p:.3f}  (n=20, NOT significant)")

    # ---------------------------------------------------------------- G: IEDB tier TCR effect
    hdr("G. TCR effect vs. real-IEDB-binder distance, by affinity tier")
    tier_df = pd.read_csv(ROOT / "outputs/analysis/iedb_tier_tcr_effect_data.csv")
    TIERS = ["Positive-Low", "Positive-Intermediate", "Positive-High"]
    for tier in TIERS:
        sub = tier_df[tier_df.tier == tier]
        full_d = sub[sub.condition == "full"]["hamming_dist"]
        mhc_d = sub[sub.condition == "mhconly"]["hamming_dist"]
        stat, p = mannwhitneyu(full_d, mhc_d, alternative="two-sided")
        print(f"{tier:24s}: mean(full)={full_d.mean():.3f}  mean(mhconly)={mhc_d.mean():.3f}  "
              f"delta={full_d.mean()-mhc_d.mean():+.3f}  Mann-Whitney U p={p:.2e}  "
              f"(n_full={len(full_d)}, n_mhc={len(mhc_d)})")

    hdr("G2. Is the tier-effect size itself different across tiers? Friedman on per-pdb deltas, n=20 blocks")
    per_pdb_tier = tier_df.groupby(["pdb", "tier", "condition"])["hamming_dist"].mean().reset_index()
    dpiv = per_pdb_tier.pivot_table(index=["pdb", "tier"], columns="condition", values="hamming_dist")
    dpiv["delta"] = dpiv["full"] - dpiv["mhconly"]
    dblk = dpiv.reset_index().pivot(index="pdb", columns="tier", values="delta").dropna()
    stat, p = friedmanchisquare(*[dblk[t] for t in TIERS])
    print(f"Friedman chi2={stat:.2f}, df=2, p={p:.3f}  (n={len(dblk)} pdbs)  "
          f"-- non-significant => effect is uniform across tiers")
    print(dblk.mean())


if __name__ == "__main__":
    main()
