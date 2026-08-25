#!/usr/bin/env python3
"""Add LigandMPNN to the KD-vs-score comparison on the 51 NY-ESO-1/1G4c58c61 peptides.

ProteinMPNN and noMHC ProteinMPNN scores come from the existing score_only runs
(outputs/kd_scoring/{vanilla,nomhc}). LigandMPNN has no --path_to_fasta in score.py, so each
peptide was threaded onto the 2P5E chain-C backbone (py/thread_kd_peptides_pdb.py) and scored
autoregressively; its score here is the mean per-residue NLL over the 9 chain-C positions,
averaged over 10 decoding orders -- the same quantity ProteinMPNN's score_only reports, so the
three are directly comparable in scale and direction (lower = model finds the sequence more
favorable on this backbone).
"""
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy.stats import pearsonr, spearmanr, mannwhitneyu

ROOT = Path("/home/ubuntu/if-mhc")
SC = ROOT / "outputs/kd_scoring/ligandmpnn/score_only"


def ligandmpnn_score(pt_path):
    d = torch.load(pt_path, map_location="cpu", weights_only=False)
    lp = np.asarray(d["log_probs"])          # (batches, residues, 21)
    nat = np.asarray(d["native_sequence"])   # threaded sequence indices
    sel = np.asarray(d["chain_mask"]) == 1   # chain C only
    nll = -lp[:, sel, :][:, np.arange(sel.sum()), nat[sel]]
    return float(nll.mean())


def main():
    df = pd.read_csv(ROOT / "outputs/analysis/kd_score_correlation.csv")
    df["score_ligandmpnn"] = [
        ligandmpnn_score(SC / f"{p}.pdb.pt") if (SC / f"{p}.pdb.pt").exists()
        else (ligandmpnn_score(SC / f"{p}.pt") if (SC / f"{p}.pt").exists() else np.nan)
        for p in df["Peptide"]
    ]
    got = df["score_ligandmpnn"].notna().sum()
    print(f"LigandMPNN scores recovered for {got}/{len(df)} peptides")
    df.to_csv(ROOT / "outputs/analysis/kd_score_correlation_3model.csv", index=False)

    MODELS = [("score_vanilla", "ProteinMPNN"),
              ("score_nomhc", "noMHC ProteinMPNN"),
              ("score_ligandmpnn", "LigandMPNN")]
    d = df.dropna(subset=["score_ligandmpnn"])

    print("\n=== score scale (lower = more favorable) ===")
    for c, lab in MODELS:
        print(f"  {lab:20s} mean={d[c].mean():.3f}  sd={d[c].std():.3f}  min={d[c].min():.3f}  max={d[c].max():.3f}")

    print("\n=== cross-model agreement (Pearson r) ===")
    for i in range(len(MODELS)):
        for j in range(i + 1, len(MODELS)):
            a, la = MODELS[i]; b, lb = MODELS[j]
            r, p = pearsonr(d[a], d[b])
            print(f"  {la:20s} vs {lb:20s} r={r:+.3f} (p={p:.2e})")

    print("\n=== binders vs non-binders ===")
    for c, lab in MODELS:
        bd = d.loc[d.is_binder, c]; nb = d.loc[~d.is_binder, c]
        u, p = mannwhitneyu(bd, nb, alternative="two-sided")
        print(f"  {lab:20s} binders(n={len(bd)})={bd.mean():.3f}  non(n={len(nb)})={nb.mean():.3f}  diff={bd.mean()-nb.mean():+.3f}  MW p={p:.3f}")

    print("\n=== among measured binders: score vs log10(KD) ===")
    s = d[d.kd_value.notna()].copy(); s["logKD"] = np.log10(s.kd_value)
    for c, lab in MODELS:
        r, p = pearsonr(s[c], s.logKD); rho, ps = spearmanr(s[c], s.logKD)
        print(f"  {lab:20s} Pearson r={r:+.3f} (p={p:.3f})  Spearman rho={rho:+.3f} (p={ps:.3f})  n={len(s)}")

    print("\n=== control: score vs Hamming distance to native ===")
    for c, lab in MODELS:
        r, p = pearsonr(d[c], d.hamming_to_native)
        print(f"  {lab:20s} r={r:+.3f} (p={p:.3f})")


if __name__ == "__main__":
    main()
