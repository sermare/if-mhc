#!/usr/bin/env python3
"""Proper quantitative test of whether ProteinMPNN's (noMHC) score is predictable from its own
128-dim mean-pooled embedding -- not just eyeballing a 2D PCA/UMAP scatter.

1. Ridge regression (cross-validated) predicting score from the FULL 128-dim embedding.
2. Correlation of score against each of the top 10 individual PCs (not just PC1/PC2).
3. A plot: CV-predicted vs actual score, and |correlation| per PC.

Uses the same 672 unique noMHC 2P5E designs (T=0.1+T=0.3 pooled) and per-peptide mean score as the
last figure, re-embedded via the same h_V extraction.
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict, KFold
from scipy.stats import pearsonr

sys.path.insert(0, "/home/ubuntu/if-mhc/py")
from build_kd_round_sample_pmpnn_embedding import (
    load_model, embed_peptides, load_fasta_peptides_with_score, DESIGN_PATHS,
    JSONL, CHAIN_ID_JSONL, DEV
)
sys.path.insert(0, "/home/ubuntu/if-mhc/ProteinMPNN")
from protein_mpnn_utils import StructureDataset, tied_featurize

ROOT = Path("/home/ubuntu/if-mhc")
FIG_DIR = ROOT / "figures/fig_if19_pmpnn_score_embedding_regression"
FIG_DIR.mkdir(exist_ok=True, parents=True)


def main():
    print("Loading designs + scores...", flush=True)
    pep_scores = {}
    for temp, path in DESIGN_PATHS.items():
        for pep, score in load_fasta_peptides_with_score(path):
            if len(pep) != 9:
                continue
            pep_scores.setdefault(pep, []).append(score)
    peptides = sorted(pep_scores)
    scores = np.array([float(np.mean(pep_scores[p])) for p in peptides])
    print(f"n={len(peptides)} unique designs, score range [{scores.min():.3f}, {scores.max():.3f}]", flush=True)

    print("Embedding via noMHC ProteinMPNN...", flush=True)
    dataset = StructureDataset(str(JSONL), truncate=None, max_length=20000)
    chain_id_dict = json.loads(open(CHAIN_ID_JSONL).read())
    protein = dataset[0]
    X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, \
        masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, \
        dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, \
        bias_by_res_all, tied_beta = tied_featurize([protein], DEV, chain_id_dict, None, None, None, None, None)
    base_batch = (X, S, mask, chain_M, residue_idx, chain_encoding_all, chain_M_pos)
    model = load_model()
    emb = embed_peptides(model, base_batch, peptides, DEV)
    print(f"embedding shape={emb.shape}", flush=True)

    np.savez(FIG_DIR / "embeddings_and_scores.npz", emb=emb, scores=scores,
             peptides=np.array(peptides))

    # 1. Ridge regression on FULL 128-dim embedding, cross-validated
    print("\n=== Full 128-dim embedding -> score, 5-fold CV Ridge regression ===", flush=True)
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    ridge = RidgeCV(alphas=np.logspace(-2, 4, 20))
    pred = cross_val_predict(ridge, emb, scores, cv=kf)
    r, p = pearsonr(pred, scores)
    r2 = 1 - np.sum((scores - pred) ** 2) / np.sum((scores - scores.mean()) ** 2)
    print(f"CV Pearson r = {r:.3f} (p={p:.2e}), CV R^2 = {r2:.3f}", flush=True)

    # 2. Correlation of score with each of the top 10 individual PCs
    print("\n=== Correlation of score with individual PCs ===", flush=True)
    pca = PCA(n_components=10, random_state=0)
    pcs = pca.fit_transform(emb)
    pc_corrs = []
    for i in range(10):
        r_i, p_i = pearsonr(pcs[:, i], scores)
        pc_corrs.append(r_i)
        print(f"  PC{i+1} (var={pca.explained_variance_ratio_[i]*100:.1f}%): "
              f"r={r_i:.3f} (p={p_i:.2e})", flush=True)

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]
    ax.scatter(scores, pred, s=15, alpha=0.5)
    lims = [min(scores.min(), pred.min()), max(scores.max(), pred.max())]
    ax.plot(lims, lims, "k--", linewidth=1)
    ax.set_xlabel("actual ProteinMPNN score")
    ax.set_ylabel("CV-predicted score (Ridge, full 128-dim embedding)")
    ax.set_title(f"5-fold CV: r={r:.2f}, R^2={r2:.2f}, n={len(scores)}")

    ax = axes[1]
    ax.bar(range(1, 11), [abs(c) for c in pc_corrs], color="#4C72B0")
    ax.set_xlabel("principal component")
    ax.set_ylabel("|Pearson r| vs. score")
    ax.set_title("Per-PC correlation with score (top 10 PCs)")
    ax.set_xticks(range(1, 11))

    fig.suptitle("Does ProteinMPNN's (noMHC) own embedding predict its own score?", y=1.02, fontsize=13)
    fig.tight_layout()
    out = FIG_DIR / "fig_if19_pmpnn_score_embedding_regression.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
