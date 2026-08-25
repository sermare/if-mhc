#!/usr/bin/env python3
"""Supervised UMAP (target=score) on the FULL unpooled embedding (9x128=1152-dim per peptide, so
position 3/2/4's signal isn't diluted by averaging with the 6 uninformative positions) -- one point
per peptide. UMAP's supervised mode explicitly pulls points with similar target values together,
instead of only preserving generic neighborhood structure.
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap
from scipy.stats import pearsonr

sys.path.insert(0, "/home/ubuntu/if-mhc/py")
from build_kd_round_sample_pmpnn_embedding import (
    load_model, h_V_forward, load_fasta_peptides_with_score, DESIGN_PATHS,
    JSONL, CHAIN_ID_JSONL, DEV, AA_TO_TOK
)
sys.path.insert(0, "/home/ubuntu/if-mhc/ProteinMPNN")
from protein_mpnn_utils import StructureDataset, tied_featurize

ROOT = Path("/home/ubuntu/if-mhc")
FIG_DIR = ROOT / "figures/fig_if19_pmpnn_score_embedding_regression"


@torch.no_grad()
def embed_unpooled(model, base_batch, peptides, device, batch_size=24):
    X, S, mask, chain_M, residue_idx, chain_encoding_all, chain_M_pos = base_batch
    pep_len = len(peptides[0])
    embeddings = []
    for start in range(0, len(peptides), batch_size):
        chunk = peptides[start:start + batch_size]
        b = len(chunk)
        Xb = X.repeat(b, 1, 1, 1)
        Sb = S.repeat(b, 1).clone()
        maskb = mask.repeat(b, 1)
        chain_Mb = chain_M.repeat(b, 1)
        residue_idxb = residue_idx.repeat(b, 1)
        chain_encb = chain_encoding_all.repeat(b, 1)
        for i, pep in enumerate(chunk):
            Sb[i, :pep_len] = torch.tensor([AA_TO_TOK[c] for c in pep], device=device)
        randnb = torch.randn(chain_Mb.shape, device=device)
        h_V = h_V_forward(model, Xb, Sb, maskb, chain_Mb * chain_M_pos.repeat(b, 1),
                           residue_idxb, chain_encb, randnb)
        embeddings.append(h_V[:, :pep_len, :].reshape(b, -1).cpu().numpy())
    return np.concatenate(embeddings, axis=0)


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
    print(f"n={len(peptides)} unique designs", flush=True)

    dataset = StructureDataset(str(JSONL), truncate=None, max_length=20000)
    chain_id_dict = json.loads(open(CHAIN_ID_JSONL).read())
    protein = dataset[0]
    X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, \
        masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, \
        dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, \
        bias_by_res_all, tied_beta = tied_featurize([protein], DEV, chain_id_dict, None, None, None, None, None)
    base_batch = (X, S, mask, chain_M, residue_idx, chain_encoding_all, chain_M_pos)
    model = load_model()

    print("Embedding (unpooled, 1152-dim)...", flush=True)
    emb = embed_unpooled(model, base_batch, peptides, DEV)
    print(f"embedding shape={emb.shape}", flush=True)

    print("Unsupervised UMAP (baseline)...", flush=True)
    unsup = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=0, verbose=False)
    Z_unsup = unsup.fit_transform(emb)
    r_u, _ = pearsonr(Z_unsup[:, 0], scores)

    results = {}
    for tw in [0.8, 0.95]:
        sup = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=0,
                         target_weight=tw, verbose=False)
        Z_sup = sup.fit_transform(emb, y=scores)
        r_s1, p_s1 = pearsonr(Z_sup[:, 0], scores)
        print(f"target_weight={tw}: supervised UMAP1 vs score: r={r_s1:.3f} (p={p_s1:.2e})", flush=True)
        results[tw] = (Z_sup, r_s1)
    best_tw = max(results, key=lambda k: abs(results[k][1]))
    Z_sup, r_s1 = results[best_tw]
    print(f"unsupervised UMAP1 vs score: r={r_u:.3f}", flush=True)
    print(f"BEST supervised (target_weight={best_tw}) UMAP1 vs score: r={r_s1:.3f}", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    sc = axes[0].scatter(Z_unsup[:, 0], Z_unsup[:, 1], c=scores, cmap="viridis_r", s=20, alpha=0.8)
    axes[0].set_title(f"UNSUPERVISED UMAP (r={r_u:.2f})")
    axes[0].set_xlabel("UMAP1"); axes[0].set_ylabel("UMAP2")
    fig.colorbar(sc, ax=axes[0], fraction=0.046, label="score")

    sc = axes[1].scatter(Z_sup[:, 0], Z_sup[:, 1], c=scores, cmap="viridis_r", s=20, alpha=0.8)
    axes[1].set_title(f"SUPERVISED UMAP, target_weight={best_tw} (r={r_s1:.2f})")
    axes[1].set_xlabel("UMAP1"); axes[1].set_ylabel("UMAP2")
    fig.colorbar(sc, ax=axes[1], fraction=0.046, label="score")

    fig.suptitle(f"Unpooled (1152-dim) embedding, n={len(scores)} peptides -- "
                 "unsupervised vs. score-supervised UMAP", y=1.02, fontsize=13)
    fig.tight_layout()
    out = FIG_DIR / "fig_if19_pmpnn_supervised_umap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
