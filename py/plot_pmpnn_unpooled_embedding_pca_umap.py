#!/usr/bin/env python3
"""Same as fig_if19 but WITHOUT mean-pooling over positions -- flatten h_V's 9 positions x 128 dims
into one 1152-dim vector per peptide (full per-position resolution preserved), then PCA/UMAP that,
colored by score. Re-embeds the same 672 unique noMHC 2P5E designs.
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
import umap

sys.path.insert(0, "/home/ubuntu/if-mhc/py")
from build_kd_round_sample_pmpnn_embedding import (
    load_model, h_V_forward, load_fasta_peptides_with_score, DESIGN_PATHS,
    JSONL, CHAIN_ID_JSONL, DEV, AA_TO_TOK
)
sys.path.insert(0, "/home/ubuntu/if-mhc/ProteinMPNN")
from protein_mpnn_utils import StructureDataset, tied_featurize

ROOT = Path("/home/ubuntu/if-mhc")
FIG_DIR = ROOT / "figures/fig_if19_pmpnn_score_embedding_regression"


def embed_peptides_unpooled(model, base_batch, peptides, device, batch_size=24):
    """Same as embed_peptides but flattens (9, 128) -> 1152, instead of mean-pooling."""
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
        pep_emb = h_V[:, :pep_len, :].reshape(b, -1).cpu().numpy()
        embeddings.append(pep_emb)
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
    print("Embedding (unpooled, 9x128=1152-dim per peptide)...", flush=True)
    emb = embed_peptides_unpooled(model, base_batch, peptides, DEV)
    print(f"embedding shape={emb.shape}", flush=True)

    pca = PCA(n_components=2, random_state=0)
    Z_pca = pca.fit_transform(emb)
    print(f"PCA var explained: {pca.explained_variance_ratio_}", flush=True)

    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=0, verbose=False)
    Z_umap = reducer.fit_transform(emb)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, Z, method in [(axes[0], Z_pca, "PC"), (axes[1], Z_umap, "UMAP")]:
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=scores, cmap="viridis_r", s=20, alpha=0.8, edgecolors="none")
        ax.set_xlabel(f"{method}1"); ax.set_ylabel(f"{method}2")
        ax.set_title(f"{method} of UNPOOLED embedding (9x128=1152-dim, n={len(scores)})")
        fig.colorbar(sc, ax=ax, fraction=0.046, label="ProteinMPNN score\n(lower = more favorable)")
    fig.suptitle("noMHC ProteinMPNN embedding, NOT position-pooled, colored by score", y=1.02, fontsize=13)
    fig.tight_layout()
    out = FIG_DIR / "fig_if19_pmpnn_unpooled_embedding_pca_umap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
