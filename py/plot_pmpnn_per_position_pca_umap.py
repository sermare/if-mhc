#!/usr/bin/env python3
"""Treat each (peptide, position) pair as its own point -- 672 peptides x 9 positions = 6048 points
-- using the per-position h_V (NOT pooled/flattened this time), PCA/UMAP'd together, colored by (1)
which position it is (1-9) and (2) which amino acid actually occupies that position. Tests whether
h_V's per-position embedding organizes by position identity, by residue chemistry, both, or neither.
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
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
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")


def embed_per_position(model, base_batch, peptides, device, batch_size=24):
    """Returns (n*pep_len, 128) h_V vectors, one row per (peptide, position)."""
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
        embeddings.append(h_V[:, :pep_len, :].cpu().numpy())  # (b, pep_len, 128)
    return np.concatenate(embeddings, axis=0)  # (n, pep_len, 128)


def main():
    print("Loading designs...", flush=True)
    pep_scores = {}
    for temp, path in DESIGN_PATHS.items():
        for pep, score in load_fasta_peptides_with_score(path):
            if len(pep) != 9:
                continue
            pep_scores.setdefault(pep, []).append(score)
    peptides = sorted(pep_scores)
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
    print("Extracting per-position h_V (672 x 9 x 128)...", flush=True)
    emb3d = embed_per_position(model, base_batch, peptides, DEV)  # (672, 9, 128)
    n, pep_len, dim = emb3d.shape
    emb_flat = emb3d.reshape(n * pep_len, dim)

    position_label = np.tile(np.arange(1, pep_len + 1), n)
    aa_label = np.array([pep[pos] for pep in peptides for pos in range(pep_len)])
    print(f"total (peptide, position) points: {emb_flat.shape[0]}", flush=True)

    pca = PCA(n_components=2, random_state=0)
    Z_pca = pca.fit_transform(emb_flat)
    print(f"PCA var explained: {pca.explained_variance_ratio_}", flush=True)

    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=0, verbose=False)
    Z_umap = reducer.fit_transform(emb_flat)

    pos_cmap = ListedColormap(plt.cm.tab10(np.linspace(0, 1, pep_len)))
    aa_present = [a for a in AA_LIST if a in set(aa_label)]
    aa_color = {a: plt.get_cmap("tab20")(i / 20) for i, a in enumerate(AA_LIST)}

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for row, (Z, method) in enumerate([(Z_pca, "PC"), (Z_umap, "UMAP")]):
        ax = axes[row, 0]
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=position_label, cmap=pos_cmap, s=6, alpha=0.5, linewidths=0)
        cbar = fig.colorbar(sc, ax=ax, ticks=range(1, pep_len + 1), fraction=0.046)
        ax.set_title(f"{method}, colored by position")
        ax.set_xlabel(f"{method}1"); ax.set_ylabel(f"{method}2")

        ax = axes[row, 1]
        for a in aa_present:
            m = aa_label == a
            ax.scatter(Z[m, 0], Z[m, 1], s=6, alpha=0.5, linewidths=0, color=aa_color[a], label=a)
        ax.set_title(f"{method}, colored by amino acid")
        ax.legend(fontsize=6, markerscale=2, ncol=2, loc="best")
        ax.set_xlabel(f"{method}1")

    fig.suptitle(f"Per-position h_V (n={n} peptides x {pep_len} positions = {emb_flat.shape[0]} points), "
                 "noMHC ProteinMPNN, 2P5E", y=1.01, fontsize=13)
    fig.tight_layout()
    out = FIG_DIR / "fig_if19_pmpnn_per_position_pca_umap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
