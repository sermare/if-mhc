#!/usr/bin/env python3
"""Same per-position h_V points as fig_if19_pmpnn_per_position_pca_umap.png (672 peptides x 9
positions = 6048 points), but colored by the per-position "surprise" (negative log-probability the
model assigned to the actual residue at that exact position) -- the natural per-position analog of
score, defined at the same granularity as these points (unlike whole-peptide score, which would just
repeat the same value 9x per peptide).
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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


@torch.no_grad()
def embed_and_surprise(model, base_batch, peptides, device, batch_size=24):
    """Returns h_V (n, pep_len, 128) AND per-position surprise (n, pep_len), in one pass."""
    X, S, mask, chain_M, residue_idx, chain_encoding_all, chain_M_pos = base_batch
    pep_len = len(peptides[0])
    h_Vs, surprises = [], []
    for start in range(0, len(peptides), batch_size):
        chunk = peptides[start:start + batch_size]
        b = len(chunk)
        Xb = X.repeat(b, 1, 1, 1)
        Sb = S.repeat(b, 1).clone()
        maskb = mask.repeat(b, 1)
        chain_Mb = chain_M.repeat(b, 1)
        residue_idxb = residue_idx.repeat(b, 1)
        chain_encb = chain_encoding_all.repeat(b, 1)
        tok_ids = []
        for i, pep in enumerate(chunk):
            toks = [AA_TO_TOK[c] for c in pep]
            Sb[i, :pep_len] = torch.tensor(toks, device=device)
            tok_ids.append(toks)
        tok_ids = torch.tensor(tok_ids, device=device)
        randnb = torch.randn(chain_Mb.shape, device=device)
        h_V = h_V_forward(model, Xb, Sb, maskb, chain_Mb * chain_M_pos.repeat(b, 1),
                           residue_idxb, chain_encb, randnb)
        h_Vs.append(h_V[:, :pep_len, :].cpu().numpy())
        logits = model.W_out(h_V[:, :pep_len, :])
        log_probs = F.log_softmax(logits, dim=-1)
        surprise = -torch.gather(log_probs, 2, tok_ids.unsqueeze(-1)).squeeze(-1)
        surprises.append(surprise.cpu().numpy())
    return np.concatenate(h_Vs, axis=0), np.concatenate(surprises, axis=0)


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
    print("Extracting per-position h_V + per-position surprise...", flush=True)
    emb3d, surprise = embed_and_surprise(model, base_batch, peptides, DEV)  # (672,9,128), (672,9)
    n, pep_len, dim = emb3d.shape
    emb_flat = emb3d.reshape(n * pep_len, dim)
    surprise_flat = surprise.reshape(-1)
    position_label = np.tile(np.arange(1, pep_len + 1), n)
    print(f"total points: {emb_flat.shape[0]}, surprise range [{surprise_flat.min():.2f}, "
          f"{surprise_flat.max():.2f}]", flush=True)

    pca = PCA(n_components=2, random_state=0)
    Z_pca = pca.fit_transform(emb_flat)
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=0, verbose=False)
    Z_umap = reducer.fit_transform(emb_flat)

    from scipy.stats import pearsonr
    r_pca1, p1 = pearsonr(Z_pca[:, 0], surprise_flat)
    print(f"PC1 vs per-position surprise: r={r_pca1:.3f} (p={p1:.2e})", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, Z, method in [(axes[0], Z_pca, "PC"), (axes[1], Z_umap, "UMAP")]:
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=surprise_flat, cmap="plasma", s=6, alpha=0.5, linewidths=0)
        ax.set_xlabel(f"{method}1"); ax.set_ylabel(f"{method}2")
        ax.set_title(f"{method}, colored by per-position surprise")
        fig.colorbar(sc, ax=ax, fraction=0.046, label="surprise = -log P(actual residue)")
    fig.suptitle(f"Per-position h_V (n={n}x{pep_len}={emb_flat.shape[0]} points), colored by "
                 f"per-position surprise (PC1 r={r_pca1:.2f})", y=1.02, fontsize=13)
    fig.tight_layout()
    out = FIG_DIR / "fig_if19_pmpnn_per_position_by_surprise.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")

    # also break down by position: within each position-cluster, does surprise show a gradient?
    fig2, axes2 = plt.subplots(3, 3, figsize=(15, 13))
    for pos in range(1, pep_len + 1):
        ax = axes2[(pos - 1) // 3, (pos - 1) % 3]
        m = position_label == pos
        sc = ax.scatter(Z_pca[m, 0], Z_pca[m, 1], c=surprise_flat[m], cmap="plasma", s=10, alpha=0.7)
        r_pos, _ = pearsonr(Z_pca[m, 0], surprise_flat[m]) if m.sum() > 2 else (np.nan, np.nan)
        ax.set_title(f"position {pos} only (r={r_pos:.2f})", fontsize=10)
        fig2.colorbar(sc, ax=ax, fraction=0.046)
    fig2.suptitle("Same PCA coordinates, split per position, colored by that position's surprise",
                  y=1.01, fontsize=13)
    fig2.tight_layout()
    out2 = FIG_DIR / "fig_if19_pmpnn_per_position_by_surprise_split.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()
