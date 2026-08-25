#!/usr/bin/env python3
"""Same analysis as fig_if16 (real NGS peptides sampled across rounds, PCA/UMAP colored by round /
median count / Hamming distance to native) but using ProteinMPNN's OWN learned embedding instead of
the external CNN classifier's -- the per-residue hidden state h_V immediately before the final
W_out linear projection to amino-acid logits (ProteinMPNN.forward(), protein_mpnn_utils.py L1098),
mean-pooled over the peptide (chain C) positions to get one fixed-length vector per candidate
sequence. Run once for vanilla weights, once for noMHC weights -- these are two DIFFERENT models
(different training data), so their embedding spaces are independent, not comparable to each other
directly (same caveat as the CNN script's per-TCR independence).

Structure: 2P5E full context (chains A=MHC,B=b2m,C=peptide,D=TCRa,E=TCRb), same parsed.jsonl /
chain_id.jsonl used for the original 2P5E generation campaigns and the KD score_only run.

Run under esmcba (has ProteinMPNN's torch dependency + sklearn + umap).
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.decomposition import PCA
import umap

sys.path.insert(0, "/home/ubuntu/if-mhc/ProteinMPNN")
from protein_mpnn_utils import ProteinMPNN, StructureDataset, tied_featurize, gather_nodes, cat_neighbors_nodes

sys.path.insert(0, "/home/ubuntu/pmhc/modeling/ONG229/py")
import nyeso1_ranking_lib as lib

ROOT = Path("/home/ubuntu/if-mhc")
FIG_DIR = ROOT / "figures/fig_if17_round_sample_pmpnn_embedding"
FIG_DIR.mkdir(exist_ok=True, parents=True)
RUNS_DIR = Path("/home/ubuntu/pmhc/modeling/ONG229/lm_runs")

DEV = torch.device("cuda")
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'
AA_TO_TOK = {a: ALPHABET.index(a) for a in AA_LIST}
NATIVE = "SLLMWITQC"
TCR = "1G4c58c61"
N_SAMPLE = 20_000
SEED = 0
ROUND_LABELS = ["R0 only", "R3", "R4+"]

JSONL = ROOT / "outputs/mpnn_2p5e_T01_20k/parsed.jsonl"
CHAIN_ID_JSONL = ROOT / "outputs/mpnn_2p5e_T01_20k/chain_id.jsonl"
WEIGHTS = {
    "vanilla": {"path": None, "hidden_dim": 128},
    "nomhc": {"path": ROOT / "ProteinMPNN/nomhc_model_weights/proteinmpnn_nomhc.pt", "hidden_dim": 128},
}
VANILLA_CKPT = ROOT / "ProteinMPNN/vanilla_model_weights/v_48_020.pt"


def load_model(weights_name):
    ckpt_path = WEIGHTS[weights_name]["path"] or VANILLA_CKPT
    ckpt = torch.load(ckpt_path, map_location=DEV)
    hidden_dim = 128
    model = ProteinMPNN(num_letters=21, node_features=hidden_dim, edge_features=hidden_dim,
                         hidden_dim=hidden_dim, num_encoder_layers=3, num_decoder_layers=3,
                         augment_eps=0.0, k_neighbors=ckpt['num_edges']).to(DEV)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


@torch.no_grad()
def h_V_forward(model, X, S, mask, chain_M, residue_idx, chain_encoding_all, randn):
    """Replica of ProteinMPNN.forward() up to (not including) self.W_out -- returns h_V."""
    E, E_idx = model.features(X, mask, residue_idx, chain_encoding_all)
    h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
    h_E = model.W_e(E)
    mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
    mask_attend = mask.unsqueeze(-1) * mask_attend
    for layer in model.encoder_layers:
        h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)

    h_S = model.W_s(S)
    h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)
    h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
    h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

    chain_M = chain_M * mask
    decoding_order = torch.argsort((chain_M + 0.0001) * (torch.abs(randn)))
    mask_size = E_idx.shape[1]
    permutation_matrix_reverse = F.one_hot(decoding_order, num_classes=mask_size).float()
    order_mask_backward = torch.einsum('ij, biq, bjp->bqp',
        (1 - torch.triu(torch.ones(mask_size, mask_size, device=X.device))),
        permutation_matrix_reverse, permutation_matrix_reverse)
    mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
    mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
    mask_bw = mask_1D * mask_attend
    mask_fw = mask_1D * (1. - mask_attend)
    h_EXV_encoder_fw = mask_fw * h_EXV_encoder
    for layer in model.decoder_layers:
        h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
        h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
        h_V = layer(h_V, h_ESV, mask)
    return h_V


def embed_peptides(model, base_batch, peptides, device, batch_size=24):
    """base_batch: the tied_featurize() output for the 2P5E structure (single structure, repeated
    per-chunk). Peptide (chain C) is always the first `pep_len` positions of S (masked/designed
    chain ordering, same convention relied on throughout this project's score_only usage)."""
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
        pep_emb = h_V[:, :pep_len, :].mean(dim=1).cpu().numpy()
        embeddings.append(pep_emb)
    return np.concatenate(embeddings, axis=0)


def hamming(pep, native=NATIVE):
    return sum(a != b for a, b in zip(pep, native))


def furthest_round(df):
    counts = df[lib.ROUND_COLS].values.astype(np.float64)
    out = np.zeros(len(df), dtype=int)
    for i, col in enumerate(lib.ROUND_COLS):
        out[counts[:, i] > 0] = i
    return out


def main():
    print("Loading 2P5E structure + real NGS sample...", flush=True)
    dataset = StructureDataset(str(JSONL), truncate=None, max_length=20000)
    chain_id_dict = json.loads(open(CHAIN_ID_JSONL).read())
    protein = dataset[0]
    X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, \
        masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, \
        dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, \
        bias_by_res_all, tied_beta = tied_featurize([protein], DEV, chain_id_dict, None, None, None, None, None)
    base_batch = (X, S, mask, chain_M, residue_idx, chain_encoding_all, chain_M_pos)

    tab_data = lib.load_tab_data()
    df = tab_data[f"NYESO1__{TCR}"]
    rng = np.random.RandomState(SEED)
    n_sample = min(N_SAMPLE, len(df))
    sample_df = df.iloc[rng.choice(len(df), size=n_sample, replace=False)].copy()
    peptides = sample_df["Peptide"].values
    fr = furthest_round(sample_df)
    median_count = np.median(sample_df[lib.ROUND_COLS].values.astype(np.float64), axis=1)
    hd = np.array([hamming(p) for p in peptides])
    print(f"n={len(peptides):,} real peptides sampled", flush=True)

    def plot_grid(Z, method_name, weights_name, out_name):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        round_cmap = ListedColormap(["#bdbdbd", "#74a9cf", "#08306b"])
        round_norm = BoundaryNorm(np.arange(4) - 0.5, round_cmap.N)

        ax = axes[0]
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=fr, cmap=round_cmap, norm=round_norm, s=3, alpha=0.2, linewidths=0)
        ax.set_title("colored by furthest round reached")
        cbar = fig.colorbar(sc, ax=ax, ticks=range(3), fraction=0.046)
        cbar.ax.set_yticklabels(ROUND_LABELS)
        ax.set_xlabel(f"{method_name}1"); ax.set_ylabel(f"{method_name}2")

        ax = axes[1]
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=np.log1p(median_count), cmap="viridis", s=3, alpha=0.2, linewidths=0)
        ax.set_title("colored by log1p(median count)")
        fig.colorbar(sc, ax=ax, fraction=0.046, label="log1p(median count)")
        ax.set_xlabel(f"{method_name}1")

        ax = axes[2]
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=hd, cmap="plasma", s=3, alpha=0.2, linewidths=0)
        ax.set_title("colored by Hamming distance to native")
        fig.colorbar(sc, ax=ax, fraction=0.046, label="Hamming distance")
        ax.set_xlabel(f"{method_name}1")

        fig.suptitle(f"ProteinMPNN ({weights_name}) embedding of real NGS sample ({method_name}) -- "
                     f"native peptide: {NATIVE} (n={len(peptides):,})", y=1.03, fontsize=13)
        fig.tight_layout()
        out = FIG_DIR / out_name
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}", flush=True)

    for weights_name in ["vanilla", "nomhc"]:
        print(f"=== {weights_name} ===", flush=True)
        model = load_model(weights_name)
        emb = embed_peptides(model, base_batch, peptides, DEV)
        print(f"  embedding shape={emb.shape}", flush=True)

        pca = PCA(n_components=2, random_state=0)
        Z_pca = pca.fit_transform(emb)
        print(f"  PCA var explained: {pca.explained_variance_ratio_}", flush=True)
        plot_grid(Z_pca, "PC", weights_name, f"fig_if17_pmpnn_{weights_name}_pca.png")

        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=0, verbose=False)
        Z_umap = reducer.fit_transform(emb)
        plot_grid(Z_umap, "UMAP", weights_name, f"fig_if17_pmpnn_{weights_name}_umap.png")
        del model
        torch.cuda.empty_cache()

    print("DONE")


if __name__ == "__main__":
    main()
