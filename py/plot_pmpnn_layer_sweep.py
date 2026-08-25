#!/usr/bin/env python3
"""Empirically test three representations against ProteinMPNN score, to settle whether "going up"
(encoder-only, structure-only h_V) or "going down" (full log_probs distribution) does any better
than the decoder's h_V -- plus the one representation that SHOULD trivially work: the per-position
log-probability the model assigned to the actual residue at each position (whose mean literally IS
the score).

1. UP: encoder-only h_V (computed before any sequence conditioning at all) -- predicted to be
   ~constant across peptides, since it's structure-only and all peptides share the same backbone.
2. DOWN: full log_probs (21-dim distribution per position, NOT indexed at the actual residue) --
   predicted to fail for the same architectural reason h_V does: it's the model's belief, not
   conditioned on which residue is actually there.
3. THE ANSWER: per-position log_probs[i, S_i] (9-dim "surprise profile" per peptide, mean = score)
   -- this is the piece that was always missing: it combines the belief (log_probs) with the
   realized identity (S) that h_V and full log_probs both lack.
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
    load_model, load_fasta_peptides_with_score, DESIGN_PATHS,
    JSONL, CHAIN_ID_JSONL, DEV, AA_TO_TOK
)
sys.path.insert(0, "/home/ubuntu/if-mhc/ProteinMPNN")
from protein_mpnn_utils import StructureDataset, tied_featurize, gather_nodes, cat_neighbors_nodes

ROOT = Path("/home/ubuntu/if-mhc")
FIG_DIR = ROOT / "figures/fig_if19_pmpnn_score_embedding_regression"


@torch.no_grad()
def encoder_only_h_V(model, X, mask, residue_idx, chain_encoding_all):
    """Structure-only encoder output -- computed BEFORE any sequence (S) is involved at all."""
    E, E_idx = model.features(X, mask, residue_idx, chain_encoding_all)
    h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
    h_E = model.W_e(E)
    mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
    mask_attend = mask.unsqueeze(-1) * mask_attend
    for layer in model.encoder_layers:
        h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
    return h_V, h_E, E_idx


@torch.no_grad()
def full_forward_with_logprobs(model, X, S, mask, chain_M, residue_idx, chain_encoding_all, randn):
    """Same as h_V_forward but continues through W_out + log_softmax, returning (h_V, log_probs)."""
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

    chain_M2 = chain_M * mask
    decoding_order = torch.argsort((chain_M2 + 0.0001) * (torch.abs(randn)))
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
    logits = model.W_out(h_V)
    log_probs = F.log_softmax(logits, dim=-1)
    return h_V, log_probs


def batched_extract(model, base_batch, peptides, device, batch_size=24):
    X, S, mask, chain_M, residue_idx, chain_encoding_all, chain_M_pos = base_batch
    pep_len = len(peptides[0])
    enc_pooled, logprob_full_pooled, surprise_profile = [], [], []
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
        tok_ids = torch.tensor(tok_ids, device=device)  # (b, pep_len)

        h_V_enc, _, _ = encoder_only_h_V(model, Xb, maskb, residue_idxb, chain_encb)
        enc_pooled.append(h_V_enc[:, :pep_len, :].mean(dim=1).cpu().numpy())

        randnb = torch.randn(chain_Mb.shape, device=device)
        _, log_probs = full_forward_with_logprobs(model, Xb, Sb, maskb, chain_Mb * chain_M_pos.repeat(b, 1),
                                                    residue_idxb, chain_encb, randnb)
        lp_pep = log_probs[:, :pep_len, :]  # (b, pep_len, 21)
        logprob_full_pooled.append(lp_pep.reshape(b, -1).cpu().numpy())
        surprise = torch.gather(lp_pep, 2, tok_ids.unsqueeze(-1)).squeeze(-1)  # (b, pep_len)
        surprise_profile.append((-surprise).cpu().numpy())  # negative log-prob = per-position "score"

    return (np.concatenate(enc_pooled, axis=0), np.concatenate(logprob_full_pooled, axis=0),
            np.concatenate(surprise_profile, axis=0))


def scatter_colored(ax, Z, c, cmap="viridis_r"):
    return ax.scatter(Z[:, 0], Z[:, 1], c=c, cmap=cmap, s=20, alpha=0.8, edgecolors="none")


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
    print("Extracting encoder-only h_V, full log_probs, and per-position surprise profile...", flush=True)
    enc_emb, logprob_emb, surprise = batched_extract(model, base_batch, peptides, DEV)
    print(f"encoder-only h_V shape={enc_emb.shape}; full log_probs shape={logprob_emb.shape}; "
          f"surprise profile shape={surprise.shape}", flush=True)

    # sanity check: mean(surprise, axis=1) should equal `scores` (both are the model's own score calc)
    recomputed_score = surprise.mean(axis=1)
    check_r = np.corrcoef(recomputed_score, scores)[0, 1]
    print(f"sanity check -- recomputed score vs. cached score: r={check_r:.4f} "
          f"(should be ~1.0; small deviations from stochastic decoding order)", flush=True)

    # 1. UP: is encoder-only h_V actually constant across peptides?
    enc_std_per_dim = enc_emb.std(axis=0)
    print(f"\n=== UP (encoder-only h_V) ===\nstd across peptides per-dim: "
          f"mean={enc_std_per_dim.mean():.6f}, max={enc_std_per_dim.max():.6f} "
          f"(compare to decoder h_V's typical per-dim std ~0.3-1)", flush=True)

    pca_enc = PCA(n_components=2, random_state=0)
    Z_enc = pca_enc.fit_transform(enc_emb)
    print(f"PCA var explained: {pca_enc.explained_variance_ratio_}", flush=True)

    # 2. DOWN: full log_probs (21-dim/position, NOT indexed at S)
    pca_lp = PCA(n_components=2, random_state=0)
    Z_lp = pca_lp.fit_transform(logprob_emb)
    print(f"\n=== DOWN (full log_probs, 9x21=189-dim) ===\nPCA var explained: "
          f"{pca_lp.explained_variance_ratio_}", flush=True)
    from scipy.stats import pearsonr
    r_lp, p_lp = pearsonr(Z_lp[:, 0], scores)
    print(f"PC1 vs score: r={r_lp:.3f} (p={p_lp:.2e})", flush=True)

    # 3. THE ANSWER: per-position surprise profile (9-dim, mean = score by construction)
    pca_sp = PCA(n_components=2, random_state=0)
    Z_sp = pca_sp.fit_transform(surprise)
    print(f"\n=== per-position surprise profile (9-dim) ===\nPCA var explained: "
          f"{pca_sp.explained_variance_ratio_}", flush=True)
    r_sp, p_sp = pearsonr(Z_sp[:, 0], scores)
    print(f"PC1 vs score: r={r_sp:.3f} (p={p_sp:.2e})", flush=True)
    per_pos_var = surprise.var(axis=0)
    print(f"per-position variance in surprise (which positions drive score variation): "
          f"{np.round(per_pos_var, 4)}", flush=True)

    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=0, verbose=False)
    Z_sp_umap = reducer.fit_transform(surprise)

    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    sc = scatter_colored(axes[0, 0], Z_enc, scores)
    axes[0, 0].set_title(f"UP: encoder-only h_V (PC1 var={pca_enc.explained_variance_ratio_[0]*100:.1f}%)")
    fig.colorbar(sc, ax=axes[0, 0], fraction=0.046)

    sc = scatter_colored(axes[0, 1], Z_lp, scores)
    axes[0, 1].set_title(f"DOWN: full log_probs (r={r_lp:.2f})")
    fig.colorbar(sc, ax=axes[0, 1], fraction=0.046)

    sc = scatter_colored(axes[0, 2], Z_sp, scores)
    axes[0, 2].set_title(f"per-position surprise profile, PCA (r={r_sp:.2f})")
    fig.colorbar(sc, ax=axes[0, 2], fraction=0.046, label="score")

    sc = scatter_colored(axes[1, 2], Z_sp_umap, scores)
    axes[1, 2].set_title("per-position surprise profile, UMAP")
    fig.colorbar(sc, ax=axes[1, 2], fraction=0.046, label="score")

    axes[1, 0].bar(range(1, 10), per_pos_var, color="#4C72B0")
    axes[1, 0].set_xlabel("peptide position"); axes[1, 0].set_ylabel("variance of surprise (-log prob)")
    axes[1, 0].set_title("which positions drive score variance")
    axes[1, 0].set_xticks(range(1, 10))

    axes[1, 1].axis("off")
    axes[1, 1].text(0.05, 0.6, f"encoder-only h_V:\n  per-dim std mean={enc_std_per_dim.mean():.2e}\n"
                     f"  (decoder h_V per-dim std ~0.3-1)\n  -> essentially CONSTANT across peptides\n\n"
                     f"full log_probs PC1 vs score: r={r_lp:.2f}\n"
                     f"surprise-profile PC1 vs score: r={r_sp:.2f}\n"
                     f"(surprise profile mean = score, by construction)", fontsize=11, va="top")

    for ax in [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 2]]:
        ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")

    fig.suptitle("Layer sweep: does 'up' (encoder) or 'down' (log_probs) correlate with score any better?",
                 y=1.01, fontsize=13)
    fig.tight_layout()
    out = FIG_DIR / "fig_if19_pmpnn_layer_sweep.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
