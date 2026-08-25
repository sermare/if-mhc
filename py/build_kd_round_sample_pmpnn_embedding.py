#!/usr/bin/env python3
"""Downsampled, balanced version of the ProteinMPNN embedding analysis: 100 real peptides per round
(R0-only/R3/R4+), all 51 KD-tested peptides, and noMHC's own T=0.1 + T=0.3 unique 2P5E designs --
all embedded together (one shared PCA/UMAP fit) via noMHC ProteinMPNN's h_V (same extraction as
fig_if17), so every population is actually visible instead of the small ones being drowned out by
a 20k-point real-library background.

3 panels: colored by round (opaque, no alpha), colored by KD value (binders only; N.B. marked),
colored by generation temperature (T=0.1 vs T=0.3, our own designs only).

Run under protenix (torch+umap+sklearn+GPU) or esmcba (torch only, no plotting deps -- protenix has
both here).
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
from sklearn.decomposition import PCA
import umap

sys.path.insert(0, "/home/ubuntu/if-mhc/ProteinMPNN")
from protein_mpnn_utils import ProteinMPNN, StructureDataset, tied_featurize, gather_nodes, cat_neighbors_nodes

sys.path.insert(0, "/home/ubuntu/pmhc/modeling/ONG229/py")
import nyeso1_ranking_lib as lib

ROOT = Path("/home/ubuntu/if-mhc")
FIG_DIR = ROOT / "figures/fig_if18_kd_round_sample_pmpnn_embedding"
FIG_DIR.mkdir(exist_ok=True, parents=True)

DEV = torch.device("cuda")
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'
AA_TO_TOK = {a: ALPHABET.index(a) for a in AA_LIST}
NATIVE = "SLLMWITQC"
TCR = "1G4c58c61"
N_PER_ROUND = 100
SEED = 0

JSONL = ROOT / "outputs/mpnn_2p5e_T01_20k/parsed.jsonl"
CHAIN_ID_JSONL = ROOT / "outputs/mpnn_2p5e_T01_20k/chain_id.jsonl"
NOMHC_CKPT = ROOT / "ProteinMPNN/nomhc_model_weights/proteinmpnn_nomhc.pt"

DESIGN_PATHS = {
    "T=0.1": ROOT / "outputs/mpnn_2p5e_T01_20k/seqs/nomhc_2P5E.fa",
    "T=0.3": ROOT / "outputs/mpnn_2p5e_T03_20k/seqs/nomhc_2P5E.fa",
}


def load_model():
    ckpt = torch.load(NOMHC_CKPT, map_location=DEV)
    hidden_dim = 128
    model = ProteinMPNN(num_letters=21, node_features=hidden_dim, edge_features=hidden_dim,
                         hidden_dim=hidden_dim, num_encoder_layers=3, num_decoder_layers=3,
                         augment_eps=0.0, k_neighbors=ckpt['num_edges']).to(DEV)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


@torch.no_grad()
def h_V_forward(model, X, S, mask, chain_M, residue_idx, chain_encoding_all, randn):
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


def parse_kd(val):
    if pd.isna(val) or val == "N.B.":
        return (val != "N.B."), np.nan
    try:
        return True, float(val)
    except ValueError:
        return None, np.nan


def peptide_from_ligandmpnn_line(line):
    return line.strip().split(":")[2]


def load_fasta_peptides_with_score(path):
    import re
    pairs = []
    with open(path) as f:
        lines = f.read().splitlines()
    for i in range(0, len(lines) - 1, 2):
        header = lines[i]
        if not header.startswith(">") or "designed_chains" in header:
            continue  # skip the native-reference entry (entry 0), keep only generated samples
        m = re.search(r"score=([-\d.]+)", header)
        if m is None:
            continue
        pairs.append((lines[i + 1].strip(), float(m.group(1))))
    return pairs


def main():
    rng = np.random.RandomState(SEED)

    print("Sampling 100 real peptides per round...", flush=True)
    tab_data = lib.load_tab_data()
    df = tab_data[f"NYESO1__{TCR}"]
    counts = df[lib.ROUND_COLS].values.astype(np.float64)
    r0_only = df[(counts[:, 0] > 0) & (counts[:, 1] == 0) & (counts[:, 2] == 0)]
    r3_only = df[(counts[:, 1] > 0) & (counts[:, 2] == 0)]
    r4_plus = df[(counts[:, 2] > 0)]
    round_samples = {}
    for label, sub in [("R0", r0_only), ("R3", r3_only), ("R4", r4_plus)]:
        n = min(N_PER_ROUND, len(sub))
        round_samples[label] = sub.iloc[rng.choice(len(sub), size=n, replace=False)]["Peptide"].tolist()
        print(f"  {label}: {len(round_samples[label])} peptides", flush=True)

    print("Loading KD-tested peptides...", flush=True)
    kin = pd.read_csv("/tmp/kd_peptides.csv")
    parsed = kin["KD_raw"].apply(parse_kd)
    kin["is_binder"] = parsed.apply(lambda t: t[0])
    kin["kd_value"] = parsed.apply(lambda t: t[1])
    kd_peps = kin["Peptide"].tolist()

    print("Loading our own noMHC designs (T=0.1, T=0.3), pooled with the model's own score...", flush=True)
    pep_scores = {}
    for temp, path in DESIGN_PATHS.items():
        for pep, score in load_fasta_peptides_with_score(path):
            if len(pep) != 9:
                continue
            pep_scores.setdefault(pep, []).append(score)
    design_peps = sorted(pep_scores)
    design_score = [float(np.mean(pep_scores[p])) for p in design_peps]
    print(f"  {len(design_peps)} unique designs (T=0.1+T=0.3 pooled), "
          f"mean score range [{min(design_score):.2f}, {max(design_score):.2f}]", flush=True)

    # combined population, with a "population" tag for coloring
    all_peps = round_samples["R0"] + round_samples["R3"] + round_samples["R4"] + kd_peps + design_peps
    pop_round = (["R0"] * len(round_samples["R0"]) + ["R3"] * len(round_samples["R3"]) +
                 ["R4"] * len(round_samples["R4"]) + [None] * len(kd_peps) + [None] * len(design_peps))
    pop_kd_binder = [None] * (len(round_samples["R0"]) + len(round_samples["R3"]) + len(round_samples["R4"])) + \
                    kin["is_binder"].tolist() + [None] * len(design_peps)
    pop_kd_value = [np.nan] * (len(round_samples["R0"]) + len(round_samples["R3"]) + len(round_samples["R4"])) + \
                   kin["kd_value"].tolist() + [np.nan] * len(design_peps)
    pop_score = [np.nan] * (len(round_samples["R0"]) + len(round_samples["R3"]) + len(round_samples["R4"]) + len(kd_peps)) + \
                design_score

    print(f"total peptides to embed: {len(all_peps)}", flush=True)

    dataset = StructureDataset(str(JSONL), truncate=None, max_length=20000)
    chain_id_dict = json.loads(open(CHAIN_ID_JSONL).read())
    protein = dataset[0]
    X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, \
        masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, \
        dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, \
        bias_by_res_all, tied_beta = tied_featurize([protein], DEV, chain_id_dict, None, None, None, None, None)
    base_batch = (X, S, mask, chain_M, residue_idx, chain_encoding_all, chain_M_pos)

    model = load_model()
    emb = embed_peptides(model, base_batch, all_peps, DEV)
    print(f"embedding shape={emb.shape}", flush=True)

    pop_round = np.array(pop_round, dtype=object)
    pop_kd_binder = np.array(pop_kd_binder, dtype=object)
    pop_kd_value = np.array(pop_kd_value, dtype=float)
    pop_score = np.array(pop_score, dtype=float)

    def plot_grid(Z, method_name, out_name):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

        ax = axes[0]
        round_colors = {"R0": "#bdbdbd", "R3": "#74a9cf", "R4": "#08306b"}
        for label, color in round_colors.items():
            m = pop_round == label
            ax.scatter(Z[m, 0], Z[m, 1], s=30, alpha=1.0, color=color, edgecolors="black",
                       linewidths=0.4, label=f"{label} (n={m.sum()})")
        other = pop_round == None
        ax.scatter(Z[other, 0], Z[other, 1], s=10, alpha=0.15, color="lightgray", zorder=1)
        ax.set_title("colored by round (opaque)")
        ax.legend(fontsize=8, loc="best")
        ax.set_xlabel(f"{method_name}1"); ax.set_ylabel(f"{method_name}2")

        ax = axes[1]
        not_kd = np.isnan(pop_kd_value) & (pop_kd_binder == None)
        ax.scatter(Z[not_kd, 0], Z[not_kd, 1], s=10, alpha=0.1, color="lightgray", zorder=1)
        binder_mask = pop_kd_binder == True
        nb_mask = pop_kd_binder == False
        sc = ax.scatter(Z[binder_mask, 0], Z[binder_mask, 1], c=-np.log10(pop_kd_value[binder_mask]),
                         cmap="plasma_r", s=110, edgecolors="black", linewidths=0.8, zorder=5,
                         label="confirmed binder")
        ax.scatter(Z[nb_mask, 0], Z[nb_mask, 1], marker="X", s=60, color="dimgray",
                   edgecolors="black", linewidths=0.5, zorder=4, label="confirmed N.B.")
        ax.set_title("colored by KD value")
        ax.legend(fontsize=8, loc="best")
        fig.colorbar(sc, ax=ax, fraction=0.046, label="pKD = -log10(KD, M)")
        ax.set_xlabel(f"{method_name}1")

        ax = axes[2]
        other2 = np.isnan(pop_score)
        ax.scatter(Z[other2, 0], Z[other2, 1], s=10, alpha=0.1, color="lightgray", zorder=1)
        has_score = ~other2
        sc2 = ax.scatter(Z[has_score, 0], Z[has_score, 1], c=pop_score[has_score], cmap="viridis_r",
                          s=20, alpha=0.8, edgecolors="none")
        fig.colorbar(sc2, ax=ax, fraction=0.046, label="ProteinMPNN score\n(lower = more favorable)")
        ax.set_title(f"our noMHC designs (n={has_score.sum()}), colored by model score")
        ax.set_xlabel(f"{method_name}1")

        fig.suptitle(f"ProteinMPNN (noMHC) embedding, downsampled+balanced ({method_name}) -- "
                     f"native peptide: {NATIVE}", y=1.03, fontsize=13)
        fig.tight_layout()
        out = FIG_DIR / out_name
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}", flush=True)

    print("Running PCA...", flush=True)
    pca = PCA(n_components=2, random_state=0)
    Z_pca = pca.fit_transform(emb)
    print(f"  PCA var explained: {pca.explained_variance_ratio_}", flush=True)
    plot_grid(Z_pca, "PC", "fig_if18_kd_round_sample_pmpnn_pca.png")

    print("Running UMAP...", flush=True)
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=0, verbose=False)
    Z_umap = reducer.fit_transform(emb)
    plot_grid(Z_umap, "UMAP", "fig_if18_kd_round_sample_pmpnn_umap.png")

    print("DONE")


if __name__ == "__main__":
    main()
