#!/usr/bin/env python3
"""PCA and UMAP of JUST the 51 real KD-tested NY-ESO-1/1G4c58c61 peptides (not projected into the
big background library space like fig_if13 -- fit standalone, directly on these 51 points), using
the same 1G4c58c61 CNN embedding (128-dim penultimate layer) as fig_if13/fig_if14.

Must run under `protenix` (torch+umap+sklearn+GPU).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap

ROOT = Path("/home/ubuntu/if-mhc")
FIG_DIR = ROOT / "figures/fig_if15_kd_peptides_pca_umap"
FIG_DIR.mkdir(exist_ok=True, parents=True)
RUNS_DIR = Path("/home/ubuntu/pmhc/modeling/ONG229/lm_runs")

DEV = torch.device("cuda")
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_IDX = {a: i for i, a in enumerate(AA_LIST)}
NATIVE = "SLLMWITQC"
TCR = "1G4c58c61"
BEST_SEED = 17


class CNNRanker(nn.Module):
    def __init__(self, seq_len, n_aa=20, embed_dim=32, channels=64):
        super().__init__()
        self.embed = nn.Embedding(n_aa, embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.ReLU()
        self.head_fc1 = nn.Linear(seq_len * channels, 128)
        self.head_relu = nn.ReLU()
        self.head_dropout = nn.Dropout(0.1)
        self.head_fc2 = nn.Linear(128, 1)

    def embedding(self, x):
        e = self.embed(x).transpose(1, 2)
        h = self.act(self.conv1(e))
        h = self.act(self.conv2(h))
        flat = h.transpose(1, 2).reshape(h.size(0), -1)
        return self.head_relu(self.head_fc1(flat))

    def forward(self, x):
        return self.head_fc2(self.head_dropout(self.embedding(x))).squeeze(-1)


def load_state_dict_remapped(model, ckpt_path):
    sd = torch.load(ckpt_path, map_location=DEV)
    remap = {"head.0.weight": "head_fc1.weight", "head.0.bias": "head_fc1.bias",
              "head.3.weight": "head_fc2.weight", "head.3.bias": "head_fc2.bias"}
    model.load_state_dict({remap.get(k, k): v for k, v in sd.items()})


def encode_peptides(peptides):
    return np.array([[AA_IDX[c] for c in p] for p in peptides], dtype=np.int64)


@torch.no_grad()
def get_embeddings(model, peptides):
    model.eval()
    X = torch.from_numpy(encode_peptides(peptides)).to(DEV)
    return model.embedding(X).cpu().numpy()


def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))


def parse_kd(val):
    if pd.isna(val) or val == "N.B.":
        return (val != "N.B."), np.nan
    try:
        return True, float(val)
    except ValueError:
        return None, np.nan


def main():
    kin = pd.read_csv("/tmp/kd_peptides.csv")
    parsed = kin["KD_raw"].apply(parse_kd)
    kin["is_binder"] = parsed.apply(lambda t: t[0])
    kin["kd_value"] = parsed.apply(lambda t: t[1])
    kin["hamming_to_native"] = kin["Peptide"].apply(lambda p: hamming(p, NATIVE))
    peptides = kin["Peptide"].tolist()

    model = CNNRanker(9).to(DEV)
    ckpt = RUNS_DIR / f"nyeso1_reduced_sweep_countweighted_seed{BEST_SEED}_{TCR}" / "model_best.pt"
    load_state_dict_remapped(model, ckpt)
    emb = get_embeddings(model, peptides)
    print(f"embedded {len(peptides)} KD-tested peptides, shape={emb.shape}")

    pca = PCA(n_components=2, random_state=0)
    Z_pca = pca.fit_transform(emb)
    print(f"PCA var explained: {pca.explained_variance_ratio_}")

    reducer = umap.UMAP(n_components=2, n_neighbors=10, min_dist=0.1, random_state=0, verbose=False)
    Z_umap = reducer.fit_transform(emb)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for ax, Z, method in [(axes[0], Z_pca, "PC"), (axes[1], Z_umap, "UMAP")]:
        binder_mask = kin["is_binder"].values == True
        nb_mask = kin["is_binder"].values == False
        sc = ax.scatter(Z[binder_mask, 0], Z[binder_mask, 1], c=-np.log10(kin["kd_value"].values[binder_mask]),
                         cmap="plasma_r", s=140, edgecolors="black", linewidths=0.8, zorder=5,
                         label="confirmed binder")
        ax.scatter(Z[nb_mask, 0], Z[nb_mask, 1], marker="X", s=70, color="dimgray",
                   edgecolors="black", linewidths=0.5, zorder=4, label="confirmed N.B.")
        for i, pep in enumerate(peptides):
            if binder_mask[i]:
                ax.annotate(pep, (Z[i, 0], Z[i, 1]), fontsize=7.5, xytext=(5, 5), textcoords="offset points")
        ax.set_title(f"{method} of 51 KD-tested peptides (standalone fit)", fontsize=11)
        ax.set_xlabel(f"{method}1"); ax.set_ylabel(f"{method}2")
        ax.legend(fontsize=8, loc="best")
        fig.colorbar(sc, ax=ax, fraction=0.046, label="pKD = -log10(KD, M)  [higher = stronger]")
    fig.suptitle("NY-ESO-1/1G4c58c61 CNN embedding: the 51 KD-tested peptides only "
                 f"(native: {NATIVE})", y=1.03, fontsize=13)
    fig.tight_layout()
    out = FIG_DIR / "fig_if15_kd_peptides_pca_umap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
