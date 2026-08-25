#!/usr/bin/env python3
"""Sample real NY-ESO-1/1G4c58c61 NGS peptides across rounds, embed via the CNN model (same
128-dim penultimate-layer embedding as fig_if13/14/15), fit PCA and UMAP standalone on this sample,
and produce 2 figures (1 row x 3 columns each): colored by furthest round reached, by log1p(median
count across rounds), and by Hamming distance to the native peptide.

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
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn.decomposition import PCA
import umap

sys.path.insert(0, "/home/ubuntu/pmhc/modeling/ONG229/py")
import nyeso1_ranking_lib as lib

ROOT = Path("/home/ubuntu/if-mhc")
FIG_DIR = ROOT / "figures/fig_if16_round_sample_pca_umap"
FIG_DIR.mkdir(exist_ok=True, parents=True)
RUNS_DIR = Path("/home/ubuntu/pmhc/modeling/ONG229/lm_runs")

DEV = torch.device("cuda")
AA_IDX = {a: i for i, a in enumerate(lib.AA_LIST)}
NATIVE = "SLLMWITQC"
TCR = "1G4c58c61"
BEST_SEED = 17
N_SAMPLE = 100_000
SEED = 0
ROUND_LABELS = ["R0 only", "R3", "R4+"]


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
def get_embeddings(model, peptides, batch_size=8192):
    model.eval()
    out = []
    for i in range(0, len(peptides), batch_size):
        X = torch.from_numpy(encode_peptides(peptides[i:i + batch_size])).to(DEV)
        out.append(model.embedding(X).cpu().numpy())
    return np.concatenate(out, axis=0)


def hamming(pep, native=NATIVE):
    return sum(a != b for a, b in zip(pep, native))


def furthest_round(df):
    counts = df[lib.ROUND_COLS].values.astype(np.float64)
    out = np.zeros(len(df), dtype=int)
    for i, col in enumerate(lib.ROUND_COLS):
        out[counts[:, i] > 0] = i
    return out


def main():
    print("Loading + sampling real NGS library...", flush=True)
    tab_data = lib.load_tab_data()
    df = tab_data[f"NYESO1__{TCR}"]
    rng = np.random.RandomState(SEED)
    n_sample = min(N_SAMPLE, len(df))
    sample_df = df.iloc[rng.choice(len(df), size=n_sample, replace=False)].copy()

    peptides = sample_df["Peptide"].values
    fr = furthest_round(sample_df)
    median_count = np.median(sample_df[lib.ROUND_COLS].values.astype(np.float64), axis=1)
    hd = np.array([hamming(p) for p in peptides])

    model = CNNRanker(9).to(DEV)
    ckpt = RUNS_DIR / f"nyeso1_reduced_sweep_countweighted_seed{BEST_SEED}_{TCR}" / "model_best.pt"
    load_state_dict_remapped(model, ckpt)
    emb = get_embeddings(model, peptides)
    print(f"n={len(peptides):,}, embedding shape={emb.shape}", flush=True)

    def plot_grid(Z, method_name, out_name):
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

        fig.suptitle(f"NY-ESO-1/1G4c58c61 real NGS sample, CNN embedding ({method_name}) -- "
                     f"native peptide: {NATIVE} (n={len(peptides):,})", y=1.03, fontsize=13)
        fig.tight_layout()
        out = FIG_DIR / out_name
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}", flush=True)

    print("Running PCA...", flush=True)
    pca = PCA(n_components=2, random_state=0)
    Z_pca = pca.fit_transform(emb)
    print(f"  PCA var explained: {pca.explained_variance_ratio_}", flush=True)
    plot_grid(Z_pca, "PC", "fig_if16_round_sample_pca.png")

    print("Running UMAP (slower)...", flush=True)
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=0, verbose=False)
    Z_umap = reducer.fit_transform(emb)
    plot_grid(Z_umap, "UMAP", "fig_if16_round_sample_umap.png")

    print("DONE")


if __name__ == "__main__":
    main()
