#!/usr/bin/env python3
"""Just the PCA and UMAP of the cached noMHC ProteinMPNN embeddings (672 unique 2P5E designs),
colored by the model's own score."""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap

FIG_DIR = Path("/home/ubuntu/if-mhc/figures/fig_if19_pmpnn_score_embedding_regression")
d = np.load(FIG_DIR / "embeddings_and_scores.npz")
emb, scores = d["emb"], d["scores"]

pca = PCA(n_components=2, random_state=0)
Z_pca = pca.fit_transform(emb)
reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=0, verbose=False)
Z_umap = reducer.fit_transform(emb)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, Z, method in [(axes[0], Z_pca, "PC"), (axes[1], Z_umap, "UMAP")]:
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=scores, cmap="viridis_r", s=20, alpha=0.8, edgecolors="none")
    ax.set_xlabel(f"{method}1"); ax.set_ylabel(f"{method}2")
    ax.set_title(f"{method} of noMHC ProteinMPNN embedding (n={len(scores)})")
    fig.colorbar(sc, ax=ax, fraction=0.046, label="ProteinMPNN score\n(lower = more favorable)")
print(f"PCA var explained: {pca.explained_variance_ratio_}")
fig.suptitle("noMHC ProteinMPNN embedding, colored by score", y=1.02, fontsize=13)
fig.tight_layout()
out = FIG_DIR / "fig_if19_pmpnn_embedding_pca_umap.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}")
