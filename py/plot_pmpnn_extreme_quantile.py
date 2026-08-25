#!/usr/bin/env python3
"""Instead of trying to show a continuous score gradient across all 672 designs, isolate the
EXTREMES -- best-scoring 20% vs worst-scoring 20% -- and see if those two groups separate more
cleanly than the full continuous range does. Uses the cached pooled 128-dim embeddings."""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import umap

FIG_DIR = Path("/home/ubuntu/if-mhc/figures/fig_if19_pmpnn_score_embedding_regression")
d = np.load(FIG_DIR / "embeddings_and_scores.npz")
emb, scores = d["emb"], d["scores"]
n = len(scores)

q_lo, q_hi = np.percentile(scores, [20, 80])
best_mask = scores <= q_lo   # lower score = more favorable
worst_mask = scores >= q_hi
print(f"n={n}, best (score<={q_lo:.3f}): {best_mask.sum()}, worst (score>={q_hi:.3f}): {worst_mask.sum()}")

sub_mask = best_mask | worst_mask
emb_sub = emb[sub_mask]
labels_sub = np.where(best_mask[sub_mask], "best 20%", "worst 20%")

pca = PCA(n_components=2, random_state=0)
Z_pca = pca.fit_transform(emb_sub)

lda = LinearDiscriminantAnalysis(n_components=1)
Z_lda = lda.fit_transform(emb_sub, labels_sub)
acc = lda.score(emb_sub, labels_sub)
print(f"LDA (best vs worst 20%) in-sample separation accuracy: {acc:.3f}")

from sklearn.model_selection import cross_val_score
cv_acc = cross_val_score(LinearDiscriminantAnalysis(), emb_sub, labels_sub, cv=5).mean()
print(f"LDA 5-fold CV accuracy: {cv_acc:.3f} (chance = 0.50)")

reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=0, verbose=False)
Z_umap = reducer.fit_transform(emb_sub)

fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
colors = {"best 20%": "#1a9850", "worst 20%": "#d73027"}
for ax, Z, title in [(axes[0], Z_pca, "PCA"), (axes[1], Z_umap, "UMAP")]:
    for label, color in colors.items():
        m = labels_sub == label
        ax.scatter(Z[m, 0], Z[m, 1], s=25, alpha=0.7, color=color, label=f"{label} (n={m.sum()})")
    ax.set_title(f"{title}: best vs worst 20% by score")
    ax.legend(fontsize=9)
    ax.set_xlabel(f"{title}1"); ax.set_ylabel(f"{title}2")

ax = axes[2]
for label, color in colors.items():
    m = labels_sub == label
    ax.hist(Z_lda[m, 0], bins=20, alpha=0.6, color=color, label=f"{label} (n={m.sum()})")
ax.set_title(f"LDA axis (CV accuracy={cv_acc:.2f}, chance=0.50)")
ax.set_xlabel("LDA component 1"); ax.legend(fontsize=9)

fig.suptitle(f"Best 20% vs worst 20% scoring designs (score<={q_lo:.2f} vs >={q_hi:.2f})", y=1.02, fontsize=13)
fig.tight_layout()
out = FIG_DIR / "fig_if19_pmpnn_extreme_quantile.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}")
