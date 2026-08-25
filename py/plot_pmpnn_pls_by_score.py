#!/usr/bin/env python3
"""One point per peptide, projected via PLS regression (supervised -- explicitly finds the
direction(s) in embedding space that best track score) instead of PCA/UMAP (unsupervised, finds
max-variance directions that turned out NOT to align with score). Uses the cached 672 unique noMHC
2P5E design embeddings + scores.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, cross_val_predict
from scipy.stats import pearsonr

FIG_DIR = Path("/home/ubuntu/if-mhc/figures/fig_if19_pmpnn_score_embedding_regression")
d = np.load(FIG_DIR / "embeddings_and_scores.npz")
emb, scores = d["emb"], d["scores"]
print(f"n={len(scores)}, embedding dim={emb.shape[1]}")

pls = PLSRegression(n_components=2)
pls.fit(emb, scores)
Z = pls.transform(emb)
r1, p1 = pearsonr(Z[:, 0], scores)
r2, p2 = pearsonr(Z[:, 1], scores)
print(f"PLS component 1 vs score (in-sample): r={r1:.3f} (p={p1:.2e})")
print(f"PLS component 2 vs score (in-sample): r={r2:.3f} (p={p2:.2e})")

# honest check: cross-validated, so this isn't just PLS memorizing the training score
kf = KFold(n_splits=5, shuffle=True, random_state=0)
cv_pred = cross_val_predict(PLSRegression(n_components=2), emb, scores, cv=kf).ravel()
r_cv, p_cv = pearsonr(cv_pred, scores)
print(f"5-fold CV predicted score vs actual score: r={r_cv:.3f} (p={p_cv:.2e})")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
ax = axes[0]
sc = ax.scatter(Z[:, 0], Z[:, 1], c=scores, cmap="viridis_r", s=25, alpha=0.8, edgecolors="none")
ax.set_xlabel(f"PLS component 1 (r={r1:.2f} vs score)")
ax.set_ylabel(f"PLS component 2 (r={r2:.2f} vs score)")
ax.set_title(f"PLS projection of embedding (n={len(scores)}), colored by score")
fig.colorbar(sc, ax=ax, fraction=0.046, label="ProteinMPNN score\n(lower = more favorable)")

ax = axes[1]
ax.scatter(scores, cv_pred, s=15, alpha=0.5)
lims = [min(scores.min(), cv_pred.min()), max(scores.max(), cv_pred.max())]
ax.plot(lims, lims, "k--", linewidth=1)
ax.set_xlabel("actual score")
ax.set_ylabel("5-fold CV predicted score (PLS, 2 components)")
ax.set_title(f"honesty check: CV r={r_cv:.2f} (not just fit to training data)")

fig.suptitle("Supervised projection (PLS) -- explicitly aligned with score, not PCA/UMAP variance",
             y=1.02, fontsize=13)
fig.tight_layout()
out = FIG_DIR / "fig_if19_pmpnn_pls_by_score.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}")
