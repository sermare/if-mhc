#!/usr/bin/env python3
"""Emit notebooks/28_nyeso1_embedding_vs_inverse_folding_score.ipynb.

Inverse-folding scores for a 15k random subsample of the NY-ESO-1 selection peptides (rounds 2-4)
laid onto the low-LR CNN embedding, colored by score, with regressions asking whether the structural
score carries any signal about where a peptide sits in the selection.

Build + execute:
  /home/ubuntu/miniforge3/bin/python3 py/build_nyeso_embedding_score_notebook.py
  cd /home/ubuntu/if-mhc && /home/ubuntu/miniforge3/envs/esmcba/bin/jupyter nbconvert \
      --to notebook --execute --inplace notebooks/28_nyeso1_embedding_vs_inverse_folding_score.ipynb
"""
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook(); C = []
def md(t): C.append(nbf.v4.new_markdown_cell(t))
def co(t): C.append(nbf.v4.new_code_cell(t))

md(r"""# Inverse-folding score on the NY-ESO-1 selection embedding

Does a structure-based inverse-folding score know anything about where a peptide sits in a real
selection experiment?

**Peptides.** A 15,000-peptide subsample of the 100,000 NY-ESO-1 9-mers the low-LR CNN embedding
covers for the 1G4c58c61 TCR. The embedding's own peptide order is a uniform random draw
(`rng.choice(replace=False)`, seed 1) and is not sorted, so the first 15,000 is itself a valid random
subsample: round proportions match the full set to within 0.4 points (R2 51.1% vs 50.9%, R3 37.2% vs
37.6%, R4 11.6% vs 11.5%), giving 1,744 R4 peptides. These are peptides whose furthest observed round
is R2, R3, or R4 -- the file is `r2plus`, so R0 and R1 are not represented here.

**Embedding.** `nyeso1_NEWCNN_embedding_full5round_r2plus_1G4c58c61.npz`. NEWCNN is the low-LR model
(lr=1e-5, bs=256), a 5-seed ensemble whose 128-dim embeddings are concatenated to 640-dim, then
reduced to 2D by PCA and UMAP. The 1G4c58c61 file is the right one to pair with these scores because
that is the TCR in 2P5E, the structure the peptides were scored against.

**Score.** Mean per-residue negative log-likelihood over the 9 peptide positions on the 2P5E backbone
in full MHC+TCR context, lower = more favorable. Computed with a batched scorer
(`py/batch_score_peptides_mpnn.py`) validated to r=0.99 against ProteinMPNN's own `score_only`.

**The question.** Selection round and read count are experimental measures of enrichment. Hamming
distance to native is a sequence measure. If the inverse-folding score is informative about binding
rather than merely about sequence similarity, it should relate to round/count beyond what Hamming
already explains.""")

co(r"""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from scipy.stats import pearsonr, spearmanr, kruskal, mannwhitneyu, t as tdist

ROOT = Path("/home/ubuntu/if-mhc")
EMB = Path("/home/ubuntu/pmhc/modeling/ONG229/comparison/"
           "nyeso1_NEWCNN_embedding_full5round_r2plus_1G4c58c61.npz")
FIG_DIR = ROOT / "figures/fig_nyeso1_embedding_score"
FIG_DIR.mkdir(exist_ok=True, parents=True)

d = np.load(EMB, allow_pickle=True)
emb = pd.DataFrame({
    "peptide": [str(p) for p in d["train_peps"]],
    "round_cat": d["train_round_cat"],
    "median_count": d["train_median_count"],
    "hamming": d["train_hamming"],
    "pca1": d["train_pca"][:, 0], "pca2": d["train_pca"][:, 1],
    "umap1": d["train_umap"][:, 0], "umap2": d["train_umap"][:, 1],
})

# The index peptide SLLMWITQC is not part of the 100k selection sample, but the embedding file
# carries the KD panel projected into the same space, so it can still be located exactly.
_kd = [str(x) for x in d["kd_peps"]]
INDEX_PEP_NAME = "SLLMWITQC"
_i = _kd.index(INDEX_PEP_NAME)
INDEX_XY = {"umap": tuple(d["kd_umap"][_i]), "pca": tuple(d["kd_pca"][_i])}
print(f"index peptide {INDEX_PEP_NAME} at UMAP {np.round(INDEX_XY['umap'], 2)}, "
      f"PCA {np.round(INDEX_XY['pca'], 2)} (not in the 100k library sample)")

SCORES = [("vanilla", "ProteinMPNN"), ("nomhc", "noMHC ProteinMPNN"), ("ligandmpnn", "LigandMPNN")]
avail = []
for key, lab in SCORES:
    cands = [ROOT / f"outputs/analysis/nyeso1_r2plus_mpnn_{key}_scores_15k.npz",
             ROOT / f"outputs/analysis/nyeso1_r2plus_mpnn_{key}_scores.npz",
             ROOT / f"outputs/analysis/nyeso1_r2plus_{key}_scores_15k.npz",
             ROOT / f"outputs/analysis/nyeso1_r2plus_{key}_scores.npz"]
    f = next((c for c in cands if c.exists()), cands[0])
    if f.exists():
        z = np.load(f, allow_pickle=True)
        emb[f"score_{key}"] = pd.Series(
            dict(zip([str(p) for p in z["peptides"]], z["score"]))).reindex(emb.peptide).values
        avail.append((key, lab))
        print(f"loaded {lab}: {np.isfinite(emb[f'score_{key}']).sum():,} scores")
    else:
        print(f"MISSING {lab} ({f.name}) -- skipped")

score_cols = [f"score_{k}" for k, _ in avail]
if score_cols:
    keep = np.isfinite(emb[score_cols]).all(axis=1)
    print(f"\nrestricting to the {int(keep.sum()):,} peptides scored by all {len(avail)} model(s) "
          f"(of {len(emb):,} in the embedding)")
    emb = emb[keep].reset_index(drop=True)
print(f"{len(emb):,} peptides | rounds {dict(emb.round_cat.value_counts().sort_index())}")
emb.head()""")

md(r"""### Delta to the index peptide

Raw score is on an arbitrary scale, so a second view expresses every peptide relative to the index
peptide **SLLMWITQC** (the NY-ESO-1 sequence crystallized in 2P5E):

`delta = score(peptide) - score(SLLMWITQC)`

Negative means the model finds that peptide *more* favorable on this backbone than the native one,
positive means less. The index score is measured with 200 decoding orders rather than the 1 used for
the library, because it is subtracted from every peptide and its own noise would otherwise shift the
whole delta distribution.""")

co(r"""ref = np.load(ROOT / "outputs/analysis/nyeso1_index_peptide_reference.npz", allow_pickle=True)
INDEX_PEP = str(ref["peptide"])
index_score = {k: float(ref[f"score_{k}"]) for k, _ in avail if f"score_{k}" in ref}
print(f"index peptide {INDEX_PEP}, scored with n_orders={int(ref['n_orders'])}:")
for key, lab in avail:
    if key in index_score:
        emb[f"delta_{key}"] = emb[f"score_{key}"] - index_score[key]
        frac = float((emb[f"delta_{key}"] < 0).mean())
        print(f"  {lab:20s} index={index_score[key]:.4f} | "
              f"{100*frac:.1f}% of library scores better than index")
    else:
        print(f"  {lab:20s} no index reference -- delta skipped")""")

md(r"""## 1. Score distribution by selection round

The most direct question. If the score tracks enrichment, later rounds should score lower (more
favorable). Kruskal-Wallis across the three rounds, then pairwise R2 vs R4.""")

co(r"""for key, lab in avail:
    c = f"score_{key}"
    sub = emb[np.isfinite(emb[c])]
    groups = [sub.loc[sub.round_cat == r, c].values for r in sorted(sub.round_cat.unique())]
    H, p = kruskal(*groups)
    means = {int(r): round(float(sub.loc[sub.round_cat == r, c].mean()), 4)
             for r in sorted(sub.round_cat.unique())}
    r2, r4 = sub.loc[sub.round_cat == 2, c], sub.loc[sub.round_cat == 4, c]
    u, p24 = mannwhitneyu(r2, r4, alternative="two-sided")
    d_cohen = (r4.mean() - r2.mean()) / np.sqrt((r2.var() + r4.var()) / 2)
    print(f"{lab}:")
    print(f"   mean by round {means}")
    print(f"   Kruskal-Wallis H={H:.1f} p={p:.2e} | R2 vs R4 MW p={p24:.2e}, Cohen d={d_cohen:+.3f}")""")

md(r"""Note on interpretation: at n in the tens of thousands almost any difference clears
significance, so the effect size (Cohen's d) matters far more than the p-value. A |d| below ~0.1 is
negligible however small p is.""")

md(r"""## 2. Correlation with read count and with distance to native

`median_count` is the enrichment magnitude; `hamming` is the sequence-similarity control.""")

co(r"""rows = []
for key, lab in avail:
    c = f"score_{key}"
    sub = emb[np.isfinite(emb[c])]
    for target in ["median_count", "hamming", "round_cat"]:
        r, p = pearsonr(sub[c], sub[target]); rho, ps = spearmanr(sub[c], sub[target])
        rows.append({"model": lab, "vs": target, "pearson_r": round(r, 4),
                     "spearman_rho": round(rho, 4), "p": f"{ps:.1e}", "n": len(sub)})
pd.DataFrame(rows)""")

md(r"""## 3. Regression: does the score add anything beyond Hamming distance?

The score is expected to correlate with distance to native simply because it is likelihood-like. The
question that matters is whether it explains selection round *after* Hamming is accounted for. Two
nested OLS models, compared by adjusted R-squared and by the score's partial contribution.""")

co(r"""def ols(y, Xcols):
    '''Plain-numpy OLS so this notebook runs on the project's default kernel without statsmodels.
    Returns R^2, coefficients, and two-sided p-values (t-test on beta/se).'''
    X = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in Xcols])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    rss = float(resid @ resid)
    tss = float(((y - y.mean()) ** 2).sum())
    n, k = X.shape
    sigma2 = rss / (n - k)
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    tvals = beta / se
    pvals = 2 * tdist.sf(np.abs(tvals), df=n - k)
    return 1 - rss / tss, beta, pvals

for key, lab in avail:
    c = f"score_{key}"
    sub = emb[np.isfinite(emb[c])]
    y = sub.round_cat.astype(float).values
    r2_0, _, _ = ols(y, [sub.hamming])
    r2_1, beta, pv = ols(y, [sub.hamming, sub[c]])
    print(f"{lab}:")
    print(f"   hamming only        R2={r2_0:.5f}")
    print(f"   hamming + score     R2={r2_1:.5f}   (delta R2 = {r2_1 - r2_0:+.5f})")
    print(f"   score coefficient   beta={beta[2]:+.4f}  p={pv[2]:.2e}")""")

md(r"""## 4. The embedding, colored by score

Same 2D layout the CNN produces, recolored by the structural score. If the two views agree, regions
of the embedding should show coherent color structure rather than noise.""")

co(r"""n_show = 30000
rng = np.random.RandomState(0)
idx = rng.choice(len(emb), size=min(n_show, len(emb)), replace=False)
sub = emb.iloc[idx]
PROJ = [("umap1", "umap2", "UMAP"), ("pca1", "pca2", "PCA")]

ROUND_COLORS = ["#4C72B0", "#DD8452", "#55A868"]   # R2, R3, R4


def draw(ax, sub, xc, yc, values, cmap, label, categorical=False, diverging=False):
    '''One embedding panel. Categorical uses discrete colors + a tick-per-class colorbar;
    continuous clips to the 2nd/98th percentile so a few outliers do not flatten the scale.'''
    if categorical:
        cats = sorted(pd.unique(values))
        cmap_o = mpl.colors.ListedColormap(ROUND_COLORS[:len(cats)])
        norm = mpl.colors.BoundaryNorm(np.arange(len(cats) + 1) - 0.5, len(cats))
        codes = pd.Categorical(values, categories=cats).codes
        sc = ax.scatter(sub[xc], sub[yc], c=codes, cmap=cmap_o, norm=norm, s=3, alpha=0.6)
        cb = plt.colorbar(sc, ax=ax, ticks=range(len(cats)))
        cb.ax.set_yticklabels([f"R{int(c)}" for c in cats])
    else:
        if diverging:
            lim = float(np.nanpercentile(np.abs(values), 98))
            vmin, vmax = -lim, lim
        else:
            vmin, vmax = np.nanpercentile(values, [2, 98])
        sc = ax.scatter(sub[xc], sub[yc], c=values, cmap=cmap, s=3, alpha=0.6,
                        vmin=vmin, vmax=vmax)
        cb = plt.colorbar(sc, ax=ax)
    cb.set_label(label, fontsize=9)
    ix, iy = INDEX_XY["umap" if xc.startswith("umap") else "pca"]
    ax.scatter([ix], [iy], marker="*", s=420, c="white", edgecolors="black", linewidths=1.6,
               zorder=10, label=f"index {INDEX_PEP_NAME}")
    # place the label inward: the index sits at the extreme edge in both projections, so a fixed
    # rightward offset would run under the colorbar in UMAP
    x0, x1 = ax.get_xlim()
    to_left = ix > x0 + 0.72 * (x1 - x0)
    ax.annotate(INDEX_PEP_NAME, (ix, iy), fontsize=9, fontweight="bold", zorder=11,
                xytext=(-12 if to_left else 12, -15), textcoords="offset points",
                ha="right" if to_left else "left",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="none"))
    ax.set_xlabel(f"{xc[:-1].upper()} 1"); ax.set_ylabel(f"{xc[:-1].upper()} 2")


def embedding_grid(rows, title, outname):
    '''rows: list of (column_or_series, cmap, colorbar label, row title, kwargs).'''
    fig, axes = plt.subplots(len(rows), len(PROJ), figsize=(6.2 * len(PROJ), 4.9 * len(rows)),
                             squeeze=False)
    for i, (vals, cmap, lab, rtitle, kw) in enumerate(rows):
        for j, (xc, yc, proj) in enumerate(PROJ):
            ax = axes[i, j]
            draw(ax, sub, xc, yc, vals, cmap, lab, **kw)
            ax.set_title(f"{proj}: {rtitle}", fontsize=12)
    fig.suptitle(title, fontsize=14, y=1.0)
    fig.tight_layout()
    out = FIG_DIR / outname
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.show()
    print(f"wrote {out}")


rows = []
for key, lab in avail:
    rows.append((sub[f"score_{key}"], "magma_r", f"{lab} score\n(lower = more favorable)",
                 f"{lab} score", {}))
rows.append((sub["hamming"], "viridis", "Hamming distance to SLLMWITQC",
             "Hamming to index", {}))
rows.append((sub["round_cat"], None, "furthest selection round",
             "selection round", {"categorical": True}))

embedding_grid(rows,
               "NY-ESO-1 selection embedding (low-LR CNN, 1G4c58c61) colored by inverse-folding "
               f"score, Hamming distance, and round\n{len(sub):,} of {len(emb):,} peptides shown",
               "fig_nyeso1_embedding_colored_by_score.png")""")

md(r"""### Same embedding, colored by delta to the index peptide

Diverging colormap centered at zero: blue = scores better than native SLLMWITQC, red = worse. This
makes the native-relative structure visible in a way the raw score cannot, since zero is now a
meaningful reference rather than an arbitrary point on the scale.""")

co(r"""dcols = [(k, l) for k, l in avail if f"delta_{k}" in emb.columns]
if dcols:
    rows = []
    for key, lab in dcols:
        rows.append((sub[f"delta_{key}"], "coolwarm",
                     f"{lab} score - index\n(blue = better than SLLMWITQC)",
                     f"{lab} delta to index", {"diverging": True}))
    rows.append((sub["hamming"], "viridis", "Hamming distance to SLLMWITQC",
                 "Hamming to index", {}))
    rows.append((sub["round_cat"], None, "furthest selection round",
                 "selection round", {"categorical": True}))
    embedding_grid(rows,
                   "NY-ESO-1 selection embedding colored by score relative to the index peptide "
                   f"SLLMWITQC, with Hamming and round\n{len(sub):,} peptides shown",
                   "fig_nyeso1_embedding_colored_by_delta_to_index.png")
else:
    print("no delta columns available")""")

md(r"""## 5. Score by round, distribution view

The violin makes the effect size visible in a way the p-value does not.""")

co(r"""fig, axes = plt.subplots(1, len(avail), figsize=(5.6 * len(avail), 4.8), squeeze=False)
for j, (key, lab) in enumerate(avail):
    c = f"score_{key}"
    ax = axes[0, j]
    sub = emb[np.isfinite(emb[c])]
    rounds = sorted(sub.round_cat.unique())
    data = [sub.loc[sub.round_cat == r, c].values for r in rounds]
    parts = ax.violinplot(data, positions=range(len(rounds)), showmeans=True, showextrema=False,
                          widths=0.8)
    for pc in parts["bodies"]:
        pc.set_facecolor("#0072B2"); pc.set_alpha(0.4)
    if "cmeans" in parts:
        parts["cmeans"].set_color("#333333")
    ax.set_xticks(range(len(rounds)))
    ax.set_xticklabels([f"R{int(r)}\n(n={len(g):,})" for r, g in zip(rounds, data)])
    ax.set_ylabel("score (lower = more favorable)")
    ax.set_title(lab, fontsize=12)
fig.suptitle("Inverse-folding score by furthest selection round", fontsize=13)
fig.tight_layout()
out = FIG_DIR / "fig_nyeso1_score_by_round.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
plt.show()
print(f"wrote {out}")""")

md(r"""## 6. Top-scoring peptides and their final-round counts

The analyses above are population-level. This asks the practical question instead: if you took the
peptides the model likes most and looked them up in the actual selection, would they be enriched in
the final round? Per-round counts come from the raw ONG229 count table, joined back by sequence.""")

co(r"""COUNTS = Path("/home/ubuntu/pmhc/modeling/work/full_5round/"
              "ONG229_1G4c58c61_peptide_counts.csv")
cnt = pd.read_csv(COUNTS, usecols=["Peptide", "R0", "R1", "R2", "R3", "R4"])
emb = emb.merge(cnt, left_on="peptide", right_on="Peptide", how="left").drop(columns="Peptide")
print(f"joined per-round counts | missing R4: {int(emb.R4.isna().sum())}")

TOP_N = 100
for key, lab in avail:
    c = f"score_{key}"
    top = emb.nsmallest(TOP_N, c)
    rest = emb.drop(top.index)
    print(f"\n{lab} -- top {TOP_N} by score (best = lowest):")
    print(f"   score range      {top[c].min():.3f} to {top[c].max():.3f}")
    print(f"   R4 count         mean={top.R4.mean():.1f}  median={top.R4.median():.0f}  "
          f"max={top.R4.max():.0f}")
    print(f"   reached R4       {int((top.R4 > 0).sum())}/{TOP_N} ({100*(top.R4 > 0).mean():.0f}%)"
          f"   vs rest {100*(rest.R4 > 0).mean():.1f}%  "
          f"({(top.R4 > 0).mean() / (rest.R4 > 0).mean():.1f}x enrichment)")""")

co(r"""fig, axes = plt.subplots(2, len(avail), figsize=(7.2 * len(avail), 10), squeeze=False)

for j, (key, lab) in enumerate(avail):
    c = f"score_{key}"
    top = emb.nsmallest(TOP_N, c).copy()
    rest = emb.drop(top.index)

    # (a) score vs final-round count for the top N
    ax = axes[0, j]
    sc = ax.scatter(top[c], np.log10(top.R4 + 1), c=top.hamming, cmap="viridis",
                    s=70, edgecolors="black", linewidths=0.5, zorder=5)
    cb = plt.colorbar(sc, ax=ax); cb.set_label("Hamming to SLLMWITQC", fontsize=9)
    r, pv = spearmanr(top[c], top.R4)
    ax.set_xlabel(f"{lab} score (lower = more favorable)")
    ax.set_ylabel("log10(R4 count + 1)")
    ax.set_title(f"{lab}: top {TOP_N} peptides\nSpearman rho={r:+.2f} (p={pv:.2f}) within the top {TOP_N}",
                 fontsize=11)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.annotate(f"{int((top.R4 == 0).sum())} of {TOP_N} never reached R4",
                xy=(0.02, 0.94), xycoords="axes fraction", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="grey"))

    # (b) does the top set reach R4 more often than the rest?
    ax = axes[1, j]
    fr = [100 * (top.R4 > 0).mean(), 100 * (rest.R4 > 0).mean()]
    bars = ax.bar([f"top {TOP_N}", f"rest (n={len(rest):,})"], fr,
                  color=["#CC79A7", "#999999"], edgecolor="black")
    for b, v in zip(bars, fr):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%", ha="center", fontsize=11)
    ax.set_ylabel("% of peptides with any R4 read")
    ax.set_ylim(0, max(fr) * 1.25)
    ax.set_title(f"{lab}: reaching the final round", fontsize=11)

fig.suptitle(f"Do the peptides ProteinMPNN scores best actually survive to the final selection round?\n"
             f"top {TOP_N} of {len(emb):,}", fontsize=13, y=1.0)
fig.tight_layout()
out = FIG_DIR / "fig_nyeso1_top100_score_vs_final_round_counts.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
plt.show()
print(f"wrote {out}")""")

md(r"""## 6. Reading the result

Judge by the numbers printed above, with two cautions:

1. **At this n, p-values are nearly meaningless.** Use Cohen's d and delta-R-squared. A score that
   is "highly significant" but moves round assignment by d = 0.02 is not a usable signal.
2. **Hamming distance is the confound throughout.** The inverse-folding score is likelihood-like, so
   it partly reports how native-like a sequence is; selection rounds also enrich near-native
   sequences. The nested regression in section 3 is the part that separates these.""")

nb["cells"] = C
out_nb = Path("/home/ubuntu/if-mhc/notebooks/28_nyeso1_embedding_vs_inverse_folding_score.ipynb")
out_nb.parent.mkdir(exist_ok=True, parents=True)
nbf.write(nb, str(out_nb))
print(f"wrote {out_nb}")
