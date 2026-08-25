#!/usr/bin/env python3
"""Emit notebooks/27_kd_peptide_inverse_folding_scores.ipynb.

Three inverse-folding models scored against 51 NY-ESO-1/1G4c58c61 peptides with measured KD:
ProteinMPNN, noMHC ProteinMPNN, and LigandMPNN, all on the 2P5E backbone in full MHC+TCR context.

Build + execute:
  /home/ubuntu/miniforge3/bin/python3 py/build_kd_score_notebook.py
  cd /home/ubuntu/if-mhc && /home/ubuntu/miniforge3/envs/esmcba/bin/jupyter nbconvert \
      --to notebook --execute --inplace notebooks/27_kd_peptide_inverse_folding_scores.ipynb
"""
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook(); C = []
def md(t): C.append(nbf.v4.new_markdown_cell(t))
def co(t): C.append(nbf.v4.new_code_cell(t))

md(r"""# Inverse-folding scores vs. measured KD, 51 NY-ESO-1 peptides

Three inverse-folding models score the same 51 peptides on the same backbone, and we ask whether any
of them recovers the experimentally measured binding outcome.

**Peptides.** 51 real, KD-tested peptides for the 1G4c58c61 TCR against NY-ESO-1 (native SLLMWITQC).
8 have a measured dissociation constant; the other 43 are confirmed non-binders (`N.B.`), so there is
no KD to regress against for those -- they only support a binder / non-binder contrast.

**Structure and context.** All scoring is on 2P5E (the 1G4c58c61 / NY-ESO-1 complex), full context:
MHC + beta-2m + TCR alpha/beta held fixed, peptide chain C scored.

**Score.** Every model reports the same quantity, mean per-residue negative log-likelihood over the
9 peptide positions, so lower = the model finds that sequence more favorable on this backbone. The
three are directly comparable in scale and direction.

**One methodological difference worth stating.** ProteinMPNN's `score_only` accepts a fasta, so the
two ProteinMPNN-series runs simply supply each sequence. LigandMPNN's `score.py` has no fasta input
-- it scores whatever sequence is in the PDB -- so each peptide was threaded onto the chain-C
backbone first (`py/thread_kd_peptides_pdb.py`), keeping backbone N/CA/C/O only (the crystal side
chains belong to the native peptide and would be wrong for a mutant) and taking altloc A at the two
disordered positions. Context chains pass through untouched.""")

co(r"""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import itertools
from scipy.stats import pearsonr, spearmanr, mannwhitneyu
from numpy.linalg import lstsq

ROOT = Path("/home/ubuntu/if-mhc")
FIG_DIR = ROOT / "figures/fig_kd_inverse_folding_scores"
FIG_DIR.mkdir(exist_ok=True, parents=True)

MODELS = [("score_vanilla", "ProteinMPNN", "#0072B2"),
          ("score_nomhc", "noMHC ProteinMPNN", "#E69F00"),
          ("score_ligandmpnn", "LigandMPNN", "#CC79A7")]

df = pd.read_csv(ROOT / "outputs/analysis/kd_score_correlation_3model.csv")
df = df.dropna(subset=[c for c, _, _ in MODELS]).reset_index(drop=True)
print(f"{len(df)} peptides | binders={int(df.is_binder.sum())} | "
      f"with numeric KD={int(df.kd_value.notna().sum())}")
df[["Peptide", "KD_raw", "is_binder", "hamming_to_native",
    "score_vanilla", "score_nomhc", "score_ligandmpnn"]].head(10)""")

md(r"""## 1. Score scales

All three land in the same range, which is the precondition for comparing them at all.""")

co(r"""rows = []
for c, lab, _ in MODELS:
    rows.append({"model": lab, "mean": df[c].mean(), "sd": df[c].std(),
                 "min": df[c].min(), "max": df[c].max()})
pd.DataFrame(rows).round(3)""")

md(r"""## 2. Do the three models agree with each other?

If they disagree outright, there is no single "inverse-folding view" of these peptides to talk about.""")

co(r"""print("pairwise Pearson r on the same 51 peptides:")
for i in range(len(MODELS)):
    for j in range(i + 1, len(MODELS)):
        a, la, _ = MODELS[i]; b, lb, _ = MODELS[j]
        r, p = pearsonr(df[a], df[b])
        print(f"  {la:20s} vs {lb:20s} r={r:+.3f} (p={p:.2e})")""")

md(r"""## 3. Binders vs. non-binders

The primary test. An informative score would put binders **lower** (more favorable).""")

co(r"""for c, lab, _ in MODELS:
    b = df.loc[df.is_binder, c]; nb = df.loc[~df.is_binder, c]
    u, p = mannwhitneyu(b, nb, alternative="two-sided")
    star = "*" if p < 0.05 else " "
    print(f"{star} {lab:20s} binders(n={len(b)})={b.mean():.3f}  "
          f"non-binders(n={len(nb)})={nb.mean():.3f}  diff={b.mean()-nb.mean():+.3f}  MW p={p:.3f}")""")

md(r"""## 4. Among the 8 measured binders, does the score track KD magnitude?

Separating binders from non-binders and *ranking* affinity are different problems. n=8 here, so this
is badly underpowered and reported for completeness rather than as a test.""")

co(r"""s = df[df.kd_value.notna()].copy()
s["logKD"] = np.log10(s.kd_value)
for c, lab, _ in MODELS:
    r, p = pearsonr(s[c], s.logKD); rho, ps = spearmanr(s[c], s.logKD)
    print(f"  {lab:20s} Pearson r={r:+.3f} (p={p:.3f})  Spearman rho={rho:+.3f} (p={ps:.3f})  n={len(s)}")""")

md(r"""## 5. The confound: distance to native

These scores are sequence-likelihood-like, so they may simply be reporting how far a peptide is from
the native SLLMWITQC rather than anything about binding. Two checks: does the score track Hamming
distance, and are binders closer to native than non-binders in the first place?""")

co(r"""print("score vs. Hamming distance to native:")
for c, lab, _ in MODELS:
    r, p = pearsonr(df[c], df.hamming_to_native)
    star = "*" if p < 0.05 else " "
    print(f"{star} {lab:20s} r={r:+.3f} (p={p:.3f})")

b = df.loc[df.is_binder, "hamming_to_native"]; nb = df.loc[~df.is_binder, "hamming_to_native"]
u, p = mannwhitneyu(b, nb, alternative="two-sided")
print(f"\nbinders are closer to native: {b.mean():.2f} vs {nb.mean():.2f} (MW p={p:.4f})")
print(f"  binder Hamming values: {sorted(b.tolist())}")""")

md(r"""### 5a. Does the binder signal survive controlling for distance?

Two ways to ask, because with n=8 they can disagree: a partial correlation over all 51, and a
within-stratum comparison restricted to Hamming 6-8 where both classes actually overlap.""")

co(r"""df["y"] = df.is_binder.astype(float)
X = np.c_[np.ones(len(df)), df.hamming_to_native]
print("partial r(score, binder | hamming), all 51:")
for c, lab, _ in MODELS:
    rs = df[c] - X @ lstsq(X, df[c], rcond=None)[0]
    ry = df.y - X @ lstsq(X, df.y, rcond=None)[0]
    r, p = pearsonr(rs, ry)
    star = "*" if p < 0.05 else " "
    print(f"{star} {lab:20s} r={r:+.3f} (p={p:.4f})")

st = df[(df.hamming_to_native >= 6) & (df.hamming_to_native <= 8)]
print(f"\nwithin Hamming 6-8 stratum (n={len(st)}):")
for c, lab, _ in MODELS:
    bb = st.loc[st.is_binder, c]; nn = st.loc[~st.is_binder, c]
    u, p = mannwhitneyu(bb, nn, alternative="two-sided")
    star = "*" if p < 0.05 else " "
    print(f"{star} {lab:20s} binders(n={len(bb)})={bb.mean():.3f}  "
          f"non(n={len(nn)})={nn.mean():.3f}  diff={bb.mean()-nn.mean():+.3f}  MW p={p:.3f}")""")

md(r"""## 6. Figure

Left column: score distribution split by binder status. Right column: score vs. pKD for the 8
measured binders, colored by Hamming distance to native, with non-binders shown as a rug so the
overlap between the two classes stays visible.""")

co(r"""fig, axes = plt.subplots(3, 2, figsize=(13, 13))
for row, (c, lab, col) in enumerate(MODELS):
    ax = axes[row, 0]
    b = df.loc[df.is_binder, c]; nb = df.loc[~df.is_binder, c]
    parts = ax.violinplot([nb.values, b.values], positions=[0, 1], showmeans=True, showextrema=False,
                          widths=0.75)
    for pc in parts["bodies"]:
        pc.set_facecolor(col); pc.set_alpha(0.35)
    if "cmeans" in parts:
        parts["cmeans"].set_color("#333333")
    rng = np.random.RandomState(0)
    for xpos, v in [(0, nb.values), (1, b.values)]:
        ax.scatter(xpos + rng.uniform(-0.09, 0.09, len(v)), v, s=26, color=col,
                   edgecolors="black", linewidths=0.4, zorder=5)
    u, p = mannwhitneyu(b, nb, alternative="two-sided")
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"non-binder\n(n={len(nb)})", f"binder\n(n={len(b)})"],
                                              fontsize=11)
    ax.set_ylabel("score (lower = more favorable)", fontsize=11)
    ax.set_title(f"{lab}\nMann-Whitney p={p:.3f}", fontsize=12)
    ax.tick_params(labelsize=10)

    ax = axes[row, 1]
    sc = ax.scatter(s[c], -np.log10(s.kd_value), c=s.hamming_to_native, cmap="viridis_r",
                    s=110, edgecolors="black", linewidths=0.6, zorder=5)
    for _, r_ in s.iterrows():
        ax.annotate(r_.Peptide, (r_[c], -np.log10(r_.kd_value)), fontsize=7.5,
                    xytext=(6, 4), textcoords="offset points")
    lo = df[c].min() - 0.05
    ax.plot(df.loc[~df.is_binder, c], np.full((~df.is_binder).sum(), lo * 0 + ax.get_ylim()[0]),
            "x", color="grey", alpha=0.55, markersize=6, label=f"non-binder (no KD, n={len(nb)})")
    cb = fig.colorbar(sc, ax=ax); cb.set_label("Hamming to SLLMWITQC", fontsize=10)
    r_p, p_p = pearsonr(s[c], np.log10(s.kd_value))
    ax.set_xlabel("score (lower = more favorable)", fontsize=11)
    ax.set_ylabel("pKD = -log10(KD, M)\n[higher = stronger]", fontsize=11)
    ax.set_title(f"{lab} vs measured KD\nPearson r={r_p:+.2f} (p={p_p:.2f}), n={len(s)}", fontsize=12)
    ax.legend(fontsize=8, loc="lower left"); ax.tick_params(labelsize=10)

fig.suptitle("Inverse-folding score vs. measured binding, 51 NY-ESO-1/1G4c58c61 peptides (2P5E, full context)",
             fontsize=14, y=0.997)
fig.tight_layout()
out = FIG_DIR / "fig_kd_three_model_scores.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"wrote {out}")""")

md(r"""## 7. ROC: can any score classify binder vs. non-binder?

Sections 3 and 5 asked this with a rank test and with regression. The ROC view puts every model on
one axis and, crucially, plots **Hamming distance to native as a competing predictor** rather than as
a nuisance to control for. If a model's curve does not clearly beat the Hamming curve, that model is
not contributing anything a trivial sequence-similarity baseline does not already give you.

Direction convention: lower score and lower Hamming both mean "more native-like / more favorable", so
each predictor is negated before scoring.

With 8 positives and 43 negatives the AUC is very imprecise, so a bootstrap 95% interval is reported
alongside. Read the intervals, not the point estimates.""")

co(r"""from sklearn.metrics import roc_curve, roc_auc_score

y = df.is_binder.astype(int).values
PREDICTORS = [(-df[c].values, lab, col) for c, lab, col in MODELS]
PREDICTORS.append((-df.hamming_to_native.values.astype(float),
                   "Hamming to native (baseline)", "#444444"))

def boot_auc(y, x, n_boot=2000, seed=0):
    rng = np.random.RandomState(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    out = []
    for _ in range(n_boot):
        i = np.concatenate([rng.choice(pos, len(pos), replace=True),
                            rng.choice(neg, len(neg), replace=True)])
        if len(np.unique(y[i])) < 2:
            continue
        out.append(roc_auc_score(y[i], x[i]))
    return np.percentile(out, [2.5, 97.5])

fig, ax = plt.subplots(figsize=(7.2, 6.6))
rows = []
for x, lab, col in PREDICTORS:
    auc = roc_auc_score(y, x)
    lo, hi = boot_auc(y, x)
    fpr, tpr, _ = roc_curve(y, x)
    style = "--" if "Hamming" in lab else "-"
    ax.plot(fpr, tpr, style, color=col, lw=2.2, label=f"{lab}: AUC={auc:.3f} [{lo:.2f}-{hi:.2f}]")
    rows.append({"predictor": lab, "AUC": round(auc, 3),
                 "boot95_lo": round(lo, 3), "boot95_hi": round(hi, 3)})

ax.plot([0, 1], [0, 1], ":", color="grey", lw=1.2, label="chance (AUC=0.5)")
ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
ax.set_title(f"Binder vs. non-binder, 51 NY-ESO-1 peptides\n"
             f"{int(y.sum())} binders vs {int((1-y).sum())} non-binders", fontsize=12)
ax.legend(fontsize=9, loc="lower right"); ax.set_aspect("equal")
fig.tight_layout()
out = FIG_DIR / "fig_kd_roc_all_models_vs_hamming.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"wrote {out}")
pd.DataFrame(rows)""")

md(r"""## 7. What this shows

Read the numbers printed above rather than this cell, but the shape of the result:

- The three models **agree strongly with each other** (r = 0.76-0.86), so they are largely measuring
  one shared quantity.
- Only **LigandMPNN separates binders from non-binders** on the full set. The two ProteinMPNN-series
  scores do not.
- **No model tracks KD magnitude** among the 8 measured binders.
- Every score **correlates with distance to native**, and binders are themselves closer to native
  than non-binders, so the binder signal is confounded. Controlling for it, LigandMPNN's separation
  survives within the overlapping Hamming stratum but not as a partial correlation over all 51.

The honest summary is that LigandMPNN carries signal the ProteinMPNN series does not -- plausible,
since it is the only one of the three with explicit ligand/heteroatom context -- but n=8 binders
cannot establish it. The distance-to-native correlation is the only robust association in the set.""")

md(r"""## 7. One structure or seven? Between-structure spread, and whether resolution explains it

Everything above scored these peptides against a single crystal (2P5E). That folds two things
together: how a peptide scores, and which structure it was scored against. Seven NY-ESO-1 crystals are
available locally -- six carrying the wild-type SLLMWITQC and one (2BNQ) the 9V variant -- so the same
51 peptides can be scored on all of them, giving a mean and a spread per peptide.

A spread is only interpretable against a noise floor. ProteinMPNN's score is an expectation over
random autoregressive decoding orders, so re-scoring the *same* structure under different seeds
already moves it. `py/score_nyeso_kd_multistructure.py --noise-floor` measures that directly, and the
between-structure spread is compared against it rather than reported on its own.""")

co(r"""multi = pd.read_csv(ROOT / "outputs/analysis/nyeso_kd_multistructure_scores.csv")
noise = pd.read_csv(ROOT / "outputs/analysis/nyeso_kd_seed_noise_floor.csv")
floor = noise.seed_sd.mean()
print(f"{multi.structure.nunique()} structures x {multi.peptide.nunique()} peptides")
print(f"seed-only noise floor (same structure, 3 decoding-order seeds): SD {floor:.4f}\n")

for c, lab in [("score_vanilla", "ProteinMPNN"), ("score_nomhc", "ProteinMPNN (no MHC)")]:
    sd = multi.groupby("peptide")[c].std(ddof=1)
    print(f"  {lab:22s} between-structure SD {sd.mean():.3f} "
          f"(median {sd.median():.3f}, max {sd.max():.3f}) = {sd.mean() / floor:.0f}x the floor")

# do the structures at least agree on the ORDER of the peptides?
w = multi.pivot_table(index="peptide", columns="structure", values="score_vanilla")
rs = [pearsonr(w[a], w[b])[0] for a, b in itertools.combinations(sorted(w.columns), 2)]
print(f"\npairwise Pearson between structures: min {min(rs):.2f}, median {np.median(rs):.2f}, "
      f"max {max(rs):.2f} ({len(rs)} pairs)")""")

co(r"""# per structure: its mean score, its spread over peptides, and how far it sits from the
# 7-structure consensus for the same peptide
cons = multi.groupby("peptide")["score_vanilla"].mean()
multi["dev"] = (multi["score_vanilla"] - multi["peptide"].map(cons)).abs()
per = (multi.groupby(["structure", "resolution_A", "native"])
            .agg(mean_score=("score_vanilla", "mean"),
                 sd_over_peptides=("score_vanilla", "std"),
                 mean_abs_dev=("dev", "mean")).reset_index()
            .sort_values("resolution_A"))
print(per.round(3).to_string(index=False))

print()
for col in ["mean_score", "sd_over_peptides", "mean_abs_dev"]:
    r, pv = pearsonr(per.resolution_A, per[col])
    rho, pr = spearmanr(per.resolution_A, per[col])
    print(f"resolution vs {col:17s} r={r:+.2f} (p={pv:.3f})  rho={rho:+.2f} (p={pr:.3f})  n={len(per)}")""")

co(r"""fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

# (a) per-peptide spread across structures, against the seed-only floor
sd = multi.groupby("peptide")["score_vanilla"].std(ddof=1)
axes[0].hist(sd, bins=18, color="#4C72B0", edgecolor="black")
axes[0].axvline(floor, color="crimson", ls="--", lw=2,
                label=f"seed-only floor ({floor:.3f})")
axes[0].axvline(sd.mean(), color="black", ls=":", lw=2, label=f"mean ({sd.mean():.3f})")
axes[0].set_xlabel("SD of a peptide's score across the 7 structures", fontsize=12)
axes[0].set_ylabel("peptides", fontsize=12)
axes[0].set_title("Between-structure spread is real,\nnot decoding-order noise", fontsize=13)
axes[0].legend(fontsize=10)

# (b) every peptide on every structure, structures ordered by resolution
order = per.structure.tolist()
for i, s in enumerate(order):
    v = multi.loc[multi.structure == s, "score_vanilla"].values
    axes[1].scatter(np.full(len(v), i) + np.random.RandomState(0).normal(0, 0.07, len(v)),
                    v, s=14, alpha=0.5, color="#4C72B0")
    axes[1].scatter([i], [v.mean()], s=140, color="crimson", zorder=5,
                    marker="_", linewidths=3)
axes[1].set_xticks(range(len(order)))
axes[1].set_xticklabels([f"{s}\n{r:.2f} Å" for s, r in zip(per.structure, per.resolution_A)],
                        fontsize=10)
axes[1].set_ylabel("ProteinMPNN score", fontsize=12)
axes[1].set_title("All 51 peptides per structure\n(red bar = mean), best resolution first",
                  fontsize=13)

# (c) does resolution explain how far a structure sits from consensus?
axes[2].scatter(per.resolution_A, per.mean_abs_dev, s=110, color="#4C72B0",
                edgecolor="black", zorder=3)
for _, r_ in per.iterrows():
    axes[2].annotate(r_.structure, (r_.resolution_A, r_.mean_abs_dev), fontsize=10,
                     xytext=(6, 4), textcoords="offset points")
m, b = np.polyfit(per.resolution_A, per.mean_abs_dev, 1)
xs = np.linspace(per.resolution_A.min(), per.resolution_A.max(), 50)
axes[2].plot(xs, m * xs + b, "k--", lw=1.2)
r_dev, p_dev = pearsonr(per.resolution_A, per.mean_abs_dev)
axes[2].set_xlabel("resolution (Å)", fontsize=12)
axes[2].set_ylabel("mean |score − 7-structure consensus|", fontsize=12)
axes[2].set_title(f"Deviation from consensus vs. resolution\nr={r_dev:+.2f} (p={p_dev:.3f}), "
                  f"n={len(per)}", fontsize=13)
for ax in axes:
    ax.tick_params(labelsize=11)
fig.tight_layout()
out = FIG_DIR / "fig_kd_multistructure_spread_vs_resolution.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"wrote {out}")""")

md(r"""### What the seven structures say

**The structure you score against matters, and it matters far more than decoding-order noise.** A
peptide's score moves by an SD of $0.147$ across the seven crystals, against a seed-only floor of
$0.0156$ -- roughly a tenfold ratio. Scoring on one crystal and treating the number as a property of
the peptide is therefore not safe: a large part of it is a property of that crystal.

**The structures still agree on the ordering.** Pairwise correlations between structures run $0.73$ to
$0.96$ (median $0.82$), so they largely rank the 51 peptides the same way while sitting at different
absolute levels. What varies between crystals is the offset, not the ranking.

**Resolution does not explain it.** Resolution predicts neither a structure's mean score
($r=-0.30$, $p=0.51$) nor its spread over peptides ($r=+0.12$, $p=0.80$). The one suggestion of a
relationship is that poorer-resolution crystals sit further from the seven-structure consensus
($r=+0.66$, $p=0.11$), which is the direction one would expect if coordinate error added
idiosyncrasy -- but at $n=7$ it does not reach significance and should not be reported as an effect.

The practical consequence is about method rather than biology. If a score is going to be compared
across peptides, it should be averaged over whatever independent structures of the same complex exist,
because the between-crystal offset is an order of magnitude larger than the sampling noise that the
single-structure analysis above implicitly treats as the only source of error.""")

nb["cells"] = C
out_nb = Path("/home/ubuntu/if-mhc/notebooks/27_kd_peptide_inverse_folding_scores.ipynb")
out_nb.parent.mkdir(exist_ok=True, parents=True)
nbf.write(nb, str(out_nb))
print(f"wrote {out_nb}")
