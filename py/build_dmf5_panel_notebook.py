#!/usr/bin/env python3
"""Emit notebooks/29_dmf5_kd_panel_two_structures.ipynb.

The DMF5 KD panel scored on both 6AM5 and 6AMU by three inverse-folding models. This is the
cross-reactive two-state system: one TCR, two chemically distinct peptides, two bound conformations,
and measured KD for both index peptides plus 11 variants.

Build + execute:
  /home/ubuntu/miniforge3/bin/python3 py/build_dmf5_panel_notebook.py
  cd /home/ubuntu/if-mhc && /home/ubuntu/miniforge3/bin/jupyter nbconvert \
      --to notebook --execute --inplace notebooks/29_dmf5_kd_panel_two_structures.ipynb
"""
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook(); C = []
def md(t): C.append(nbf.v4.new_markdown_cell(t))
def co(t): C.append(nbf.v4.new_code_cell(t))

md(r"""# DMF5 KD panel on two structures: does each backbone prefer its own peptide?

The DMF5 TCR engages two chemically distinct peptides through two different bound backbone
conformations. Both are solved, and both index peptides have measured KD, which makes this a rare
two-state test with ground truth on each side.

| structure | peptide chain C | shorthand | measured KD |
|---|---|---|---|
| 6AM5 | SMLGIGIVPV | GIG | 43 uM |
| 6AMU | MMWDRGLGMM | DRG | 32 uM |

**Panel.** 15 peptides with measured KD, dG and Tm. The 13 that are 10-mers are used here; the two
9-mers (NLSNLGILV, MMWDRGLGM) cannot be threaded onto a 10-residue backbone without a gap or
deletion, which would change the geometry being scored, so they are excluded rather than forced.

**Design.** Every 10-mer is scored on *both* backbones by all three models, giving a
13 x 2 x 3 grid. Because each structure's own peptide is in the panel, the sharpest question is
whether a backbone scores its own crystallized peptide better than the other one's -- and whether the
score difference between backbones tracks which peptide the structure actually holds.

**Score.** Mean per-residue negative log-likelihood over the 10 peptide positions, lower = more
favorable. ProteinMPNN and its no-MHC variant use the batched scorer; LigandMPNN scores one threaded
PDB per peptide (backbone-only chain C, side chains dropped since they belong to the native
sequence). 50 decoding orders throughout -- the panel is small, so precision is cheap.""")

co(r"""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, spearmanr, wilcoxon

ROOT = Path("/home/ubuntu/if-mhc")
FIG_DIR = ROOT / "figures/fig_dmf5_kd_two_structures"
FIG_DIR.mkdir(exist_ok=True, parents=True)

MODELS = [("score_vanilla", "ProteinMPNN", "#0072B2"),
          ("score_nomhc", "noMHC ProteinMPNN", "#E69F00"),
          ("score_ligandmpnn", "LigandMPNN", "#CC79A7")]
NATIVE = {"6AM5": "SMLGIGIVPV", "6AMU": "MMWDRGLGMM"}

df = pd.read_csv(ROOT / "outputs/analysis/dmf5_kd_panel_3model_2structure_scores.csv")
print(f"{len(df)} rows | structures {sorted(df.structure.unique())} | "
      f"{df.peptide.nunique()} peptides | missing scores "
      f"{int(df[[c for c,_,_ in MODELS]].isna().sum().sum())}")
df.head(6)""")

md(r"""## 1. Does each structure score its own peptide best?

The most direct two-state question. For each backbone, rank all 13 peptides by score and see where
the crystallized one lands. Rank 1 = the model's top pick is the peptide actually in the structure.""")

co(r"""for struct in ["6AM5", "6AMU"]:
    sub = df[df.structure == struct]
    print(f"{struct} (holds {NATIVE[struct]}):")
    for c, lab, _ in MODELS:
        s = sub.sort_values(c).reset_index(drop=True)
        rank = int(s.index[s.peptide == NATIVE[struct]][0]) + 1
        own = float(sub.loc[sub.peptide == NATIVE[struct], c].iloc[0])
        other = float(sub.loc[sub.peptide == NATIVE["6AMU" if struct == "6AM5" else "6AM5"], c].iloc[0])
        flag = "*" if rank == 1 else " "
        print(f"  {flag} {lab:20s} own peptide rank {rank}/13  "
              f"(score {own:.3f}; the other structure's peptide scores {other:.3f})")
    print()""")

md(r"""## 2. Score difference between the two backbones

For each peptide, `delta = score(on 6AM5) - score(on 6AMU)`. Negative means the peptide fits the GIG
backbone better; positive means it fits the DRG backbone better. If the models read the two
conformations as genuinely different environments, the two index peptides should sit at opposite
ends of this axis.""")

co(r"""wide = df.pivot_table(index="peptide", columns="structure",
                      values=[c for c, _, _ in MODELS])
rows = []
for c, lab, _ in MODELS:
    d = wide[c]["6AM5"] - wide[c]["6AMU"]
    rows.append(pd.Series(d, name=lab))
delta = pd.concat(rows, axis=1)
delta["GIG-like"] = ["GIG" in p or p.startswith(("SML", "SMA", "ELA", "NMG")) for p in delta.index]
delta.loc[["SMLGIGIVPV", "MMWDRGLGMM"]].round(3)""")

co(r"""print("delta = score(6AM5) - score(6AMU); negative = fits GIG backbone better\n")
print(delta.drop(columns="GIG-like").round(3).to_string())
print("\nthe two index peptides:")
for c, lab, _ in MODELS:
    g = delta.loc["SMLGIGIVPV", lab]; r = delta.loc["MMWDRGLGMM", lab]
    ok = "as expected" if g < r else "NOT as expected"
    print(f"  {lab:20s} SMLGIGIVPV(GIG) {g:+.3f} vs MMWDRGLGMM(DRG) {r:+.3f}  -> {ok}")""")

md(r"""## 3. Does score track measured KD?

Lower KD = tighter binding. Scores are per-structure, so each is correlated against KD separately.
One peptide (MMWDRGLGMV) is "no binding detected" and has no KD, and three are censored at >500 uM,
recorded as 500. Both facts limit what a correlation can mean here, so Spearman is the honest
statistic and n is small either way.""")

co(r"""kd = df[df.KD_uM.notna()].copy()
kd["logKD"] = np.log10(kd.KD_uM)
print(f"{kd.peptide.nunique()} peptides with a KD value "
      f"({int((kd.KD_uM == 500).sum() / 2)} censored at >500 uM)\n")
for struct in ["6AM5", "6AMU"]:
    sub = kd[kd.structure == struct]
    print(f"{struct}:")
    for c, lab, _ in MODELS:
        r, p = pearsonr(sub[c], sub.logKD); rho, ps = spearmanr(sub[c], sub.logKD)
        print(f"    {lab:20s} Pearson r={r:+.3f} (p={p:.3f})  Spearman rho={rho:+.3f} "
              f"(p={ps:.3f})  n={len(sub)}")""")

md(r"""## 4. Are the two structures actually giving different answers?

If scores on 6AM5 and 6AMU were near-identical for every peptide, the two-state framing would add
nothing. Paired Wilcoxon across the 13 peptides, plus the correlation between the two backbones.""")

co(r"""for c, lab, _ in MODELS:
    a = wide[c]["6AM5"]; b = wide[c]["6AMU"]
    stat, p = wilcoxon(a, b)
    r, _ = pearsonr(a, b)
    print(f"{lab:20s} mean 6AM5={a.mean():.3f}  6AMU={b.mean():.3f}  "
          f"paired Wilcoxon p={p:.4f}  |  between-structure r={r:+.3f}")""")

md(r"""## 5. Figure""")

co(r"""fig, axes = plt.subplots(2, len(MODELS), figsize=(6.2 * len(MODELS), 10.5))

for j, (c, lab, col) in enumerate(MODELS):
    # (a) per-peptide score on each backbone, index peptides highlighted
    ax = axes[0, j]
    order = df[df.structure == "6AM5"].sort_values(c).peptide.tolist()
    x = np.arange(len(order))
    for struct, mk, cc in [("6AM5", "o", "#0072B2"), ("6AMU", "s", "#D55E00")]:
        vals = df.set_index(["structure", "peptide"]).loc[struct, c].reindex(order)
        ax.plot(x, vals.values, mk + "-", color=cc, alpha=0.85, markersize=7,
                label=f"on {struct} ({NATIVE[struct]})")
    for pep, cc in [("SMLGIGIVPV", "#0072B2"), ("MMWDRGLGMM", "#D55E00")]:
        if pep in order:
            ax.axvline(order.index(pep), color=cc, ls=":", lw=1.6, alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=90, fontsize=7.5)
    ax.set_ylabel("score (lower = more favorable)")
    ax.set_title(f"{lab}: same peptide on both backbones", fontsize=11)
    ax.legend(fontsize=8)

    # (b) score vs measured KD, both structures
    ax = axes[1, j]
    for struct, mk, cc in [("6AM5", "o", "#0072B2"), ("6AMU", "s", "#D55E00")]:
        sub = kd[kd.structure == struct]
        ax.scatter(sub[c], sub.logKD, marker=mk, s=80, color=cc, edgecolors="black",
                   linewidths=0.5, alpha=0.85, label=f"on {struct}")
        rho, ps = spearmanr(sub[c], sub.logKD)
        ax.plot([], [], " ", label=f"   rho={rho:+.2f} (p={ps:.2f})")
    for _, r_ in kd[kd.structure == "6AM5"].iterrows():
        if r_.peptide in NATIVE.values():
            ax.annotate(r_.peptide, (r_[c], r_.logKD), fontsize=7.5,
                        xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel("score (lower = more favorable)")
    ax.set_ylabel("log10(KD, uM)  [lower = tighter]")
    ax.set_title(f"{lab}: score vs measured KD", fontsize=11)
    ax.legend(fontsize=7.5)

fig.suptitle("DMF5 KD panel: 13 10-mers scored on both 6AM5 (GIG) and 6AMU (DRG) backbones",
             fontsize=13, y=1.0)
fig.tight_layout()
out = FIG_DIR / "fig_dmf5_panel_two_structures.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"wrote {out}")""")

co(r"""# heatmap: peptides x (model, structure), z-scored within each column so models are comparable
fig, ax = plt.subplots(figsize=(9, 7))
cols, labels = [], []
for c, lab, _ in MODELS:
    for struct in ["6AM5", "6AMU"]:
        v = df[df.structure == struct].set_index("peptide")[c]
        cols.append((v - v.mean()) / v.std())
        labels.append(f"{lab}\non {struct}")
M = pd.concat(cols, axis=1)
M.columns = labels
order = M.mean(axis=1).sort_values().index
M = M.loc[order]
im = ax.imshow(M.values, cmap="magma_r", aspect="auto")
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8.5)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([f"{p}  (KD {df.loc[df.peptide==p,'KD_uM'].iloc[0]})" for p in order], fontsize=8)
for i, p in enumerate(order):
    if p in NATIVE.values():
        ax.get_yticklabels()[i].set_fontweight("bold")
fig.colorbar(im, ax=ax, label="z-scored score within column\n(lower = more favorable)")
ax.set_title("DMF5 panel, all three models on both backbones\n(index peptides in bold)", fontsize=12)
fig.tight_layout()
out = FIG_DIR / "fig_dmf5_panel_heatmap.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"wrote {out}")""")

md(r"""## 6. Reading the result

The questions this can and cannot answer, in order of how much the data supports them:

1. **Does each backbone prefer its own peptide?** (section 1) This is the cleanest test, because it
   needs no KD at all -- just the rank of the crystallized peptide among 13.
2. **Do the two backbones disagree?** (section 4) If not, the whole two-state premise collapses.
3. **Does score track KD?** (section 3) Weakest of the three: n=12 with three values censored at
   >500 uM and one non-binder excluded, so treat any correlation as suggestive at best.

Caveat that applies throughout: the DRG-side peptides in this panel are mostly single-point mutants
of MMWDRGLGMM, so score differences among them are small by construction, and the GIG side is much
more sequence-diverse. That asymmetry, not model quality, may drive part of any structure-level
difference.""")

nb["cells"] = C
out_nb = Path("/home/ubuntu/if-mhc/notebooks/29_dmf5_kd_panel_two_structures.ipynb")
out_nb.parent.mkdir(exist_ok=True, parents=True)
nbf.write(nb, str(out_nb))
print(f"wrote {out_nb}")
