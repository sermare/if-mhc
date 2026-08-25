#!/usr/bin/env python3
"""Emit notebooks/26_iedb_strength_recovery.ipynb.

Three analyses, each run TWICE -- once for `mhconly` (MHC present, NO TCR input) designs and once
for `full` (MHC+TCR) designs -- as clearly separate, parallel plots so the TCR-present vs TCR-absent
comparison is direct:

1. Per-crystal, per-position recovery matrix against the CRYSTAL'S OWN NATIVE PEPTIDE only -- no
   IEDB involved. One matrix per crystal, rows=position, columns=model.
2. A violin per position, pooling (crystal, model) recovery rates at that position across all
   crystals -- still native-peptide recovery, just summarized across the whole panel.
3. The IEDB-tier analog of fig_if11 (Hamming-histogram-by-round-and-source): same grid layout, same
   nearest-design-match + mean-distance-line mechanic, but rows = IEDB binding-strength tier
   (Low/Intermediate/High) instead of NGS round. One grid per crystal.

Only structures with fully complete designs (all 4 models, BOTH conditions) at build time are
included.

Build + execute:
  /home/ubuntu/miniforge3/bin/python3 py/build_iedb_strength_recovery_notebook.py
  cd /home/ubuntu/if-mhc && /home/ubuntu/miniforge3/bin/jupyter nbconvert \
      --to notebook --execute --inplace notebooks/26_iedb_strength_recovery.ipynb
"""
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook(); C = []
def md(t): C.append(nbf.v4.new_markdown_cell(t))
def co(t): C.append(nbf.v4.new_code_cell(t))

md(r"""# Panel-wide recovery: per-crystal native recovery, and IEDB binding-strength tiers as "rounds"

Every analysis below is run twice, as clearly separate plots:
- **mhconly** -- MHC present, NO TCR input (ProteinMPNN/noMHC ProteinMPNN/ESM-IF1/LigandMPNN all run without a TCR)
- **full** -- MHC + TCR present (same 4 models, TCR included)

so the TCR-present vs TCR-absent comparison is direct rather than mixed into one plot.

1. **Per-crystal, per-position recovery vs. the crystal's own native/index peptide** -- plain
   recovery, no IEDB involved.
2. **A violin per peptide position**, pooling recovery rates across crystals and models.

The IEDB binding-strength-tier Hamming comparison (previously a separate histogram grid per
crystal x condition here) now lives in its own consolidated one-figure summary, built by
`py/plot_iedb_hamming_summary.py` -> `figures/fig_iedb3_tier_hamming_hist/fig_iedb3_hamming_summary.png`
(mean nearest-Hamming-distance to real IEDB binders, mhconly vs full, per crystal + pooled) --
not regenerated here to avoid re-cluttering that directory with per-crystal grids on every rerun.

Other companion figures (also standalone scripts, not in this notebook): per-crystal x model
recovery matrices (`py/plot_iedb_recovery_crystal_by_model.py` +
`..._per_residue.py`), and model-agreement correlation (`py/plot_iedb_model_correlation.py`).

**Scope:** only structures with COMPLETE designs (all 4 models, both `mhconly` and `full`) at build
time. The panel-wide campaign (`jobs/run_panel_queue.sh`) is still running the rest.""")

co(r"""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path("/home/ubuntu/if-mhc")
AA = list("ACDEFGHIKLMNPQRSTVWY")
MODELS = ["vanilla", "noMHC", "ESM-IF1", "LigandMPNN"]
MODEL_LABEL = {"vanilla": "ProteinMPNN", "noMHC": "noMHC ProteinMPNN (No MHC)", "ESM-IF1": "ESM-IF1",
               "LigandMPNN": "LigandMPNN"}
CONDITIONS = ["mhconly", "full"]
COND_LABEL = {"mhconly": "mhconly (NO TCR input)", "full": "full (MHC + TCR)"}
TIERS = ["Positive-Low", "Positive-Intermediate", "Positive-High"]
TIER_SHORT = {"Positive-Low": "Low", "Positive-Intermediate": "Intermediate", "Positive-High": "High"}

dataset = pd.read_csv(ROOT / "inputs/pmhc_tcr_dataset/dataset.csv")

def condition_complete(pdb, cond):
    base = ROOT / f"outputs/panel/{pdb}/{cond}"
    return all((base / tool / "COMPLETE").exists() for tool in ["mpnn", "esmif", "ligandmpnn"])

candidates = [p for p in dataset["pdb"] if p not in ("3HG1", "2P5E")]
ready = [p for p in candidates if all(condition_complete(p, c) for c in CONDITIONS)]
print(f"structures with complete designs (both conditions) right now: {ready}")""")

co(r"""def peptide_from_ligandmpnn_line(line):
    return line.strip().split(":")[2]

def load_designs(pdb, cond):
    rows = []
    for weights, fname in [("vanilla", f"vanilla_{pdb}.fa"), ("noMHC", f"nomhc_{pdb}.fa")]:
        path = ROOT / f"outputs/panel/{pdb}/{cond}/mpnn/seqs/{fname}"
        with open(path) as f:
            lines = f.read().splitlines()
        for i in range(0, len(lines) - 1, 2):
            if lines[i].startswith(">"):
                rows.append({"peptide": lines[i + 1].strip(), "model": weights})
    path = ROOT / f"outputs/panel/{pdb}/{cond}/esmif/seqs/{pdb}.fa"
    with open(path) as f:
        lines = f.read().splitlines()
    for i in range(0, len(lines) - 1, 2):
        if lines[i].startswith(">"):
            rows.append({"peptide": lines[i + 1].strip(), "model": "ESM-IF1"})
    path = ROOT / f"outputs/panel/{pdb}/{cond}/ligandmpnn/seqs/{pdb}.fa"
    with open(path) as f:
        lines = f.read().splitlines()
    for i in range(0, len(lines) - 1, 2):
        if lines[i].startswith(">"):
            rows.append({"peptide": peptide_from_ligandmpnn_line(lines[i + 1]), "model": "LigandMPNN"})
    return pd.DataFrame(rows)

designs = {(pdb, cond): load_designs(pdb, cond) for pdb in ready for cond in CONDITIONS}
natives = {pdb: dataset.loc[dataset.pdb == pdb, "peptide"].iloc[0] for pdb in ready}
for pdb in ready:
    for cond in CONDITIONS:
        print(f"{pdb}/{cond} (native={natives[pdb]}): {len(designs[(pdb, cond)]):,} designs")""")

md(r"""## 1. Per-crystal, per-position recovery vs. the native peptide (no IEDB)

For each crystal, each model, each condition: fraction of designs matching the native peptide's
amino acid at each position. One heatmap per crystal per condition.""")

co(r"""native_recovery = {}
for cond in CONDITIONS:
    for pdb in ready:
        native = natives[pdb]
        length = len(native)
        df = designs[(pdb, cond)]
        mat = np.zeros((length, len(MODELS)))
        for j, model in enumerate(MODELS):
            peps = df.loc[df.model == model, "peptide"]
            peps = [p for p in peps if len(p) == length]
            hits = np.zeros(length)
            for p in peps:
                for pos in range(length):
                    if p[pos] == native[pos]:
                        hits[pos] += 1
            mat[:, j] = hits / len(peps) if peps else np.nan
        native_recovery[(pdb, cond)] = mat

Path(ROOT / "figures/fig_iedb1_recovery_matrix").mkdir(exist_ok=True, parents=True)
for cond in CONDITIONS:
    fig, axes = plt.subplots(1, len(ready), figsize=(5.5 * len(ready), 5), squeeze=False)
    for i, pdb in enumerate(ready):
        native = natives[pdb]
        mat = native_recovery[(pdb, cond)]
        ax = axes[0, i]
        im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels([MODEL_LABEL[m] for m in MODELS], rotation=30, ha="right")
        ax.set_yticks(range(len(native))); ax.set_yticklabels(range(1, len(native) + 1))
        ax.set_xlabel("model"); ax.set_ylabel("peptide position")
        ax.set_title(f"{pdb} (native={native})", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="recovery of native residue")
    fig.suptitle(f"Per-position recovery vs. native/crystal peptide -- {COND_LABEL[cond]}", y=1.03)
    fig.tight_layout()
    out = ROOT / f"figures/fig_iedb1_recovery_matrix/fig_iedb1_recovery_matrix_{cond}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"wrote {out}")""")

md(r"""## 2. Violin per position, across all crystals

Still native-peptide recovery (no IEDB) -- each violin at position *p* pools the (crystal, model)
recovery rates at that position across the whole panel, with the individual (crystal, model) points
overlaid on top (jittered, colored by crystal) so you can see which specific crystals sit where in
the distribution, not just the aggregate shape. Shown once per condition.""")

co(r"""import matplotlib.cm as cm
max_len = max(len(natives[pdb]) for pdb in ready)
Path(ROOT / "figures/fig_iedb2_violin_summary").mkdir(exist_ok=True, parents=True)
pdb_colors = {pdb: cm.tab20(i / max(len(ready), 1)) for i, pdb in enumerate(ready)}
rng_jitter = np.random.RandomState(0)

for cond in CONDITIONS:
    by_position = {p: [] for p in range(max_len)}     # values only, for the violin shape
    by_position_pdb = {p: [] for p in range(max_len)}  # (value, pdb) pairs, for the point overlay
    for pdb in ready:
        length = len(natives[pdb])
        mat = native_recovery[(pdb, cond)]
        for pos in range(length):
            for j, model in enumerate(MODELS):
                by_position[pos].append(mat[pos, j])
                by_position_pdb[pos].append((mat[pos, j], pdb))

    fig, ax = plt.subplots(figsize=(2 + 1.3 * max_len, 6.5))
    positions_with_data = [p for p in range(max_len) if by_position[p]]
    data = [by_position[p] for p in positions_with_data]
    parts = ax.violinplot(data, positions=[p + 1 for p in positions_with_data], showmeans=True,
                          widths=0.8)
    for pc in parts["bodies"]:
        pc.set_facecolor("#4C72B0"); pc.set_alpha(0.35)
    for key in ["cmeans", "cmaxes", "cmins", "cbars"]:
        if key in parts:
            parts[key].set_color("#333333")

    for p in positions_with_data:
        vals_pdbs = by_position_pdb[p]
        xs = (p + 1) + rng_jitter.uniform(-0.18, 0.18, size=len(vals_pdbs))
        for (val, pdb), x in zip(vals_pdbs, xs):
            ax.scatter(x, val, s=22, alpha=0.85, color=pdb_colors[pdb],
                       edgecolors="black", linewidths=0.3, zorder=5)

    ax.set_xlabel("peptide position")
    ax.set_ylabel("recovery of native residue\n(pooled across crystals x models)")
    ax.set_title(f"Per-position native recovery -- {COND_LABEL[cond]}\npooled across {len(ready)} crystals x "
                 f"{len(MODELS)} models (n={len(ready) * len(MODELS)} points/position)")
    ax.set_xticks([p + 1 for p in positions_with_data])
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=pdb_colors[pdb],
                          markeredgecolor="black", markersize=7, label=f"{natives[pdb]} ({pdb})")
               for pdb in ready]
    ax.legend(handles=handles, title="peptide (crystal)", fontsize=7, ncol=2, loc="upper right",
              bbox_to_anchor=(1.38, 1.02))
    fig.tight_layout()
    out = ROOT / f"figures/fig_iedb2_violin_summary/fig_iedb2_violin_summary_{cond}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"wrote {out}")""")

nb["cells"] = C
Path("/home/ubuntu/if-mhc/notebooks").mkdir(exist_ok=True)
nbf.write(nb, "/home/ubuntu/if-mhc/notebooks/26_iedb_strength_recovery.ipynb")
print("wrote notebooks/26_iedb_strength_recovery.ipynb")
