#!/usr/bin/env python3
"""Emit notebooks/panel/05_skempi_validation.ipynb -- cross-reference the SKEMPI 2.0 TCR/pMHC alanine
scan against our own per-position design entropy/recovery for 3HG1 (MEL5 x ELAGIGILTV), the one panel
structure with real experimental peptide-position mutation data.

1. SKEMPI TCR/pMHC subset: overview, overlap with our own structures
2. 3HG1 alanine scan: which peptide positions were tested, and what happened to binding
3. Our own 3HG1 designs: per-position recovery and entropy, 4 models, full context, T=0.1
4. Cross-reference: does the alanine scan's verdict line up with our recovery/entropy?
5. Statistical comparison: SKEMPI-validated TCR-critical positions vs. everything else

Build + execute:
  /home/ubuntu/miniforge3/bin/python3 py/build_panel_skempi_notebook.py
  cd /home/ubuntu/if-mhc && /home/ubuntu/miniforge3/bin/jupyter nbconvert \
      --to notebook --execute --inplace notebooks/panel/05_skempi_validation.ipynb
"""
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook(); C = []
def md(t): C.append(nbf.v4.new_markdown_cell(t))
def co(t): C.append(nbf.v4.new_code_cell(t))

md(r"""# SKEMPI validation: does a real alanine scan agree with our recovery/entropy?

Everything in notebooks 01-04 is static geometry (contacts, distances, B-factor) or model
self-consistency (entropy, recovery of the native residue) -- never a measured binding effect. SKEMPI
2.0 is a curated database of *experimentally measured* binding-affinity changes upon mutation, and it
has a dedicated `TCR/pMHC` category (751 mutation rows, 38 complexes). One of those complexes is
**3HG1** -- the MEL5 TCR bound to ELAGIGILTV/HLA-A2 -- which also has full 4-model (ProteinMPNN,
ProteinMPNN (no MHC), ESM-IF1, LigandMPNN), T=0.1, full pMHC+TCR-context design data from this project's
earlier (pre-panel) design campaign. That overlap is the one place we can ask, with a real experiment
in hand: does a peptide position where alanine substitution measurably abolishes TCR-pMHC binding look
different, in our own recovery/entropy metrics, from a position that wasn't (or couldn't be) tested?

1. SKEMPI TCR/pMHC subset: overview, overlap with our own structures
2. 3HG1 alanine scan: which peptide positions were tested, and what happened to binding
3. Our own 3HG1 designs: per-position recovery and entropy, 4 models, full context, T=0.1
4. Cross-reference: does the alanine scan's verdict line up with our recovery/entropy?
5. Statistical comparison: SKEMPI-validated TCR-critical positions vs. everything else""")

co(r"""import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import logomaker
from pathlib import Path
from scipy.stats import mannwhitneyu, spearmanr

ROOT = Path("/home/ubuntu/if-mhc")
FIG_DIR = ROOT / "figures/fig_panel5_skempi_validation"
FIG_DIR.mkdir(exist_ok=True, parents=True)
AA = list("ACDEFGHIKLMNPQRSTVWY")
MODELS = ["vanilla", "noMHC", "ESM-IF1", "LigandMPNN"]
# Every (structure, arm, model) cell is truncated to the same number of designs. The raw
# counts differ slightly by generator (9,984 / 10,000 / 10,016), and unique-design counts
# scale with sample size, so comparisons across cells need a common N. 9,984 is the largest
# N every cell can supply; designs are i.i.d. draws, so a prefix is a valid subsample.
N_DESIGNS = 9984
MODEL_COLOR = {"vanilla": "#0072B2", "noMHC": "#E69F00", "ESM-IF1": "#009E73", "LigandMPNN": "#CC79A7"}
MODEL_LABEL = {"vanilla": "ProteinMPNN", "noMHC": "ProteinMPNN (no MHC)", "ESM-IF1": "ESM-IF1",
               "LigandMPNN": "LigandMPNN"}
PANEL_STRUCTS = ["2P5W", "1QSF", "1QRN", "2BNR", "2GJ6", "2F53", "2F54", "3QDG", "3QEQ", "3QFJ",
                 "3GSN", "1OGA", "3UTS", "5C0A", "5C0B", "5HHO", "5EU6", "2VLR", "4MJI", "5NME",
                 "1BD2", "1LP9", "1MI5", "1QSE", "2AK4", "2BNQ", "2E7L", "2J8U", "2JCC", "2OI9",
                 "2PYE", "2UWE", "3C60", "3D3V", "3H9S", "3PWP", "3QDJ", "3QIB", "4FTV", "4JFD",
                 "4JFE", "4JFF", "4L3E", "4MNQ", "4OZG", "4P23", "4P5T", "5E9D", "6AM5", "6AMU"]
FOCAL_STRUCTS = ["2P5E", "3HG1"]  # this project's two primary TCR-pMHC design targets

skempi = pd.read_csv(ROOT / "inputs/skempi/skempi_tcr_pmhc.csv")
skempi["pdb_code"] = skempi["#Pdb"].str.split("_").str[0]
print(f"{len(skempi)} SKEMPI TCR/pMHC mutation rows, {skempi['pdb_code'].nunique()} distinct complexes")""")

md(r"""## 1. SKEMPI TCR/pMHC subset: overview, overlap with our own structures""")

co(r"""all_skempi_pdbs = sorted(skempi["pdb_code"].unique())
panel_overlap = sorted(set(all_skempi_pdbs) & set(PANEL_STRUCTS))
focal_overlap = sorted(set(all_skempi_pdbs) & set(FOCAL_STRUCTS))
print(f"all {len(all_skempi_pdbs)} SKEMPI TCR/pMHC complexes: {all_skempi_pdbs}")
print(f"\noverlap with the 20-structure panel (notebooks 01-04): {panel_overlap}")
print(f"overlap with this project's two focal design targets: {focal_overlap}")

rows_per_pdb = skempi.groupby("pdb_code").size().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(12, 5))
colors = ["crimson" if p in FOCAL_STRUCTS else ("#4C72B0" if p in PANEL_STRUCTS else "#AAAAAA")
         for p in rows_per_pdb.index]
ax.bar(range(len(rows_per_pdb)), rows_per_pdb.values, color=colors)
ax.set_xticks(range(len(rows_per_pdb)))
ax.set_xticklabels(rows_per_pdb.index, rotation=90, fontsize=7)
ax.set_ylabel("# SKEMPI mutation rows")
ax.set_title("SKEMPI TCR/pMHC mutation rows per complex\n"
             "crimson = this project's focal structures (2P5E, 3HG1); blue = also in the 20-structure panel")
fig.tight_layout()
out = FIG_DIR / "fig_panel5_skempi_rows_per_complex.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"wrote {out}")""")

md(r"""## 1b. The actual n=20 panel overlap: peptide mutations in 1QRN, 1QSF, 2BNR, 3QFJ

**3HG1 -- the subject of the rest of this notebook -- is not one of the 20 panel structures.** It's a
separate, earlier design campaign on the same TCR-pMHC family. That's a real scope gap: the richest
SKEMPI signal available happens to sit just outside the panel this whole project is otherwise about.

The n=20 panel does have real overlap with SKEMPI (7 complexes, Section 1), but only 4 of those --
1QRN, 1QSF, 2BNR, 3QFJ -- have a peptide-chain (chain C) mutation at all, and each has exactly **one**.
Nowhere near 3HG1's 5-position alanine scan, but it's the part of SKEMPI that actually maps onto data
this project has already generated in notebooks 01-04/06 (panel, full pMHC+TCR context, T=0.1).""")

co(r"""panel_pep_muts = [
    {"pdb": "1QRN", "peptide": "LLFGYAVYV", "position": 6, "mutation": "A6P",
     "affinity_mut_M": "9.1E-07", "affinity_wt_M": "n.b."},
    {"pdb": "1QSF", "peptide": "LLFGYPVAV", "position": 8, "mutation": "A8Y",
     "affinity_mut_M": "9.1E-07", "affinity_wt_M": "n.b."},
    {"pdb": "2BNR", "peptide": "SLLMWITQC", "position": 9, "mutation": "C9V",
     "affinity_mut_M": "5.7E-06", "affinity_wt_M": "1.33E-05"},
    {"pdb": "3QFJ", "peptide": "LLFGFPVYV", "position": 5, "mutation": "F5Y",
     "affinity_mut_M": "9.0E-07", "affinity_wt_M": "1.2E-06"},
]

def peptide_from_ligandmpnn_line(line):
    return line.strip().split(":")[2]

def load_panel_designs(pdb, length):
    rows = []
    for weights, fname in [("vanilla", f"vanilla_{pdb}.fa"), ("noMHC", f"nomhc_{pdb}.fa")]:
        path = ROOT / f"outputs/panel/{pdb}/full/mpnn/seqs/{fname}"
        lines = path.read_text().splitlines() if path.exists() else []
        lines = lines[:2 * N_DESIGNS + 2]   # uniform sample size across every cell
        for i in range(2, len(lines) - 1, 2):
            if lines[i].startswith(">"):
                seq = lines[i + 1].strip()
                if len(seq) == length:
                    rows.append({"peptide": seq, "model": weights})
    path = ROOT / f"outputs/panel/{pdb}/full/esmif/seqs/{pdb}.fa"
    lines = path.read_text().splitlines() if path.exists() else []
    lines = lines[:2 * N_DESIGNS + 0]   # uniform sample size across every cell
    for i in range(0, len(lines) - 1, 2):
        if lines[i].startswith(">"):
            seq = lines[i + 1].strip()
            if len(seq) == length:
                rows.append({"peptide": seq, "model": "ESM-IF1"})
    path = ROOT / f"outputs/panel/{pdb}/full/ligandmpnn/seqs/{pdb}.fa"
    lines = path.read_text().splitlines() if path.exists() else []
    lines = lines[:2 * N_DESIGNS + 2]   # uniform sample size across every cell
    for i in range(2, len(lines) - 1, 2):
        if lines[i].startswith(">"):
            seq = peptide_from_ligandmpnn_line(lines[i + 1])
            if len(seq) == length:
                rows.append({"peptide": seq, "model": "LigandMPNN"})
    return pd.DataFrame(rows)

def shannon_entropy_local(counts):
    counts = np.array(counts, dtype=float)
    counts = counts[counts > 0]
    if counts.sum() == 0:
        return np.nan
    p = counts / counts.sum()
    return -(p * np.log2(p)).sum()

for rec in panel_pep_muts:
    pdb, length, pos0 = rec["pdb"], len(rec["peptide"]), rec["position"] - 1
    panel_designs = load_panel_designs(pdb, length)
    all_peps = panel_designs["peptide"].tolist()
    col = [p[pos0] for p in all_peps]
    rec["our_recovery"] = float(np.mean([aa == rec["peptide"][pos0] for aa in col]))
    rec["our_entropy"] = shannon_entropy_local(pd.Series(col).value_counts().values)
    rec["is_anchor"] = rec["position"] in (2, length)
    rec["n_designs"] = len(all_peps)

panel_pep_df = pd.DataFrame(panel_pep_muts)
panel_pep_df""")

md(r"""2BNR's tested position (9, the C-terminal P$\Omega$ anchor of this 9-mer) is the one clean
quantitative pair here: the native Cys is measured at 1.33e-05 M, the C9V mutant at 5.7e-06 M -- V binds
*tighter* than the crystallized native residue. If the native anchor residue isn't uniquely optimal for
affinity, that's a plausible piece of *why* this project keeps finding low recovery at $P\Omega$ across
every structure in the panel (notebook 03) -- not necessarily a model failure, but the position
tolerating (or even preferring) alternatives to the one residue that happened to get crystallized.
$n=4$ points, one per structure, is far too little to lean on this hard -- it's a lead worth keeping in
mind alongside the richer 3HG1 case study below, not a replacement for it.""")

md(r"""## 2. 3HG1 alanine scan: which peptide positions were tested, and what happened to binding

3HG1's native peptide is **ELAGIGILTV** (the MART-1/MelanA decapeptide, HLA-A2-restricted). Position 3
is already alanine in the wild type, so it can't be alanine-scanned in the usual sense. SKEMPI's
`Mutation(s)_PDB` codes chain C as the peptide -- filtering to single, peptide-only mutations isolates
a clean per-position readout (the combined peptide+TCR-CDR mutations below are also n.b., but that's
uninformative once the peptide side alone already abolishes binding).""")

co(r"""native_3hg1 = "ELAGIGILTV"
length_3hg1 = len(native_3hg1)

skempi_3hg1 = skempi[skempi.pdb_code == "3HG1"].copy()
skempi_3hg1["muts_list"] = skempi_3hg1["Mutation(s)_PDB"].str.split(",")
skempi_3hg1["chains"] = skempi_3hg1["muts_list"].apply(lambda ms: set(m[1] for m in ms))
skempi_3hg1["peptide_only"] = skempi_3hg1["chains"] == {"C"}

pep_only = skempi_3hg1[skempi_3hg1["peptide_only"]][
    ["Mutation(s)_PDB", "Affinity_mut (M)", "Affinity_wt (M)"]]
print(f"{len(skempi_3hg1)} total 3HG1 rows; {skempi_3hg1['peptide_only'].sum()} are peptide-only mutations")
pep_only""")

co(r"""tested_positions = {}  # 1-indexed peptide position -> outcome
for _, row in pep_only.iterrows():
    mut = row["Mutation(s)_PDB"]  # e.g. "GC4A" = native G, chain C, position 4, mutated to A
    pos = int(mut[2:-1])
    tested_positions[pos] = row["Affinity_mut (M)"]

CATEGORY = {}
for pos in range(1, length_3hg1 + 1):
    if pos in tested_positions:
        CATEGORY[pos] = "Ala abolishes binding (SKEMPI-tested)"
    elif native_3hg1[pos - 1] == "A":
        CATEGORY[pos] = "already Ala in WT (untestable)"
    elif pos in (2, length_3hg1):
        CATEGORY[pos] = "known MHC anchor (untested)"
    else:
        CATEGORY[pos] = "not tested"

for pos in range(1, length_3hg1 + 1):
    print(f"P{pos} ({native_3hg1[pos - 1]}): {CATEGORY[pos]}"
          + (f" -- affinity_mut={tested_positions[pos]}" if pos in tested_positions else ""))""")

md(r"""## 3. Our own 3HG1 designs: per-position recovery and entropy, 4 models, full context, T=0.1

Same pipeline convention as the panel notebooks -- raw design draws (not deduplicated), pMHC+TCR
context, T=0.1. 3HG1 isn't part of the 20-structure panel campaign, so its designs come from this
project's earlier full-matrix design campaign instead (`outputs/mpnn_3hg1_100k`,
`outputs/esmif_3hg1_pilot`, `outputs/ligandmpnn_3hg1_pilot`), same models, same temperature, same
full pMHC+TCR context.""")

co(r"""def peptide_from_ligandmpnn_line(line):
    return line.strip().split(":")[2]

def load_3hg1_designs():
    rows = []
    vanilla_path = ROOT / "outputs/mpnn_3hg1_100k/archive_T01_partial/vanilla_3HG1_T01_partial_26993.fa"
    nomhc_path = ROOT / "outputs/mpnn_3hg1_100k/archive_T01_partial/nomhc_3HG1_T01_partial_26528.fa"
    for weights, path in [("vanilla", vanilla_path), ("noMHC", nomhc_path)]:
        with open(path) as f:
            lines = f.read().splitlines()
        for i in range(2, len(lines) - 1, 2):  # skip the first (reference) record
            if lines[i].startswith(">"):
                seq = lines[i + 1].strip()
                if len(seq) == length_3hg1:
                    rows.append({"peptide": seq, "model": weights})

    esmif_path = ROOT / "outputs/esmif_3hg1_pilot/seqs/3HG1.fa"
    with open(esmif_path) as f:
        lines = f.read().splitlines()
    for i in range(0, len(lines) - 1, 2):
        if lines[i].startswith(">"):
            seq = lines[i + 1].strip()
            if len(seq) == length_3hg1:
                rows.append({"peptide": seq, "model": "ESM-IF1"})

    lig_path = ROOT / "outputs/ligandmpnn_3hg1_pilot/seqs/3HG1.fa"
    with open(lig_path) as f:
        lines = f.read().splitlines()
    for i in range(2, len(lines) - 1, 2):  # skip the first (reference) record
        if lines[i].startswith(">"):
            seq = peptide_from_ligandmpnn_line(lines[i + 1])
            if len(seq) == length_3hg1:
                rows.append({"peptide": seq, "model": "LigandMPNN"})
    return pd.DataFrame(rows)

designs_3hg1 = load_3hg1_designs()
print(designs_3hg1.groupby("model").size())""")

co(r"""def shannon_entropy(counts):
    counts = np.array(counts, dtype=float)
    counts = counts[counts > 0]
    if counts.sum() == 0:
        return np.nan
    p = counts / counts.sum()
    return -(p * np.log2(p)).sum()

per_position_records = []
for model in MODELS:
    peps = designs_3hg1.loc[designs_3hg1.model == model, "peptide"].tolist()
    for pos in range(length_3hg1):
        col = [p[pos] for p in peps]
        recovery = float(np.mean([aa == native_3hg1[pos] for aa in col]))
        counts = pd.Series(col).value_counts()
        entropy = shannon_entropy(counts.values)
        per_position_records.append({"position": pos + 1, "model": model, "recovery": recovery,
                                     "entropy": entropy})

# pooled: all 4 models' raw designs combined
all_peps = designs_3hg1["peptide"].tolist()
pooled_records = []
for pos in range(length_3hg1):
    col = [p[pos] for p in all_peps]
    recovery = float(np.mean([aa == native_3hg1[pos] for aa in col]))
    counts = pd.Series(col).value_counts()
    entropy = shannon_entropy(counts.values)
    pooled_records.append({"position": pos + 1, "recovery": recovery, "entropy": entropy})

per_position_df = pd.DataFrame(per_position_records)
pooled_df = pd.DataFrame(pooled_records)
pooled_df["category"] = pooled_df["position"].map(CATEGORY)
pooled_df["native_aa"] = pooled_df["position"].apply(lambda p: native_3hg1[p - 1])
pooled_df""")

md(r"""## 4. Cross-reference: does the alanine scan's verdict line up with our recovery/entropy?

Per-model (4 columns) and all-models-pooled (black), recovery and entropy, one plot per position --
colored/marked by SKEMPI category. If our metrics tracked real TCR-binding importance, the SKEMPI-tested
positions (which we *know* are functionally critical -- alanine there measurably kills binding) should
look distinct from the untested ones.""")

co(r"""CATEGORY_COLOR = {
    "Ala abolishes binding (SKEMPI-tested)": "crimson",
    "known MHC anchor (untested)": "#0072B2",
    "already Ala in WT (untestable)": "#888888",
    "not tested": "#CCCCCC",
}

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, ycol, ylabel in [(axes[0], "recovery", "recovery of native residue"),
                         (axes[1], "entropy", "entropy at this position (bits)")]:
    for model in MODELS:
        sub = per_position_df[per_position_df.model == model]
        ax.plot(sub["position"], sub[ycol], "o-", color=MODEL_COLOR[model], alpha=0.5, markersize=6,
               label=MODEL_LABEL[model])
    ax.plot(pooled_df["position"], pooled_df[ycol], "o-", color="black", linewidth=2.5, markersize=9,
           label="all 4 pooled", zorder=5)
    for _, row in pooled_df.iterrows():
        ax.axvspan(row["position"] - 0.4, row["position"] + 0.4,
                  color=CATEGORY_COLOR[row["category"]], alpha=0.15, zorder=0)
    ax.set_xticks(pooled_df["position"])
    ax.set_xticklabels([f"P{p}\n{native_3hg1[p - 1]}" for p in pooled_df["position"]])
    ax.set_xlabel("peptide position (native residue)")
    ax.set_ylabel(ylabel)
handles = [plt.Line2D([0], [0], color=c, lw=8, alpha=0.4, label=lbl) for lbl, c in CATEGORY_COLOR.items()]
axes[0].legend(fontsize=7, loc="upper left")
axes[1].legend(handles=handles, fontsize=7, loc="upper right", title="SKEMPI category")
fig.suptitle("3HG1: per-position recovery and entropy vs. SKEMPI alanine-scan category", y=1.03)
fig.tight_layout()
out = FIG_DIR / "fig_panel5_3hg1_recovery_entropy_vs_skempi.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"wrote {out}")""")

md(r"""### Design logo, all 4 models pooled, x-axis annotated with SKEMPI category""")

co(r"""mat = pd.DataFrame(0.0, index=range(1, length_3hg1 + 1), columns=AA)
for pep in all_peps:
    for pos, aa in enumerate(pep, start=1):
        if aa in AA:
            mat.loc[pos, aa] += 1.0
info_mat = logomaker.transform_matrix(mat, from_type="counts", to_type="information", pseudocount=0.1)

fig, ax = plt.subplots(figsize=(7, 4))
logomaker.Logo(info_mat, ax=ax, color_scheme="chemistry")
ax.set_ylim(0, np.log2(len(AA)))
ax.set_xticks(range(1, length_3hg1 + 1))
ax.set_xticklabels([f"P{p}\n{native_3hg1[p - 1]}" for p in range(1, length_3hg1 + 1)], fontsize=8)
for p in range(1, length_3hg1 + 1):
    ax.text(p, -0.55, "*" if CATEGORY[p].startswith("Ala abolishes") else "", ha="center", fontsize=14,
           color="crimson", clip_on=False)
ax.set_ylabel("bits")
ax.set_title(f"3HG1 design logo, all 4 models pooled (n={len(all_peps):,})\n"
            "* = SKEMPI-confirmed: alanine here abolishes TCR-pMHC binding", fontsize=10)
fig.tight_layout()
out = FIG_DIR / "fig_panel5_3hg1_design_logo.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"wrote {out}")""")

md(r"""## 5. Statistical comparison: SKEMPI-validated TCR-critical positions vs. everything else

$n=5$ tested positions vs. $n=5$ untested, all-models-pooled recovery and entropy -- Mann-Whitney U
(unpaired, since these are different positions, not matched samples). With $n=5$ per group this is
underpowered and exploratory, not a confirmatory test; treat the p-value as a rough signal, not proof.""")

co(r"""tested_pos = [p for p, c in CATEGORY.items() if c.startswith("Ala abolishes")]
other_pos = [p for p, c in CATEGORY.items() if not c.startswith("Ala abolishes")]
print(f"SKEMPI-critical positions: {tested_pos} ({[native_3hg1[p - 1] for p in tested_pos]})")
print(f"everything else: {other_pos} ({[native_3hg1[p - 1] for p in other_pos]})")

for ycol in ["recovery", "entropy"]:
    tested_vals = pooled_df.loc[pooled_df.position.isin(tested_pos), ycol].values
    other_vals = pooled_df.loc[pooled_df.position.isin(other_pos), ycol].values
    stat, p = mannwhitneyu(tested_vals, other_vals, alternative="two-sided")
    print(f"\n{ycol}: SKEMPI-critical mean={tested_vals.mean():.3f}, everything-else mean="
          f"{other_vals.mean():.3f}, Mann-Whitney p={p:.3f} (n=5 vs n=5)")

print("\n--- and the two known MHC anchors specifically (P2, P10) vs. the 5 SKEMPI-critical positions ---")
anchor_pos = [p for p, c in CATEGORY.items() if c == "known MHC anchor (untested)"]
for ycol in ["recovery", "entropy"]:
    anchor_vals = pooled_df.loc[pooled_df.position.isin(anchor_pos), ycol].values
    tested_vals = pooled_df.loc[pooled_df.position.isin(tested_pos), ycol].values
    print(f"{ycol}: anchors {dict(zip(anchor_pos, anchor_vals.round(3)))}, "
          f"SKEMPI-critical mean={tested_vals.mean():.3f}")""")

md(r"""## 6. Do the models actually pick up the TCR-recognition motif, and does removing the TCR erode it?

Two separate questions, both usable now that SKEMPI has told us which 5 positions (P4, P6, P7, P8, P9)
are confirmed TCR-critical, independent of the pMHC anchoring story (P2, P$\Omega$):

1. **Do the models understand this motif?** Recovery (matching the native residue) is one signal, but a
   stronger one is *cross-model agreement*: if four independently-trained, architecturally different
   models (ProteinMPNN, ProteinMPNN (no MHC), ESM-IF1, LigandMPNN) all converge on the same residue at a
   position with no coordination between them, that position is carrying a real, learnable constraint --
   not a quirk of one model's training. Measured here as each model's own single most-common (mode)
   residue at a position, then how many of the models' modes agree with each other.
2. **Does removing the TCR erode it?** If the TCR-critical positions are actually learned as
   TCR-recognition constraints (not just generic peptide stability), removing the TCR from the design
   context should hurt recovery/agreement at P4/6/7/8/9 specifically, while leaving the pMHC-anchor
   positions (P2, P$\Omega$) comparatively unaffected -- that contrast is the actual test, not just "does
   mhconly recovery drop somewhere."

Only 3 of the 4 models have a true pMHC-only (no TCR, MHC still present) run for 3HG1 in this project's
earlier campaign -- ProteinMPNN, ProteinMPNN (no MHC), and LigandMPNN. ESM-IF1 only has a "nocontext"
run (MHC *and* TCR both removed), which would confound the comparison, so ESM-IF1 is excluded from this
section specifically (it's still in every other section of this notebook).""")

co(r"""MHCONLY_MODELS = ["vanilla", "noMHC", "LigandMPNN"]

def load_3hg1_mhconly_designs():
    rows = []
    vanilla_path = ROOT / "outputs/context_ablation_3hg1/mpnn_mhconly/seqs/vanilla_3HG1_mhconly.fa"
    nomhc_path = ROOT / "outputs/context_ablation_3hg1/mpnn_mhconly/seqs/nomhc_3HG1_mhconly.fa"
    for weights, path in [("vanilla", vanilla_path), ("noMHC", nomhc_path)]:
        with open(path) as f:
            lines = f.read().splitlines()
        for i in range(2, len(lines) - 1, 2):
            if lines[i].startswith(">"):
                seq = lines[i + 1].strip()
                if len(seq) == length_3hg1:
                    rows.append({"peptide": seq, "model": weights})

    lig_path = ROOT / "outputs/context_ablation_3hg1/ligandmpnn_mhconly/seqs/3HG1.fa"
    with open(lig_path) as f:
        lines = f.read().splitlines()
    for i in range(2, len(lines) - 1, 2):
        if lines[i].startswith(">"):
            seq = peptide_from_ligandmpnn_line(lines[i + 1])
            if len(seq) == length_3hg1:
                rows.append({"peptide": seq, "model": "LigandMPNN"})
    return pd.DataFrame(rows)

designs_3hg1_mhconly = load_3hg1_mhconly_designs()
print(designs_3hg1_mhconly.groupby("model").size())

def model_mode_agreement(peps_by_model, length):
    records = []
    for pos in range(length):
        modes = {model: pd.Series([p[pos] for p in peps]).value_counts().idxmax()
                for model, peps in peps_by_model.items()}
        vote_counts = pd.Series(list(modes.values())).value_counts()
        records.append({"position": pos + 1, "consensus_aa": vote_counts.index[0],
                        "n_agree": int(vote_counts.iloc[0]), "n_models": len(peps_by_model),
                        "modes": modes})
    return pd.DataFrame(records)

def pooled_recovery_by_position(peps_by_model, native, length):
    all_peps = [p for peps in peps_by_model.values() for p in peps]
    return [float(np.mean([p[pos] == native[pos] for p in all_peps])) for pos in range(length)]

peps_full_3model = {m: designs_3hg1.loc[designs_3hg1.model == m, "peptide"].tolist() for m in MHCONLY_MODELS}
peps_mhconly_3model = {m: designs_3hg1_mhconly.loc[designs_3hg1_mhconly.model == m, "peptide"].tolist()
                       for m in MHCONLY_MODELS}

agreement_full = model_mode_agreement(peps_full_3model, length_3hg1)
agreement_mhconly = model_mode_agreement(peps_mhconly_3model, length_3hg1)
recovery_full_3model = pooled_recovery_by_position(peps_full_3model, native_3hg1, length_3hg1)
recovery_mhconly_3model = pooled_recovery_by_position(peps_mhconly_3model, native_3hg1, length_3hg1)

shift_df = pd.DataFrame({
    "position": range(1, length_3hg1 + 1),
    "native_aa": list(native_3hg1),
    "category": [CATEGORY[p] for p in range(1, length_3hg1 + 1)],
    "recovery_full": recovery_full_3model,
    "recovery_mhconly": recovery_mhconly_3model,
    "n_agree_full": agreement_full["n_agree"],
    "n_agree_mhconly": agreement_mhconly["n_agree"],
})
shift_df["recovery_delta"] = shift_df["recovery_full"] - shift_df["recovery_mhconly"]
shift_df["agreement_delta"] = shift_df["n_agree_full"] - shift_df["n_agree_mhconly"]
shift_df""")

md(r"""### Recovery and cross-model agreement, full vs. pMHC-only, per position""")

co(r"""fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
x = shift_df["position"]
axes[0].plot(x, shift_df["recovery_full"], "o-", color="#4C72B0", label="pMHC+TCR")
axes[0].plot(x, shift_df["recovery_mhconly"], "o-", color="#999999", label="pMHC only")
axes[0].set_ylabel("recovery of native residue\n(3 shared models pooled)")
axes[0].legend(fontsize=8)

axes[1].plot(x, shift_df["n_agree_full"], "o-", color="#4C72B0", label="pMHC+TCR")
axes[1].plot(x, shift_df["n_agree_mhconly"], "o-", color="#999999", label="pMHC only")
axes[1].set_ylabel("# models (of 3) agreeing on the\nsame consensus residue")
axes[1].set_yticks([1, 2, 3])
axes[1].legend(fontsize=8)

for ax in axes:
    for _, row in shift_df.iterrows():
        ax.axvspan(row["position"] - 0.4, row["position"] + 0.4,
                  color=CATEGORY_COLOR[row["category"]], alpha=0.15, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"P{p}\n{native_3hg1[p - 1]}" for p in x])
    ax.set_xlabel("peptide position (native residue)")
fig.suptitle("3HG1: does removing the TCR erode recovery/agreement, and where?", y=1.02)
fig.tight_layout()
out = FIG_DIR / "fig_panel5_3hg1_full_vs_mhconly_shift.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"wrote {out}")""")

md(r"""### Is the shift concentrated at the SKEMPI-confirmed TCR-critical positions?

If the TCR-critical positions are genuinely learned as TCR-recognition constraints (not generic peptide
stability), the full-to-mhconly *drop* in recovery/agreement should be larger there than at the
pMHC-anchor positions or the untested ones. $n=5$ vs $n=5$ (or $n=2$ for the anchor-only comparison) --
small, exploratory, not confirmatory.""")

co(r"""tested_pos = [p for p, c in CATEGORY.items() if c.startswith("Ala abolishes")]
anchor_pos = [p for p, c in CATEGORY.items() if c == "known MHC anchor (untested)"]
other_pos = [p for p, c in CATEGORY.items() if not c.startswith("Ala abolishes")]

for col in ["recovery_delta", "agreement_delta"]:
    tested_vals = shift_df.loc[shift_df.position.isin(tested_pos), col].values
    other_vals = shift_df.loc[shift_df.position.isin(other_pos), col].values
    anchor_vals = shift_df.loc[shift_df.position.isin(anchor_pos), col].values
    stat, p = mannwhitneyu(tested_vals, other_vals, alternative="two-sided")
    print(f"{col}: SKEMPI-critical mean={tested_vals.mean():+.3f} (positions {tested_pos}), "
         f"everything-else mean={other_vals.mean():+.3f}, Mann-Whitney p={p:.3f}")
    print(f"  -- for reference, the 2 known MHC anchors alone: {dict(zip(anchor_pos, anchor_vals.round(3)))}")
print("\n(positive delta = pMHC+TCR context recovers/agrees more than pMHC-only; "
     "negative = pMHC-only does better)")""")

md(r"""## 7. IEDB: real TCR-binding measurements across the whole panel

**A caveat first, because it matters for how to read everything below:** almost none of these
measurements can cleanly separate "this mutation broke peptide-MHC loading" from "this mutation broke
TCR recognition specifically." A `cytotoxicity`, `activation`, `IL-2 release`, or `proliferation` result
is a net functional readout -- it goes negative if the peptide stops binding the MHC well enough to be
presented, *or* if it's presented fine but the TCR no longer recognizes it, and IEDB's abstracted
`qualitative_measure` field doesn't distinguish the two. `qualitative binding|multimer/tetramer` is a bit
cleaner (a multimer has to be a stable, correctly folded pMHC to even be made), but is still a yes/no
stain, not a binding measurement. The one assay type here that's actually run *after* the pMHC complex
is already confirmed stably folded is `dissociation constant KD|surface plasmon resonance (SPR)` --
refolding has to succeed before an SPR experiment is even possible, so a KD change measured this way is
the closest thing in this dataset to an isolated TCR-engagement effect. Where it matters below, the
assay type is called out explicitly rather than treating every "Negative" the same.

Using the 13 TCR clones from Section 1, matched by CDR3 back to IEDB's `tcell_search`, then re-diffed
against each individual panel structure's own native peptide (accounting for a shared background
mutation where one exists, e.g. 5EU6's variants are all also P9A) -- 44 (structure, position) pairs
across the panel have at least one real single-position IEDB variant tested, and 21 of those show a
"Negative" or "Positive-Low" result somewhere: real evidence that position matters for binding.""")

co(r"""master_full = pd.read_csv(ROOT / "outputs/analysis/panel_dataset_master_table.csv").set_index("pdb")

GROUP_MAP = {
    "1OGA": "1OGA", "2VLR": "2VLR", "5HHO": "5HHO", "3QEQ": "3QEQ",
    "3UTS": "3UTS,5C0A,5C0B", "5C0A": "3UTS,5C0A,5C0B", "5C0B": "3UTS,5C0A,5C0B",
    "3GSN": "3GSN", "4MJI": "4MJI", "5EU6": "5EU6", "3QDG": "3QDG", "2P5W": "2P5W",
    "2BNR": "2BNR,2F53,2F54", "2F53": "2BNR,2F53,2F54", "2F54": "2BNR,2F53,2F54",
    "5NME": "5NME", "1QSF": "1QSF,1QRN,2GJ6,3QFJ", "1QRN": "1QSF,1QRN,2GJ6,3QFJ",
    "2GJ6": "1QSF,1QRN,2GJ6,3QFJ", "3QFJ": "1QSF,1QRN,2GJ6,3QFJ",
}
with open(ROOT / "inputs/iedb/tcr_matched_raw.json") as f:
    iedb_raw = json.load(f)

per_pdb_variants = {}
for pdb, group in GROUP_MAP.items():
    native = master_full.loc[pdb, "peptide"]
    length = len(native)
    variants = []
    for r in iedb_raw[group]:
        pep = r["linear_sequence"]
        if len(pep) != length:
            continue
        diffs = tuple(i + 1 for i, (a, b) in enumerate(zip(native, pep)) if a != b)
        if not diffs:
            continue
        variants.append({"diffs": diffs, "qual": r["qualitative_measure"], "assay": r["assay_names"],
                         "pubmed_id": r["pubmed_id"]})
    per_pdb_variants[pdb] = variants

# Section 1b (SKEMPI) already presented these exact measurements, from these exact papers -- reusing
# them here via IEDB, unlabeled, would silently double-count the same experiment as if it were a second,
# independent source. Flag (not silently drop) any IEDB row citing the same PMID.
SKEMPI_PANEL_PMIDS = {"1QRN": "10435578", "1QSF": "10435578", "2BNR": "15837811", "3QFJ": "22019736"}

position_records = []
for pdb, variants in per_pdb_variants.items():
    native = master_full.loc[pdb, "peptide"]
    multi = [v for v in variants if len(v["diffs"]) > 1]
    background_pos = None
    if multi:
        common = set(multi[0]["diffs"])
        for v in multi[1:]:
            common &= set(v["diffs"])
        if len(common) == 1:
            background_pos = next(iter(common))
    for v in variants:
        diffs = v["diffs"]
        if len(diffs) == 1:
            tested = diffs[0]
        elif background_pos is not None and background_pos in diffs and len(diffs) == 2:
            tested = [d for d in diffs if d != background_pos][0]
        else:
            continue
        same_as_skempi = v["pubmed_id"] == SKEMPI_PANEL_PMIDS.get(pdb)
        position_records.append({"pdb": pdb, "position": tested, "native_aa": native[tested - 1],
                                 "qualitative_measure": v["qual"], "assay": v["assay"],
                                 "pubmed_id": v["pubmed_id"], "same_source_as_skempi_1b": same_as_skempi})

variant_df = pd.DataFrame(position_records).drop_duplicates()
SEVERITY = {"Negative": 0, "Positive-Low": 1, "Positive-Intermediate": 2, "Positive": 3, "Positive-High": 4}
variant_df["severity"] = variant_df["qualitative_measure"].map(SEVERITY)

position_summary = variant_df.groupby(["pdb", "position", "native_aa"]).agg(
    worst_severity=("severity", "min"),
    quals=("qualitative_measure", lambda s: sorted(set(s))),
    any_spr=("assay", lambda s: s.str.contains("KD|SPR|dissociation", case=False).any()),
    same_source_as_skempi_1b=("same_source_as_skempi_1b", "any"),
    pubmed_ids=("pubmed_id", lambda s: sorted(set(str(x) for x in s))),
).reset_index()
position_summary["functional_change"] = position_summary["worst_severity"] <= 1  # Negative or Positive-Low
n_dup = position_summary.query("functional_change and same_source_as_skempi_1b").shape[0]
print(f"{len(position_summary)} tested (structure, position) pairs; "
     f"{position_summary['functional_change'].sum()} show Negative/Positive-Low somewhere "
     f"({position_summary.query('functional_change and any_spr').shape[0]} of those from an SPR/KD assay)")
print(f"\n*** {n_dup} of those functional-change positions cite the SAME paper already shown in "
     f"Section 1b (SKEMPI) -- these are NOT independent confirmations, they're the same experiment "
     f"pulled from a second database. Marked, not dropped, below. ***")
position_summary""")

md(r"""### Matrix: which position, in which structure, has a mutation that changes TCR binding

Rows = peptide position (1 = N-terminus); columns = structure, grouped by TCR clone. Color = the worst
(most severity-reducing) result found among all single-position variants tested at that cell; white =
not tested by any study in this dataset.""")

co(r"""SEVERITY_COLOR = {0: "#B22222", 1: "#E69F00", 2: "#F0E442", 3: "#66BD63", 4: "#1A9850"}
pdb_order = sorted(GROUP_MAP.keys(), key=lambda p: (GROUP_MAP[p], p))
max_len = max(len(master_full.loc[p, "peptide"]) for p in pdb_order)

fig, ax = plt.subplots(figsize=(0.6 * len(pdb_order) + 2, 0.55 * max_len + 1.5))
grid = position_summary.set_index(["pdb", "position"])
for j, pdb in enumerate(pdb_order):
    for pos in range(1, max_len + 1):
        if pos > len(master_full.loc[pdb, "peptide"]):
            color = "#EEEEEE"
        elif (pdb, pos) in grid.index:
            sev = grid.loc[(pdb, pos), "worst_severity"]
            color = SEVERITY_COLOR[sev]
        else:
            color = "white"
        ax.add_patch(plt.Rectangle((j, max_len - pos), 1, 1, facecolor=color, edgecolor="lightgray"))
        if (pdb, pos) in grid.index:
            aa = grid.loc[(pdb, pos), "native_aa"]
            dagger = " †" if grid.loc[(pdb, pos), "same_source_as_skempi_1b"] else ""
            ax.text(j + 0.5, max_len - pos + 0.5, aa + dagger, ha="center", va="center", fontsize=7)
ax.set_xlim(0, len(pdb_order))
ax.set_ylim(0, max_len)
ax.set_xticks(np.arange(len(pdb_order)) + 0.5)
ax.set_xticklabels(pdb_order, rotation=90, fontsize=8)
ax.set_yticks(np.arange(max_len) + 0.5)
ax.set_yticklabels([f"P{p}" for p in range(max_len, 0, -1)], fontsize=8)
ax.set_xlabel("structure")
handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, label=q)
          for q, c in zip(["Negative", "Positive-Low", "Positive-Intermediate", "Positive", "Positive-High"],
                          SEVERITY_COLOR.values())]
ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, title="worst result")
ax.set_title("IEDB single-position variant results, per structure\n"
            "(blank = not tested; letter = native residue; † = same source as Section 1b/SKEMPI)")
fig.tight_layout()
out = FIG_DIR / "fig_panel5_iedb_position_matrix.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"wrote {out}")""")

md(r"""### Recovery at those positions: pMHC only vs. pMHC+TCR, pooled

Every position flagged `functional_change` above (Negative or Positive-Low result), cross-referenced
against our own panel designs (T=0.1). Faint points/lines: one per (structure, position), full vs.
mhconly recovery. Bold point/line: all raw designs at all of these positions pooled together (not an
average of the per-position fractions) for the mean, with the SD **across the different positions**
(not within-design binomial noise) shown as the error bar -- so the error bar reflects how much these
positions disagree with each other, not sampling noise at any single one.""")

co(r"""def load_panel_designs_cond(pdb, length, cond):
    rows = []
    for weights, fname in [("vanilla", f"vanilla_{pdb}.fa"), ("noMHC", f"nomhc_{pdb}.fa")]:
        path = ROOT / f"outputs/panel/{pdb}/{cond}/mpnn/seqs/{fname}"
        lines = path.read_text().splitlines() if path.exists() else []
        lines = lines[:2 * N_DESIGNS + 2]   # uniform sample size across every cell
        for i in range(2, len(lines) - 1, 2):
            if lines[i].startswith(">"):
                seq = lines[i + 1].strip()
                if len(seq) == length:
                    rows.append(seq)
    path = ROOT / f"outputs/panel/{pdb}/{cond}/esmif/seqs/{pdb}.fa"
    lines = path.read_text().splitlines() if path.exists() else []
    lines = lines[:2 * N_DESIGNS + 0]   # uniform sample size across every cell
    for i in range(0, len(lines) - 1, 2):
        if lines[i].startswith(">"):
            seq = lines[i + 1].strip()
            if len(seq) == length:
                rows.append(seq)
    path = ROOT / f"outputs/panel/{pdb}/{cond}/ligandmpnn/seqs/{pdb}.fa"
    lines = path.read_text().splitlines() if path.exists() else []
    lines = lines[:2 * N_DESIGNS + 2]   # uniform sample size across every cell
    for i in range(2, len(lines) - 1, 2):
        if lines[i].startswith(">"):
            seq = peptide_from_ligandmpnn_line(lines[i + 1])
            if len(seq) == length:
                rows.append(seq)
    return rows

critical = position_summary[position_summary.functional_change].copy()
pooled_full_hits, pooled_mhconly_hits = [], []
per_position_recovery = []
for _, row in critical.iterrows():
    pdb, pos, native_aa = row["pdb"], row["position"], row["native_aa"]
    length = len(master_full.loc[pdb, "peptide"])
    full_peps = load_panel_designs_cond(pdb, length, "full")
    mhconly_peps = load_panel_designs_cond(pdb, length, "mhconly")
    full_hits = [p[pos - 1] == native_aa for p in full_peps]
    mhconly_hits = [p[pos - 1] == native_aa for p in mhconly_peps]
    pooled_full_hits.extend(full_hits)
    pooled_mhconly_hits.extend(mhconly_hits)
    per_position_recovery.append({"pdb": pdb, "position": pos, "native_aa": native_aa,
                                  "recovery_full": np.mean(full_hits), "recovery_mhconly": np.mean(mhconly_hits),
                                  "same_source_as_skempi_1b": row["same_source_as_skempi_1b"],
                                  "n_full": len(full_hits), "n_mhconly": len(mhconly_hits)})

per_position_df = pd.DataFrame(per_position_recovery)
# real, self-explanatory label: structure, position, native residue, and an explicit dagger on any
# entry that's the same published measurement already shown in Section 1b (SKEMPI) -- not a second,
# independent source, so it must not be visually indistinguishable from the ones that are.
per_position_df["label"] = (per_position_df["pdb"] + " P" + per_position_df["position"].astype(str)
                            + " (" + per_position_df["native_aa"] + ")"
                            + per_position_df["same_source_as_skempi_1b"].map({True: " †", False: ""}))
print(per_position_df.drop(columns=["n_full", "n_mhconly"]).to_string(index=False))
print(f"\n† = same source publication already presented in Section 1b (SKEMPI) -- "
     f"not an independent confirmation, shown here for completeness")

is_dup = per_position_df["same_source_as_skempi_1b"]
def pooled_stats(mask):
    full_hits, mhconly_hits = [], []
    for _, row in per_position_df[mask].iterrows():
        pdb, pos, native_aa = row["pdb"], row["position"], row["native_aa"]
        length = len(master_full.loc[pdb, "peptide"])
        full_hits.extend([p[pos - 1] == native_aa for p in load_panel_designs_cond(pdb, length, "full")])
        mhconly_hits.extend([p[pos - 1] == native_aa for p in load_panel_designs_cond(pdb, length, "mhconly")])
    return np.mean(full_hits), np.mean(mhconly_hits)

pooled_mean_full = np.mean(pooled_full_hits)
pooled_mean_mhconly = np.mean(pooled_mhconly_hits)
sd_across_positions_full = per_position_df["recovery_full"].std()
sd_across_positions_mhconly = per_position_df["recovery_mhconly"].std()
print(f"\npooled, ALL {len(critical)} functional-change positions (includes {is_dup.sum()} SKEMPI-duplicate): "
     f"full={pooled_mean_full:.3f} (SD across positions={sd_across_positions_full:.3f}), "
     f"mhconly={pooled_mean_mhconly:.3f} (SD across positions={sd_across_positions_mhconly:.3f})")

clean_full, clean_mhconly = pooled_stats(~is_dup)
clean_sd_full = per_position_df.loc[~is_dup, "recovery_full"].std()
clean_sd_mhconly = per_position_df.loc[~is_dup, "recovery_mhconly"].std()
print(f"pooled, EXCLUDING the {is_dup.sum()} SKEMPI-duplicate positions "
     f"({(~is_dup).sum()} genuinely IEDB-only positions): "
     f"full={clean_full:.3f} (SD across positions={clean_sd_full:.3f}), "
     f"mhconly={clean_mhconly:.3f} (SD across positions={clean_sd_mhconly:.3f})")""")

co(r"""fig, ax = plt.subplots(figsize=(7.5, 6.5))
rng = np.random.RandomState(0)
xs = {"mhconly": 0.0, "full": 1.0}
for _, row in per_position_df.iterrows():
    jitter = rng.uniform(-0.03, 0.03)
    color = "#CC3333" if row["same_source_as_skempi_1b"] else "#AAAAAA"
    ax.plot([xs["mhconly"] + jitter, xs["full"] + jitter],
           [row["recovery_mhconly"], row["recovery_full"]], color=color, alpha=0.7, linewidth=1,
           zorder=2, marker="o", markersize=4)

ax.errorbar([xs["mhconly"], xs["full"]], [pooled_mean_mhconly, pooled_mean_full],
           yerr=[sd_across_positions_mhconly, sd_across_positions_full], color="black", linewidth=2.5,
           marker="o", markersize=10, capsize=6, zorder=5,
           label=f"pooled, all {len(critical)} functional-change positions")
ax.plot([], [], color="#CC3333", marker="o", markersize=4, linewidth=1,
       label="same source as Section 1b (SKEMPI) -- not independent")
ax.plot([], [], color="#AAAAAA", marker="o", markersize=4, linewidth=1, label="IEDB-only")

ax.set_xticks([0, 1])
ax.set_xticklabels(["pMHC only", "pMHC+TCR"])
ax.set_xlim(-0.3, 1.3)
ax.set_ylabel("recovery of native residue at that position")
ax.set_title("Recovery at IEDB-confirmed functional-change positions\n"
             "faint = individual (structure, position); bold = pooled designs, SD across positions")
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
out = FIG_DIR / "fig_panel5_iedb_critical_position_recovery_shift.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"wrote {out}")""")

md(r"""### 3-row summary matrix: IEDB result, pMHC-only recovery, pMHC+TCR recovery

One column per (structure, position) functional-change entry, plus a MEAN column. Top row is IEDB's
categorical severity (Negative=0 through Positive-High=4) -- not a literal $\Delta$KD in nM, since the
API doesn't expose numeric KD values for most of these entries, only the qualitative bin. Middle and
bottom rows are our own recovery, pMHC only vs. pMHC+TCR, same as the point plot above but laid out
position-by-position instead of as a distribution.""")

co(r"""from matplotlib.colors import LinearSegmentedColormap

matrix_df = per_position_df.merge(critical[["pdb", "position", "worst_severity"]], on=["pdb", "position"])
matrix_df = matrix_df.sort_values(["pdb", "position"]).reset_index(drop=True)

labels = matrix_df["label"].tolist() + ["MEAN"]
row_severity = matrix_df["worst_severity"].tolist() + [matrix_df["worst_severity"].mean()]
row_mhconly = matrix_df["recovery_mhconly"].tolist() + [matrix_df["recovery_mhconly"].mean()]
row_full = matrix_df["recovery_full"].tolist() + [matrix_df["recovery_full"].mean()]
n = len(labels)

severity_cmap = LinearSegmentedColormap.from_list(
    "severity", [SEVERITY_COLOR[i] for i in range(5)])

fig, axes = plt.subplots(3, 1, figsize=(0.55 * n + 2, 6.5), sharex=True)
row_specs = [
    (row_severity, severity_cmap, 0, 4, "IEDB severity\n(0=Negative,\n4=Positive-High)", "{:.1f}"),
    (row_mhconly, "viridis", 0, 1, "recovery,\npMHC only", "{:.2f}"),
    (row_full, "viridis", 0, 1, "recovery,\npMHC+TCR", "{:.2f}"),
]
for ax, (values, cmap, vmin, vmax, ylabel, fmt) in zip(axes, row_specs):
    arr = np.array(values).reshape(1, -1)
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for j, v in enumerate(values):
        norm = (v - vmin) / (vmax - vmin)
        color = "white" if norm < 0.5 else "black"
        ax.text(j, 0, fmt.format(v), ha="center", va="center", color=color, fontsize=8)
    ax.axvline(n - 1.5, color="black", linewidth=1.5)
    ax.set_yticks([0])
    ax.set_yticklabels([ylabel], fontsize=8)
    ax.set_xticks([])
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

axes[-1].set_xticks(range(n))
axes[-1].set_xticklabels(labels, rotation=90, fontsize=8)
for tick, is_dup in zip(axes[-1].get_xticklabels(), matrix_df["same_source_as_skempi_1b"].tolist() + [False]):
    if is_dup:
        tick.set_color("#CC3333")
fig.suptitle("IEDB functional-change severity vs. our own recovery, per position\n"
            "(red label = same source as Section 1b/SKEMPI, not independent)", y=0.99)
fig.tight_layout()
out = FIG_DIR / "fig_panel5_iedb_severity_recovery_matrix.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"wrote {out}")""")

md(r"""## Synthesis

**Scope, stated plainly:** the panel's own SKEMPI overlap (Section 1b) is real but thin -- 4 structures,
1 peptide mutation each. It already shows a genuine anchor-tolerance lead (2BNR: native Cys at
$P\Omega$ measured *weaker* than a Val substitution) and a clean split in our own recovery at those 4
single positions (0.95 down to 0.0), but $n=4$ points can't support a position-category analysis the way
3HG1's 5-position scan can. Everything from Section 2 onward is 3HG1 -- richer, but outside the panel.

The SKEMPI alanine scan didn't touch the two positions we already suspected were special for a
*different* reason (P2 and P10, the canonical MHC pockets B/F anchors) -- those were presumably excluded
from a TCR-affinity assay specifically because mutating them risks killing peptide loading/presentation
itself, confounding a "TCR contact" readout. What it *did* test (P4, P6, P7, P8, P9) are the
TCR-facing, solvent-exposed epitope positions, and alanine there measurably abolishes MEL5 binding at
every one of them -- a real, independent confirmation that this specific set of 5 positions matters for
TCR recognition, not just a contact-count proxy.

Whether our own recovery/entropy actually distinguishes that confirmed-critical set from everything else
is the real question this notebook answers with actual numbers (see Section 5's printed output) rather
than assuming it from the panel's contact-based analysis alone. Either way, the honest caveat from
Section 2 stands: every SKEMPI outcome here is categorical (n.b. = binding abolished), not a graded
$\Delta\Delta G$ -- so this validates *identity* of important positions, not a dose-response, and $n=5$
per group is a small, exploratory comparison, not a confirmatory one.

**Section 6's group-level test (SKEMPI-critical vs. everything else) didn't reach significance** (recovery
delta p=0.40, agreement delta p=0.18, $n=5$ vs $n=5$) -- but the group average hides a floor effect: P8
and P9 are already near-zero recovery in the *full* context, so removing the TCR can't drop them much
further, which drags the group mean down. Look at the individual positions instead (Section 6's plot):
**P4 collapses from ~98% recovery (pMHC+TCR) to ~0% (pMHC only)** -- as clean a TCR-dependence signature
as this project has produced anywhere. **P5 shows the identical collapse (~68% to 0%)**, despite SKEMPI
never testing P5 at all -- this notebook's own data is proposing P5 as a TCR-recognition position that
the published alanine scan simply didn't happen to cover, a concrete, testable hypothesis rather than a
speculation. By contrast, the two MHC anchors (P2, P10) barely move between contexts (P2 even ticks
slightly *up* without the TCR) -- exactly the contrast this analysis was designed to detect: TCR-driven
positions lose their identity when the TCR is removed, pMHC-anchor positions don't, because they were
never the TCR's business in the first place.""")

md(r"""## 6. The same test, pooled over every SKEMPI complex we have designs for

Sections 4 and 5 lean on 3HG1's five-position alanine scan, which is the only scan that maps onto
the structures this project designs on. That is a real limitation: $n=5$ positions in one crystal.

It no longer applies. Designs now exist for 28 SKEMPI complexes, so the measured binding effects can
be joined to per-position recovery directly, across many crystals at once.

Construction, and what is deliberately thrown away:

- only **single** mutations, and only in the **epitope chain** of that complex (SKEMPI names chains
  by their deposited letters, which differ per entry, so the epitope chain is taken from the design
  manifest rather than assumed to be C);
- three complexes (3C60, 4P23, 4P5T) fuse the epitope to the MHC in one chain, so a mutation in the
  epitope chain is not necessarily *in the epitope*. Positions are matched against the manifest's
  explicit epitope residue ids, which drops the 45 mutations that turn out to be MHC residues;
- $\Delta\Delta G = RT\ln(K_{d,\mathrm{mut}}/K_{d,\mathrm{wt}})$, positive meaning the mutation
  **weakens** binding, so a large value marks a position the interface actually depends on;
- where a position was mutated more than once, the largest effect is kept.

Two questions, one of them the claim this project makes: is recovery higher at positions that matter,
and is the *benefit of having the TCR present* concentrated there?""")

co(r"""RT = 0.001987 * 298.15   # kcal/mol at 298 K

manifest = pd.read_csv(ROOT / "designs/skempi/t01/manifest.csv")
pep_chain = dict(zip(manifest["complex"], manifest["pep_chain"]))
pep_resids = {r.complex: [s.strip() for s in str(r.pep_resids).split(";")]
              for r in manifest.itertuples()}

mut = skempi[skempi["#Pdb"].isin(pep_chain)].copy()
mut["muts"] = mut["Mutation(s)_PDB"].str.split(",")
mut = mut[mut["muts"].str.len() == 1].copy()
mut["code"] = mut["muts"].str[0]
mut = mut[[c[1] == pep_chain[p] for c, p in zip(mut["code"], mut["#Pdb"])]]
mut = mut.dropna(subset=["Affinity_mut_parsed", "Affinity_wt_parsed"])
mut["ddG"] = RT * np.log(mut["Affinity_mut_parsed"] / mut["Affinity_wt_parsed"])
mut = mut.rename(columns={"#Pdb": "complex"})

# epitope position (1-indexed) or NaN if the mutated residue is not in the epitope at all
mut["pos"] = [pep_resids[c].index(code[2:-1]) + 1 if code[2:-1] in pep_resids[c] else np.nan
              for c, code in zip(mut["complex"], mut["code"])]
n_off = int(mut["pos"].isna().sum())
mut = mut.dropna(subset=["pos"])
mut["pos"] = mut["pos"].astype(int)
print(f"{len(mut) + n_off} single epitope-chain mutations with measured affinities; "
      f"{n_off} sit outside the epitope itself (fused MHC-epitope chains) and are dropped")

muts = mut.groupby(["complex", "pos"])["ddG"].max().reset_index()
print(f"{len(muts)} distinct (complex, position) measurements over "
      f"{muts['complex'].nunique()} complexes "
      f"-- compare {5} positions in one crystal in Sections 4-5")""")

co(r"""perpos = pd.read_parquet(ROOT / "outputs/design_corpus_perpos.parquet")
perpos = perpos[perpos.dataset == "skempi"]
perpos = (perpos.groupby(["complex", "arm", "pos"], observed=True)["recovery"]
                .mean().unstack("arm").reset_index())

d = muts.merge(perpos, on=["complex", "pos"], how="inner")
d["tcr_benefit"] = d["full"] - d["mhconly"]
CRIT = 1.0   # kcal/mol, the conventional "this mutation matters" threshold
d["critical"] = d["ddG"] >= CRIT
print(f"{len(d)} positions joined to recovery; "
      f"{int(d.critical.sum())} critical (ddG >= {CRIT} kcal/mol), {int((~d.critical).sum())} not\n")

for col, label in [("full", "recovery, pMHC+TCR"),
                   ("mhconly", "recovery, pMHC only"),
                   ("tcr_benefit", "TCR benefit (full - mhconly)")]:
    a, b = d.loc[d.critical, col], d.loc[~d.critical, col]
    p = mannwhitneyu(a, b)[1]
    rho, prho = spearmanr(d["ddG"], d[col])
    print(f"{label:32s} critical {a.mean():+.3f} vs other {b.mean():+.3f}  "
          f"MWU p={p:.4f}{' *' if p < 0.05 else '  '}   "
          f"| continuous rho(ddG)={rho:+.3f} p={prho:.4f}")""")

co(r"""fig, axes = plt.subplots(1, 3, figsize=(13, 4))

for ax, (col, label) in zip(axes[:2], [("full", "recovery (pMHC+TCR)"),
                                       ("tcr_benefit", "TCR benefit (full - pMHC only)")]):
    groups = [d.loc[~d.critical, col].values, d.loc[d.critical, col].values]
    bp = ax.boxplot(groups, tick_labels=[f"ddG < {CRIT}\n(n={len(groups[0])})",
                                         f"ddG >= {CRIT}\n(n={len(groups[1])})"],
                    widths=0.6, patch_artist=True, medianprops=dict(color="black"))
    for patch, c in zip(bp["boxes"], ["#B0B0B0", "crimson"]):
        patch.set_facecolor(c); patch.set_alpha(0.65)
    for i, g in enumerate(groups):
        ax.scatter(np.random.RandomState(0).normal(i + 1, 0.06, len(g)), g, s=12, color="black",
                   alpha=0.5, zorder=3)
    p = mannwhitneyu(groups[1], groups[0])[1]
    ax.axhline(0, lw=0.8, color="grey", ls=":")
    ax.set_ylabel(label); ax.set_title(f"{label}\nMann-Whitney p={p:.3f}", fontsize=9)

axes[2].scatter(d["ddG"], d["tcr_benefit"], s=22, color="#0072B2", alpha=0.75, edgecolor="black",
                linewidth=0.3)
axes[2].axhline(0, lw=0.8, color="grey", ls=":")
axes[2].axvline(CRIT, lw=0.8, color="crimson", ls="--")
rho, prho = spearmanr(d["ddG"], d["tcr_benefit"])
axes[2].set_xlabel(r"$\Delta\Delta G$ (kcal/mol), mutation weakens binding $\rightarrow$")
axes[2].set_ylabel("TCR benefit")
axes[2].set_title(f"per position, all complexes\nSpearman rho={rho:.2f} (p={prho:.3f})", fontsize=9)

fig.suptitle(f"Measured binding effect vs. recovery, pooled over {d['complex'].nunique()} "
             f"SKEMPI complexes", y=1.03)
fig.tight_layout()
out = FIG_DIR / "fig_panel5_pooled_skempi_ddg_vs_recovery.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"wrote {out}")

d.sort_values("ddG", ascending=False).to_csv(
    ROOT / "outputs/analysis/skempi_ddg_vs_recovery.csv", index=False)
print(d.sort_values("ddG", ascending=False).head(12).to_string(index=False))""")

md(r"""### 6b. How much weight will that p-value carry?

The pooled test clears $p<0.05$, but it clears it narrowly, and the measurements are not an even
sample: three crystals of the *same* epitope (the 3K peptide on H2-A, differing only in TCR) supply
roughly half of all positions. Before this is treated as a result, it is worth asking which parts of
the data it depends on. Each row below re-runs the identical test on a subset.""")

co(r"""FUSED = ["3C60_CD_AB", "4P23_CD_AB", "4P5T_CD_AB"]   # three crystals, one epitope (3K/H2-A)

def subset_test(sub, label):
    a = sub.loc[sub.critical, "tcr_benefit"]
    b = sub.loc[~sub.critical, "tcr_benefit"]
    if len(a) < 3 or len(b) < 3:
        print(f"  {label:46s} too few positions")
        return
    p = mannwhitneyu(a, b)[1]
    print(f"  {label:46s} crit {a.mean():+.3f} (n={len(a):2d}) vs other {b.mean():+.3f} "
          f"(n={len(b):2d})  p={p:.4f}{'  *' if p < 0.05 else ''}")

print("TCR benefit at critical vs non-critical positions, by subset")
subset_test(d, f"all {d.complex.nunique()} complexes ({len(d)} positions)")
subset_test(d[~d.complex.isin(FUSED)], "excluding the three 3K/H2-A crystals")
subset_test(d[d.complex.isin(FUSED)], "the three 3K/H2-A crystals alone")
subset_test(d[~d.complex.isin(FUSED + ["3QIB_ABP_CD"])], "class I only")
subset_test(d[d.pos > 1], "excluding P1 (initiator-Met artifact)")

ps = []
for c in d.complex.unique():
    s = d[d.complex != c]
    a, b = s.loc[s.critical, "tcr_benefit"], s.loc[~s.critical, "tcr_benefit"]
    ps.append((mannwhitneyu(a, b)[1], c))
ps.sort()
n_sig = sum(1 for p, _ in ps if p < 0.05)
print(f"\nleave-one-complex-out: p ranges {ps[0][0]:.4f} (dropping {ps[0][1]}) to "
      f"{ps[-1][0]:.4f} (dropping {ps[-1][1]})")
print(f"significant in {n_sig}/{len(ps)} folds")""")

md(r"""### What the pooled test says

The claim this project makes about the TCR is that it is read as part of the interface, so recovery
should depend on it *specifically where the interface depends on it*. Pooled over every SKEMPI complex
with designs, the benefit of having the TCR present is several times larger at positions where a
mutation measurably costs binding energy than at positions where it does not.

**The direction is consistent; the significance is not robust.** Critical positions show a larger TCR
benefit in every subset tested -- excluding the fused-construct crystals, restricting to class I,
dropping P1 -- but the pooled $p=0.047$ falls to $0.19$-$0.22$ in those subsets and survives only 6 of
17 leave-one-complex-out folds. The honest summary is a consistent direction on an underpowered
sample, not an established effect. It is a hypothesis this data supports and does not yet confirm.

Two further limits. The absolute recovery difference points the same way but does not reach
significance, and the continuous correlation with $\Delta\Delta G$ is weak, so what is being detected
is **how much the TCR context matters at a position**, not how well a position is recovered outright.
And a mutation that weakens binding marks a position the *interface* depends on, which includes MHC
anchors as well as TCR contacts; this split does not separate those two.

What would settle it is more peptide-position mutational data on distinct pMHC-TCR complexes. The
constraint is not the design side -- 50 structures are sampled at 10k designs each -- it is that only
17 of them have measured epitope mutations at all, and half the positions come from one epitope.""")

nb["cells"] = C
out_nb = Path("/home/ubuntu/if-mhc/notebooks/panel/05_skempi_validation.ipynb")
out_nb.parent.mkdir(exist_ok=True, parents=True)
nbf.write(nb, str(out_nb))
print(f"wrote {out_nb}")
