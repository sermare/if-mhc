#!/usr/bin/env python3
"""Per-peptide Hamming-distance-to-nearest-design histogram, gridded by NGS round-count threshold
and by which model produced the nearest match. Enables the analysis the user pasted (previously
gated behind `if False == True:` as a disabled prior draft), run separately for EACH native/index
peptide (3HG1/MEL5 and 2P5E/NY-ESO-1 -- both structures directly), and across ALL 4 models
(vanilla, noMHC, ESM-IF1, LigandMPNN at T=0.1) rather than just the 2 the original pasted snippet's
file happened to cover.
"""
import sys
sys.path.append("/home/ubuntu/pmhc/modeling/ONG229/py")
import ong229_ranking_lib as lib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = "/home/ubuntu/if-mhc/"

# same T=0.1 sources validated in py/build_fig_if458_with_peptide.py
DESIGN_PATHS = {
    "3HG1": {
        "vanilla": f"{ROOT}outputs/mpnn_3hg1_T03_50k/run_vanilla/seqs/3HG1.fa",
        "noMHC": f"{ROOT}outputs/mpnn_3hg1_T03_50k/run_nomhc/seqs/3HG1.fa",
        "ESM-IF1": f"{ROOT}outputs/esmif_3hg1_T03_20k/seqs/3HG1.fa",
        "LigandMPNN": f"{ROOT}outputs/ligandmpnn_3hg1_T03_20k/seqs/3HG1.fa",
    },
    "2P5E": {
        "vanilla": f"{ROOT}outputs/mpnn_2p5e_T03_20k/seqs/vanilla_2P5E.fa",
        "noMHC": f"{ROOT}outputs/mpnn_2p5e_T03_20k/seqs/nomhc_2P5E.fa",
        "ESM-IF1": f"{ROOT}outputs/esmif_2p5e_T03_20k/seqs/2P5E.fa",
        "LigandMPNN": f"{ROOT}outputs/ligandmpnn_2p5e_T03_20k/seqs/2P5E.fa",
    },
}


def encode(seqs, length):
    """Vectorized: peptides -> (N, length) uint8 array of ordinal codes, one row per seq."""
    return np.array([[ord(c) for c in s] for s in seqs], dtype=np.uint8)


def nearest_hamming_batched(real_peps, design_peps, length, batch=5000):
    """For each real peptide, the nearest design peptide by Hamming distance -- fully vectorized
    with numpy broadcasting instead of a pure-python double loop (which does not scale once the
    design pool reaches thousands of unique peptides, as T=0.3 does)."""
    design_arr = encode(design_peps, length)  # (M, L)
    best_dist = np.full(len(real_peps), 255, dtype=np.int16)
    best_idx = np.zeros(len(real_peps), dtype=np.int64)
    for start in range(0, len(real_peps), batch):
        chunk = real_peps[start:start + batch]
        chunk_arr = encode(chunk, length)  # (b, L)
        # (b, 1, L) vs (1, M, L) -> (b, M, L) -> sum over L -> (b, M)
        dist = (chunk_arr[:, None, :] != design_arr[None, :, :]).sum(axis=2)
        idx = dist.argmin(axis=1)
        best_dist[start:start + len(chunk)] = dist[np.arange(len(chunk)), idx]
        best_idx[start:start + len(chunk)] = idx
    return best_dist, [design_peps[i] for i in best_idx]


def peptide_from_ligandmpnn_line(line):
    return line.strip().split(":")[2]


def load_designs_all_models(struct):
    rows = []
    for model, path in DESIGN_PATHS[struct].items():
        with open(path) as f:
            lines = f.read().splitlines()
        for i in range(0, len(lines) - 1, 2):
            if not lines[i].startswith(">"):
                continue
            seq = peptide_from_ligandmpnn_line(lines[i + 1]) if model == "LigandMPNN" else lines[i + 1].strip()
            rows.append({"peptide": seq, "model": model})
    return pd.DataFrame(rows)


def run_for_structure(struct, native, if_peptides, ngs_df, round_cols, out_name):
    length = len(native)
    mart_peps = [p for p in set(ngs_df['Peptide']) if len(p) == length]
    if_peps = [p for p in set(if_peptides['peptide']) if len(p) == length]
    print(f"[{struct}] real unique peptides: {len(mart_peps):,}  design unique peptides: {len(if_peps):,}")

    dists, matches = nearest_hamming_batched(mart_peps, if_peps, length)
    matches_df = pd.DataFrame({'mart_peptide': mart_peps, 'closest_if_peptide': matches,
                                'hamming_dist': dists})
    matches_df.to_csv(ROOT + f'hammingdist_matches_T03_{struct}.csv')

    models = sorted(if_peptides['model'].unique())
    model_peps = {m: set(if_peptides.loc[if_peptides['model'] == m, 'peptide']) for m in models}

    def classify(pep):
        hits = [m for m in models if pep in model_peps[m]]
        if len(hits) == 0:
            return 'neither'
        if len(hits) == 1:
            return hits[0]
        return 'multiple'

    matches_df['source'] = matches_df['closest_if_peptide'].apply(classify)
    matches_df = matches_df.merge(
        ngs_df[['Peptide'] + round_cols],
        left_on='mart_peptide', right_on='Peptide', how='left'
    ).drop(columns='Peptide')

    source_order = [m for m in models if m in set(matches_df['source'])] + \
        [s for s in ['multiple', 'neither'] if s in set(matches_df['source'])]
    colors = (['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2'] * 2)[:len(round_cols)]
    bins = range(0, matches_df['hamming_dist'].max() + 2)

    fig, axes = plt.subplots(len(round_cols), len(source_order),
                              figsize=(5 * len(source_order), 2.5 * len(round_cols)),
                              sharex=True, sharey='row', squeeze=False)

    for i, (col, color) in enumerate(zip(round_cols, colors)):
        for j, source in enumerate(source_order):
            ax = axes[i, j]
            subset = matches_df[(matches_df[col] > 1) & (matches_df['source'] == source)]
            ax.hist(subset['hamming_dist'], bins=bins, color=color, edgecolor='black', align='left')
            if len(subset):
                mean_d = subset['hamming_dist'].mean()
                ax.axvline(mean_d, color='black', linestyle='--', linewidth=1.5)
                ax.text(mean_d, ax.get_ylim()[1] * 0.95, f' mean={mean_d:.2f}', fontsize=7.5,
                        ha='left', va='top', color='black')
            ax.set_title(f'{col} > 1, source={source}  (n={len(subset)})', fontsize=9, loc='left')
            if j == 0:
                ax.set_ylabel('Count')
            if i == len(round_cols) - 1:
                ax.set_xlabel('Hamming distance')

    fig.suptitle(f"{struct} -- native/index peptide: {native} (all 4 models, T=0.3)", y=1.01, fontsize=13)
    fig.tight_layout()
    out = ROOT + f'figures/fig_if11_hamming_hist_by_R_and_source/{out_name}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {out}")
    print(matches_df['source'].value_counts())
    return matches_df


if __name__ == "__main__":
    targets = sys.argv[1:] or ["3hg1", "2p5e"]

    if "3hg1" in targets:
        tab_data = lib.load_tab_data()
        if_3hg1 = load_designs_all_models("3HG1")
        ngs_3hg1 = tab_data['MART1_10mer__CAB60174_G01']
        run_for_structure("3HG1_MEL5", "ELAGIGILTV", if_3hg1, ngs_3hg1,
                          ["R0", "R1", "R2", "R3"], "hamming_hist_by_R_and_source_3HG1_T03")

    if "2p5e" in targets:
        if_2p5e = load_designs_all_models("2P5E")
        ngs_2p5e = pd.read_csv("/home/ubuntu/pmhc/modeling/work/full_5round/ONG229_1G4c58c61_peptide_counts.csv")
        run_for_structure("2P5E_NYESO1", "SLLMWITQC", if_2p5e, ngs_2p5e,
                          ["R0", "R1", "R2", "R3", "R4"], "hamming_hist_by_R_and_source_2P5E_T03")
