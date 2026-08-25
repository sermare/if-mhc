#!/usr/bin/env python3
"""Reuses the learned-embedding-space methodology from
pmhc/modeling/ONG229/py/analyze_nyeso1_cnn_embedding_projections.py (128-dim penultimate layer of
the trained CNNRanker for the 1G4c58c61 TCR, best_seed=17) to embed THREE separate populations into
the SAME fitted PCA / UMAP space (fit once on a 100k real-library background sample, matching that
script's convention), producing one 3-column figure per projection method:

  col 1: our own inverse-folding designs (2P5E: vanilla/noMHC/ESM-IF1/LigandMPNN, T=0.1+T=0.3 pooled,
         unique peptides), colored by which model generated them
  col 2: the real KD-tested peptides only, colored by KD binding strength (log10 nM; confirmed
         non-binders marked separately)
  col 3: real NGS peptides sampled from the first round (R0) and the last/terminal round (R4),
         colored by round

Must run under `protenix` (has torch+umap+sklearn+GPU).
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
from sklearn.decomposition import PCA
import umap

sys.path.insert(0, "/home/ubuntu/pmhc/modeling/ONG229/py")
import nyeso1_ranking_lib as lib

ROOT = Path("/home/ubuntu/if-mhc")
FIG_DIR = ROOT / "figures/fig_if13_nyeso1_cnn_embedding"
FIG_DIR.mkdir(exist_ok=True, parents=True)
RUNS_DIR = Path("/home/ubuntu/pmhc/modeling/ONG229/lm_runs")
WORK_DIR = Path("/home/ubuntu/pmhc/modeling/work")

DEV = torch.device("cuda")
AA_IDX = {a: i for i, a in enumerate(lib.AA_LIST)}
NATIVE = "SLLMWITQC"
TCR = "1G4c58c61"
BEST_SEED = 17
N_BACKGROUND = 100_000
N_ROUND_SAMPLE = 5000
SEED = 0

MODELS = ["vanilla", "noMHC", "ESM-IF1", "LigandMPNN"]
MODEL_COLOR = {"vanilla": "#4C72B0", "noMHC": "#DD8452", "ESM-IF1": "#55A868", "LigandMPNN": "#C44E52"}

OUR_DESIGN_PATHS = {
    "vanilla": [ROOT / "outputs/mpnn_2p5e_T01_20k/seqs/vanilla_2P5E.fa",
                ROOT / "outputs/mpnn_2p5e_T03_20k/seqs/vanilla_2P5E.fa"],
    "noMHC": [ROOT / "outputs/mpnn_2p5e_T01_20k/seqs/nomhc_2P5E.fa",
              ROOT / "outputs/mpnn_2p5e_T03_20k/seqs/nomhc_2P5E.fa"],
    "ESM-IF1": [ROOT / "outputs/esmif_2p5e_pilot/seqs/2P5E.fa",
                ROOT / "outputs/esmif_2p5e_T03_20k/seqs/2P5E.fa"],
    "LigandMPNN": [ROOT / "outputs/ligandmpnn_2p5e_pilot/seqs/2P5E.fa",
                   ROOT / "outputs/ligandmpnn_2p5e_T03_20k/seqs/2P5E.fa"],
}


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
    return np.concatenate(out, axis=0) if out else np.zeros((0, 128))


def peptide_from_ligandmpnn_line(line):
    return line.strip().split(":")[2]


def load_fasta_peptides(path, is_ligandmpnn):
    seqs = []
    with open(path) as f:
        lines = f.read().splitlines()
    for i in range(0, len(lines) - 1, 2):
        if not lines[i].startswith(">"):
            continue
        seqs.append(peptide_from_ligandmpnn_line(lines[i + 1]) if is_ligandmpnn else lines[i + 1].strip())
    return seqs


def parse_kd(val):
    if pd.isna(val) or val in ("N.B.", "N.D."):
        return (False if val == "N.B." else None), np.nan
    try:
        return True, float(val)
    except ValueError:
        return None, np.nan


def main():
    print("Loading background NGS library + fitting projections...", flush=True)
    tab_data = lib.load_tab_data()
    df = tab_data[f"NYESO1__{TCR}"]
    rng = np.random.RandomState(SEED)
    n_sample = min(N_BACKGROUND, len(df))
    bg_df = df.iloc[rng.choice(len(df), size=n_sample, replace=False)].copy()
    bg_peptides = bg_df["Peptide"].values

    model = CNNRanker(9).to(DEV)
    ckpt = RUNS_DIR / f"nyeso1_reduced_sweep_countweighted_seed{BEST_SEED}_{TCR}" / "model_best.pt"
    load_state_dict_remapped(model, ckpt)
    bg_emb = get_embeddings(model, bg_peptides)
    print(f"background: n={len(bg_peptides):,}, embedding shape={bg_emb.shape}", flush=True)

    # --- column 1: our own designs, deduplicated per model, pooled T=0.1+T=0.3 ---
    design_peps, design_models = [], []
    for m in MODELS:
        seen = set()
        for path in OUR_DESIGN_PATHS[m]:
            is_lig = m == "LigandMPNN"
            for p in load_fasta_peptides(path, is_lig):
                if len(p) == 9 and p not in seen:
                    seen.add(p)
        design_peps.extend(seen)
        design_models.extend([m] * len(seen))
        print(f"  our designs [{m}]: {len(seen):,} unique 9-mers (T0.1+T0.3 pooled)", flush=True)
    design_emb = get_embeddings(model, design_peps)

    # --- column 2: real KD-tested peptides ---
    kinetics_df = pd.read_csv(WORK_DIR / "Birnbaum collab compiled kinetics.csv", low_memory=False)
    kin_sub = kinetics_df[["Peptide", "1G4 c58c61"]].rename(columns={"1G4 c58c61": "KD_raw"})
    kin_sub = kin_sub[kin_sub["Peptide"].astype(str).str.len() == 9]
    parsed = kin_sub["KD_raw"].apply(parse_kd)
    kin_sub["is_binder"] = parsed.apply(lambda t: t[0])
    kin_sub["kd_value"] = parsed.apply(lambda t: t[1])
    kin_sub = kin_sub[kin_sub["KD_raw"].notna()]
    kd_peps = kin_sub["Peptide"].tolist()
    kd_emb = get_embeddings(model, kd_peps)
    print(f"  KD-tested peptides: {len(kd_peps)} ({kin_sub['is_binder'].sum()} confirmed binders, "
          f"{(kin_sub['is_binder'] == False).sum()} confirmed N.B.)", flush=True)

    # --- column 3: R0-only vs terminal-round(R4)-only real peptides ---
    round_cols = lib.ROUND_COLS  # ["R0", "R3", "R4"]
    counts = df[round_cols].values.astype(np.float64)
    r0_only = df[(counts[:, 0] > 0) & (counts[:, 1] == 0) & (counts[:, 2] == 0)]
    r4_only = df[(counts[:, 2] > 0)]
    n_r0 = min(N_ROUND_SAMPLE, len(r0_only))
    n_r4 = min(N_ROUND_SAMPLE, len(r4_only))
    r0_sample = r0_only.iloc[rng.choice(len(r0_only), size=n_r0, replace=False)]["Peptide"].tolist()
    r4_sample = r4_only.iloc[rng.choice(len(r4_only), size=n_r4, replace=False)]["Peptide"].tolist()
    round_peps = r0_sample + r4_sample
    round_labels = ["R0 (first)"] * len(r0_sample) + ["R4 (terminal)"] * len(r4_sample)
    round_emb = get_embeddings(model, round_peps)
    print(f"  round samples: R0-only={len(r0_sample):,}, R4-terminal={len(r4_sample):,}", flush=True)

    def make_figure(fit_fn, transform_fn, method_name, out_name):
        Z_bg = fit_fn(bg_emb)
        Z_design = transform_fn(design_emb)
        Z_kd = transform_fn(kd_emb)
        Z_round = transform_fn(round_emb)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

        ax = axes[0]
        ax.scatter(Z_bg[:, 0], Z_bg[:, 1], s=3, alpha=0.08, color="lightgray", linewidths=0)
        for m in MODELS:
            mask = np.array(design_models) == m
            ax.scatter(Z_design[mask, 0], Z_design[mask, 1], s=8, alpha=0.6,
                       color=MODEL_COLOR[m], label=f"{m} (n={mask.sum():,})")
        ax.set_title("our designs, colored by model")
        ax.legend(fontsize=7, markerscale=1.5, loc="best")
        ax.set_xlabel(f"{method_name}1"); ax.set_ylabel(f"{method_name}2")

        ax = axes[1]
        ax.scatter(Z_bg[:, 0], Z_bg[:, 1], s=3, alpha=0.08, color="lightgray", linewidths=0)
        is_binder = kin_sub["is_binder"].values
        kd_vals = kin_sub["kd_value"].values
        binder_mask = is_binder == True
        nb_mask = is_binder == False
        sc = ax.scatter(Z_kd[binder_mask, 0], Z_kd[binder_mask, 1], c=np.log10(kd_vals[binder_mask]),
                         cmap="plasma_r", s=90, edgecolors="black", linewidths=0.6, zorder=5,
                         label="confirmed binder")
        ax.scatter(Z_kd[nb_mask, 0], Z_kd[nb_mask, 1], marker="X", s=70, color="dimgray",
                   edgecolors="black", linewidths=0.5, zorder=4, label="confirmed N.B.")
        fig.colorbar(sc, ax=ax, fraction=0.046, label="log10(KD, nM)  [lower = stronger]")
        ax.set_title("KD-tested peptides, colored by binding strength")
        ax.legend(fontsize=7, loc="best")
        ax.set_xlabel(f"{method_name}1")

        ax = axes[2]
        ax.scatter(Z_bg[:, 0], Z_bg[:, 1], s=3, alpha=0.08, color="lightgray", linewidths=0)
        round_labels_arr = np.array(round_labels)
        for label, color in [("R0 (first)", "#bdbdbd"), ("R4 (terminal)", "#08306b")]:
            mask = round_labels_arr == label
            ax.scatter(Z_round[mask, 0], Z_round[mask, 1], s=6, alpha=0.4, color=color,
                       label=f"{label} (n={mask.sum():,})")
        ax.set_title("real NGS peptides, first vs. last round")
        ax.legend(fontsize=8, loc="best")
        ax.set_xlabel(f"{method_name}1")

        fig.suptitle(f"NY-ESO-1 / 1G4c58c61 CNN embedding space ({method_name}) -- "
                     f"native peptide: {NATIVE}", y=1.03, fontsize=13)
        fig.tight_layout()
        out = FIG_DIR / out_name
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}", flush=True)

    print("Running PCA...", flush=True)
    pca = PCA(n_components=2, random_state=0)
    make_figure(pca.fit_transform, pca.transform, "PC", "fig_if13_nyeso1_cnn_embedding_pca.png")
    print(f"  PCA var explained: {pca.explained_variance_ratio_}", flush=True)

    print("Running UMAP (slower)...", flush=True)
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=0, verbose=False)
    make_figure(reducer.fit_transform, reducer.transform, "UMAP", "fig_if13_nyeso1_cnn_embedding_umap.png")

    print("DONE")


if __name__ == "__main__":
    main()
