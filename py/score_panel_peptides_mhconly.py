#!/usr/bin/env python3
"""Same as score_panel_peptides.py but for the mhconly (pMHC only, no TCR) condition -- separate
output files, does not touch the full-condition outputs. Run BEFORE mhcflurry_score_mhconly.py and
esmcba_score_mhconly.py.

Usage: /home/ubuntu/miniforge3/bin/python3 py/score_panel_peptides_mhconly.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/home/ubuntu/if-mhc")
OUT_DIR = ROOT / "outputs/analysis"
# 4MJI is HLA-B*51:01, not HLA-A*02:01 -- excluded since every score below (MHCflurry, ESMCBA) is
# computed for a single allele (A*02:01) and would be scientifically meaningless for a B*51:01 peptide.
STRUCTS = ["2P5W", "1QSF", "1QRN", "2BNR", "2GJ6", "2F53", "2F54", "3QDG", "3QEQ", "3QFJ", "3GSN",
           "1OGA", "3UTS", "5C0A", "5C0B", "5HHO", "5EU6", "2VLR", "5NME",
           "1BD2", "1LP9", "1QSE", "2BNQ", "2J8U", "2JCC", "2PYE", "2UWE", "3D3V", "3H9S",
           "3PWP", "3QDJ", "4FTV", "4JFD", "4JFE", "4JFF", "4L3E", "4MNQ", "5E9D",
           "6AM5", "6AMU"]
MODELS = ["vanilla", "noMHC", "ESM-IF1", "LigandMPNN"]
# Every (structure, arm, model) cell is truncated to the same number of designs. The raw
# counts differ slightly by generator (9,984 / 10,000 / 10,016), and unique-design counts
# scale with sample size, so comparisons across cells need a common N. 9,984 is the largest
# N every cell can supply; designs are i.i.d. draws, so a prefix is a valid subsample.
N_DESIGNS = 9984

master = pd.read_csv(ROOT / "outputs/analysis/panel_dataset_master_table.csv").set_index("pdb")

def peptide_from_ligandmpnn_line(line):
    return line.strip().split(":")[2]

def load_designs(pdb, length):
    rows = []
    for weights, fname in [("vanilla", f"vanilla_{pdb}.fa"), ("noMHC", f"nomhc_{pdb}.fa")]:
        path = ROOT / f"outputs/panel/{pdb}/mhconly/mpnn/seqs/{fname}"
        lines = path.read_text().splitlines() if path.exists() else []
        lines = lines[:2 * N_DESIGNS + 2]   # uniform sample size across every cell
        for i in range(2, len(lines) - 1, 2):
            if lines[i].startswith(">"):
                header, seq = lines[i], lines[i + 1].strip()
                score = np.nan
                for tok in header.split(","):
                    tok = tok.strip()
                    if tok.startswith("score="):
                        score = float(tok.split("=")[1])
                if len(seq) == length:
                    rows.append({"peptide": seq, "model": weights, "score": score})

    path = ROOT / f"outputs/panel/{pdb}/mhconly/esmif/seqs/{pdb}.fa"
    lines = path.read_text().splitlines() if path.exists() else []
    lines = lines[:2 * N_DESIGNS + 0]   # uniform sample size across every cell
    for i in range(0, len(lines) - 1, 2):
        if lines[i].startswith(">"):
            seq = lines[i + 1].strip()
            if len(seq) == length:
                rows.append({"peptide": seq, "model": "ESM-IF1", "score": np.nan})

    path = ROOT / f"outputs/panel/{pdb}/mhconly/ligandmpnn/seqs/{pdb}.fa"
    lines = path.read_text().splitlines() if path.exists() else []
    lines = lines[:2 * N_DESIGNS + 2]   # uniform sample size across every cell
    for i in range(2, len(lines) - 1, 2):
        if lines[i].startswith(">"):
            header, seq = lines[i], peptide_from_ligandmpnn_line(lines[i + 1])
            score = np.nan
            for tok in header.split(","):
                tok = tok.strip()
                if tok.startswith("overall_confidence="):
                    score = float(tok.split("=")[1])
            if len(seq) == length:
                rows.append({"peptide": seq, "model": "LigandMPNN", "score": score})
    return pd.DataFrame(rows)

records = []
for pdb in STRUCTS:
    native = master.loc[pdb, "peptide"]
    length = int(master.loc[pdb, "pep_len"])
    resolution = float(master.loc[pdb, "resolution_A"])
    df = load_designs(pdb, length)
    df = df.drop_duplicates(subset=["model", "peptide"]).copy()
    df["pdb"] = pdb
    df["native"] = native
    df["resolution_A"] = resolution
    df["hamming_to_native"] = df["peptide"].apply(
        lambda p: sum(a != b for a, b in zip(p, native)))
    df["p2_recovered"] = df["peptide"].apply(lambda p: p[1] == native[1])
    df["pomega_recovered"] = df["peptide"].apply(lambda p: p[-1] == native[-1])
    df["n_anchors_recovered"] = df["p2_recovered"].astype(int) + df["pomega_recovered"].astype(int)
    records.append(df)

full_df = pd.concat(records, ignore_index=True)
full_df = full_df[["pdb", "model", "peptide", "score", "native", "resolution_A",
                   "hamming_to_native", "p2_recovered", "pomega_recovered", "n_anchors_recovered"]]

out = OUT_DIR / "panel_unique_peptides_metadata_mhconly.csv"
full_df.to_csv(out, index=False)
print(f"wrote {out}: {len(full_df)} (pdb, model, peptide) rows, "
      f"{full_df['peptide'].nunique()} distinct peptide strings")

unique_peptides_path = OUT_DIR / "panel_unique_peptides_for_scoring_mhconly.txt"
with open(unique_peptides_path, "w") as f:
    for pep in sorted(full_df["peptide"].unique()):
        f.write(pep + "\n")
print(f"wrote {unique_peptides_path}: {full_df['peptide'].nunique()} distinct peptides to score")
