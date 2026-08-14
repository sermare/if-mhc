#!/usr/bin/env python3
"""Score every unique designed peptide with MHCflurry 2.x (nb06 replication).

Runs in the `meta` conda env, which is the only one with mhcflurry installed.
Models live under MHCFLURRY_DATA_DIR on scratch.
"""
import os, sys
import pandas as pd

ROOT = "/global/scratch/users/sergiomar10/if-mhc"
os.environ.setdefault("MHCFLURRY_DATA_DIR",
                      "/global/scratch/users/sergiomar10/mhcflurry_models")

from mhcflurry import Class1PresentationPredictor

pairs = pd.read_csv(f"{ROOT}/outputs/skempi_if/mhcflurry_input_pairs.csv")
pred = Class1PresentationPredictor.load()
out = []
for allele, g in pairs.groupby("mhcflurry_allele"):
    peps = g.seq.astype(str).tolist()
    print(f"{allele}: {len(peps):,} peptides", flush=True)
    r = pred.predict(peptides=peps, alleles=[allele], verbose=0)
    r = r[["peptide", "affinity", "presentation_score"]].copy()
    r["mhcflurry_allele"] = allele
    out.append(r)

R = pd.concat(out, ignore_index=True).rename(columns={"peptide": "seq"})
R.to_csv(f"{ROOT}/outputs/skempi_if/mhcflurry_scores.csv", index=False)
print(f"\nscored {len(R):,} (peptide, allele) pairs")
print(R.groupby("mhcflurry_allele").affinity.describe()[["count", "25%", "50%", "75%"]].to_string())
