#!/usr/bin/env python3
"""Score the panel's unique peptides with MHCflurry (HLA-A*02:01). Run in the mhcflurry conda env.

Usage: /home/ubuntu/miniforge3/envs/mhcflurry/bin/python3 py/mhcflurry_score.py
"""
from pathlib import Path
import pandas as pd
from mhcflurry import Class1AffinityPredictor

ROOT = Path("/home/ubuntu/if-mhc")
OUT_DIR = ROOT / "outputs/analysis"
ALLELE = "HLA-A*02:01"

peptides = [l.strip() for l in open(OUT_DIR / "panel_unique_peptides_for_scoring.txt") if l.strip()]
print(f"scoring {len(peptides)} peptides with MHCflurry, allele={ALLELE}")

# MHCflurry only supports 8-15mers; all panel peptides are 8-11mers so this should pass everything,
# but filter defensively rather than crash on an edge case.
scoreable = [p for p in peptides if 8 <= len(p) <= 15]
skipped = len(peptides) - len(scoreable)
if skipped:
    print(f"skipping {skipped} peptides outside MHCflurry's 8-15mer range")

predictor = Class1AffinityPredictor.load()
df = predictor.predict_to_dataframe(peptides=scoreable, allele=ALLELE)
df = df.rename(columns={"prediction": "mhcflurry_ic50_nM", "prediction_percentile": "mhcflurry_percentile"})
df = df[["peptide", "mhcflurry_ic50_nM", "mhcflurry_percentile"]]

out = OUT_DIR / "panel_unique_peptides_mhcflurry.csv"
df.to_csv(out, index=False)
print(f"wrote {out}: {len(df)} rows")
