#!/usr/bin/env python3
"""Score the NY-ESO-1 index peptide SLLMWITQC on 2P5E with many decoding orders.

The library scores use n_orders=1, which is fine per-peptide because the Monte Carlo noise averages
out across thousands of peptides. The index score is different: it is subtracted from every peptide,
so its own noise would shift the entire delta distribution. Scored here with n_orders=200 so that
offset is stable.
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, "/home/ubuntu/if-mhc/py")
from batch_score_peptides_mpnn import score_peptides  # noqa: E402

INDEX = "SLLMWITQC"
PDB = "/home/ubuntu/if-mhc/inputs/pmhc_tcr_dataset/2P5E.pdb"
OUT = Path("/home/ubuntu/if-mhc/outputs/analysis/nyeso1_index_peptide_reference.npz")
N_ORDERS = 200

vals = {}
for w in ["vanilla", "nomhc"]:
    reps = score_peptides([INDEX] * 8, w, PDB, batch=8, n_orders=N_ORDERS)
    vals[w] = float(np.mean(reps))
    print(f"{w:10s} {INDEX} = {vals[w]:.4f}  (sd across 8 replicate draws {np.std(reps):.4f})")

np.savez(OUT, peptide=INDEX, n_orders=N_ORDERS,
         **{f"score_{k}": np.float64(v) for k, v in vals.items()})
print(f"wrote {OUT}")
