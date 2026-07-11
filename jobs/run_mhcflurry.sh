#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/if-mhc
PY=/home/ubuntu/miniforge3/envs/mhcflurry/bin/python
OUT=outputs/mhcflurry
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=""          # CPU only -> never touches the GPU
LOG="$OUT/run.log"

echo "[$(date)] MHCflurry HLA-A*02:01 binding prediction on current designs" | tee "$LOG"

$PY - <<'PYEOF' 2>>"$LOG" | tee -a "$LOG"
import glob, re, time
import pandas as pd, numpy as np
from mhcflurry import Class1PresentationPredictor

NATIVE = "SLLMWITQC"
# gather all designed peptides available so far (part1 + part2 in progress)
files = ["outputs/mpnn_50k/seqs/part1.fa",
         "outputs/mpnn_50k/seqs/2P5E.fa",
         "outputs/mpnn_50k_part2/seqs/2P5E.fa"]
peps=set()
for f in files:
    try: lines=open(f).read().splitlines()
    except FileNotFoundError: continue
    for i in range(0,len(lines)-1,2):
        h,s=lines[i],lines[i+1].strip()
        if h.startswith(">2P5E,"): continue
        if len(s)==9 and set(s)<=set("ACDEFGHIKLMNPQRSTVWY"): peps.add(s)
peps=sorted(peps);
print(f"unique designed peptides to score: {len(peps):,}")

pred=Class1PresentationPredictor.load()
t0=time.time()
res=pred.predict(peptides=peps+[NATIVE], alleles=["HLA-A*02:01"], verbose=0)
print(f"predicted in {time.time()-t0:.1f}s")
res=res.rename(columns={"peptide":"peptide"})
res["is_native"]=res.peptide==NATIVE
res.to_csv("outputs/mhcflurry/binding_predictions.csv", index=False)

nat=res[res.is_native].iloc[0]
des=res[~res.is_native]
aff=des["affinity"]
print("\n=== HLA-A*02:01 binding summary ===")
print(f"native {NATIVE}: affinity={nat.affinity:.1f} nM, presentation_score={nat.presentation_score:.3f}")
print(f"designs: median affinity={aff.median():.1f} nM | best={aff.min():.2f} nM")
print(f"strong binders (<50 nM):  {(aff<50).mean()*100:.1f}%  ({(aff<50).sum():,})")
print(f"binders     (<500 nM):    {(aff<500).mean()*100:.1f}%  ({(aff<500).sum():,})")
print(f"better than native:       {(aff<nat.affinity).mean()*100:.1f}%")
print("\nTop 10 designs by predicted affinity:")
print(des.sort_values("affinity")[["peptide","affinity","presentation_score"]].head(10).to_string(index=False))
PYEOF

echo "[$(date)] DONE -> outputs/mhcflurry/binding_predictions.csv" | tee -a "$LOG"
touch "$OUT/COMPLETE"
