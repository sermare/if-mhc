#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/if-mhc
PY=/home/ubuntu/miniforge3/envs/esmcba/bin/python
NB=/home/ubuntu/miniforge3/bin/python
OUT=outputs/mpnn_tempsweep
mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG="$OUT/run.log"

# 1) wait for ALL GPU jobs (50K full-complex + MHC-only) to release the GPU
echo "[$(date)] tempsweep waiting for 50K + mhconly to complete..." | tee "$LOG"
while [ ! -f outputs/mpnn_50k/COMPLETE ] || [ ! -f outputs/mpnn_mhconly_v20/COMPLETE ]; do sleep 30; done
echo "[$(date)] all GPU jobs complete, starting temperature sweep" | tee -a "$LOG"

# 2) generate sequences across a range of temperatures (3000 per T)
$PY ProteinMPNN/protein_mpnn_run.py \
   --jsonl_path outputs/mpnn/parsed_chains.jsonl \
   --chain_id_jsonl outputs/mpnn/assigned_chains.jsonl \
   --out_folder "$OUT" \
   --num_seq_per_target 3000 \
   --batch_size 48 \
   --sampling_temp "0.1 0.15 0.2 0.3 0.5 0.7 1.0" \
   --seed 37 \
   --model_name v_48_020 >> "$LOG" 2>&1
echo "[$(date)] sweep generation done" | tee -a "$LOG"

# 3) parse to CSV
$NB - <<'PYEOF' 2>>"$LOG"
import csv, re
fa="outputs/mpnn_tempsweep/seqs/2P5E.fa"; rows=[]
ls=open(fa).read().splitlines()
for i in range(0,len(ls)-1,2):
    h,s=ls[i],ls[i+1].strip()
    if h.startswith(">2P5E,"): continue
    d=dict(re.findall(r'(\w+)=([-\d.]+)',h))
    rows.append({"peptide":s,"T":d.get("T"),"score":d.get("score"),
                 "global_score":d.get("global_score"),"seq_recovery":d.get("seq_recovery")})
with open("outputs/mpnn_tempsweep/peptides_by_temperature.csv","w",newline="") as o:
    w=csv.DictWriter(o,fieldnames=["peptide","T","score","global_score","seq_recovery"]);w.writeheader();w.writerows(rows)
print("sweep rows:",len(rows))
PYEOF

# 4) rebuild + execute the notebook on the full data
$NB build_notebook.py >> "$LOG" 2>&1
$NB -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1800 --ExecutePreprocessor.kernel_name=python3 \
    peptide_design_analysis.ipynb >> "$LOG" 2>&1
echo "[$(date)] DONE: tempsweep + notebook executed" | tee -a "$LOG"
touch "$OUT/COMPLETE"
