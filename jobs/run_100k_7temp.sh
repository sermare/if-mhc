#!/usr/bin/env bash
# 100K full-complex ProteinMPNN IF, split EQUALLY across 7 temperatures, no Met@P1.
# 14304 seqs/temp x 7 = 100,128 total.
set -uo pipefail
cd /home/ubuntu/if-mhc
PY=/home/ubuntu/miniforge3/envs/esmcba/bin/python
NB=/home/ubuntu/miniforge3/bin/python
OUT=outputs/mpnn_100k_7temp; mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG="$OUT/run.log"
trap 'echo "[$(date)] EXIT code $? (100k_7temp)" >> "$LOG"' EXIT

$NB make_omit_M_pos1.py outputs/mpnn/parsed_chains.jsonl "$OUT/omit.jsonl" C 1 M >>"$LOG" 2>&1
echo "[$(date)] START 100K full-complex | 7 temps x 14304 | no Met@P1 no Pro | batch16 seed41" | tee -a "$LOG"
$PY -u ProteinMPNN/protein_mpnn_run.py \
   --jsonl_path outputs/mpnn/parsed_chains.jsonl \
   --chain_id_jsonl outputs/mpnn/assigned_chains.jsonl \
   --omit_AA_jsonl "$OUT/omit.jsonl" \
   --omit_AAs "PX" \
   --out_folder "$OUT" \
   --num_seq_per_target 14304 --batch_size 16 \
   --sampling_temp "0.1 0.15 0.2 0.3 0.5 0.7 1.0" \
   --seed 41 --model_name v_48_020 >>"$LOG" 2>&1

echo "[$(date)] done, parsing per-temperature" | tee -a "$LOG"
$NB - "$OUT" >>"$LOG" 2>&1 <<'PYEOF'
import sys,re,csv
from collections import Counter
out=sys.argv[1]; fa=f"{out}/seqs/2P5E.fa"; rows=[]
ls=open(fa).read().splitlines()
for i in range(0,len(ls)-1,2):
    h,s=ls[i],ls[i+1].strip()
    if "sample=" not in h or len(s)!=9: continue
    d=dict(re.findall(r'(\w+)=([-\d.]+)',h))
    rows.append({"peptide":s,"score":d.get("score"),"global_score":d.get("global_score"),
                 "seq_recovery":d.get("seq_recovery"),"T":d.get("T"),"sample":d.get("sample")})
w=csv.DictWriter(open(f"{out}/peptides_100k.csv","w",newline=""),
    fieldnames=["peptide","score","global_score","seq_recovery","T","sample"]); w.writeheader(); w.writerows(rows)
open(f"{out}/peptides_unique.txt","w").write("\n".join(sorted(set(r['peptide'] for r in rows)))+"\n")
perT=Counter(r['T'] for r in rows)
m1=sum(1 for r in rows if r['peptide'][0]=='M')
print(f"total={len(rows)} unique={len(set(r['peptide'] for r in rows))} | Met@P1={m1} (should be 0)")
print("per-temperature:", dict(sorted(perT.items(), key=lambda x:float(x[0]))))
PYEOF
echo "[$(date)] DONE 100k_7temp" | tee -a "$LOG"
touch "$OUT/COMPLETE"
