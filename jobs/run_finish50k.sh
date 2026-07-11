#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/if-mhc
PY=/home/ubuntu/miniforge3/envs/esmcba/bin/python
NB=/home/ubuntu/miniforge3/bin/python
OUT=outputs/mpnn_50k_part3
mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG="$OUT/run.log"
trap 'echo "[$(date)] EXIT code $? (part3)" >> "$LOG"' EXIT

echo "[$(date)] START part3: finish 50K full-complex, 22656 seqs @ batch24 seed39" | tee "$LOG"
$PY -u ProteinMPNN/protein_mpnn_run.py \
   --jsonl_path outputs/mpnn/parsed_chains.jsonl \
   --chain_id_jsonl outputs/mpnn/assigned_chains.jsonl \
   --out_folder "$OUT" \
   --num_seq_per_target 22656 --batch_size 24 \
   --sampling_temp "0.3" --seed 39 --model_name v_48_020 >> "$LOG" 2>&1

echo "[$(date)] part3 done, merging part1+part2+part3 -> canonical 50K" | tee -a "$LOG"
$NB - <<'PYEOF' 2>>"$LOG"
import re, csv
frags=["outputs/mpnn_50k/seqs/part1.fa",
       "outputs/mpnn_50k_part2/seqs/2P5E.fa",
       "outputs/mpnn_50k_part3/seqs/2P5E.fa"]
native=None; pairs=[]
for f in frags:
    try: ls=open(f).read().splitlines()
    except FileNotFoundError: continue
    for i in range(0,len(ls)-1,2):
        h,s=ls[i],ls[i+1].strip()
        if not h.startswith(">"): continue
        if h.startswith(">2P5E,"): native=(h,s); continue
        if len(s)==9: pairs.append((h,s))
with open("outputs/mpnn_50k/seqs/2P5E.fa","w") as o:
    o.write(native[0]+"\n"+native[1]+"\n")
    for h,s in pairs: o.write(h+"\n"+s+"\n")
rows=[]
for h,s in pairs:
    d=dict(re.findall(r'(\w+)=([-\d.]+)',h))
    rows.append({"peptide":s,"score":d.get("score"),"global_score":d.get("global_score"),
                 "seq_recovery":d.get("seq_recovery"),"T":d.get("T"),"sample":d.get("sample")})
with open("outputs/mpnn_50k/peptides_50k.csv","w",newline="") as o:
    w=csv.DictWriter(o,fieldnames=["peptide","score","global_score","seq_recovery","T","sample"]);w.writeheader();w.writerows(rows)
uniq=sorted(set(s for _,s in pairs))
open("outputs/mpnn_50k/peptides_unique.txt","w").write("\n".join(uniq)+"\n")
print(f"MERGED total={len(pairs)} unique={len(uniq)}")
PYEOF
echo "[$(date)] DONE finish50k: total=$(($(wc -l < outputs/mpnn_50k/seqs/2P5E.fa)/2 - 1))" | tee -a "$LOG"
touch outputs/mpnn_50k/COMPLETE
