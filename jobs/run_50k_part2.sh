#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/if-mhc
PY=/home/ubuntu/miniforge3/envs/esmcba/bin/python
NB=/home/ubuntu/miniforge3/bin/python
OUT=outputs/mpnn_50k_part2
mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG="$OUT/run.log"

echo "[$(date)] START part2: 31200 seqs @ batch 24 (smaller -> frees GPU memory)" | tee "$LOG"

# Generate the remaining sequences at a smaller batch (~11 GB instead of ~22 GB).
# Different seed so we extend diversity rather than repeat part1's sampling.
$PY ProteinMPNN/protein_mpnn_run.py \
   --jsonl_path outputs/mpnn/parsed_chains.jsonl \
   --chain_id_jsonl outputs/mpnn/assigned_chains.jsonl \
   --out_folder "$OUT" \
   --num_seq_per_target 31200 \
   --batch_size 24 \
   --sampling_temp "0.3" \
   --seed 38 \
   --model_name v_48_020 >> "$LOG" 2>&1

echo "[$(date)] part2 done, merging with part1" | tee -a "$LOG"

# Merge part1 (preserved) + part2 into the canonical 50K FASTA, then post-process.
$NB - <<'PYEOF' 2>>"$LOG"
import re, csv
part1="outputs/mpnn_50k/seqs/part1.fa"
part2="outputs/mpnn_50k_part2/seqs/2P5E.fa"
def read(path):
    native=None; pairs=[]
    ls=open(path).read().splitlines()
    for i in range(0,len(ls)-1,2):
        h,s=ls[i],ls[i+1].strip()
        if not h.startswith(">"): continue
        if h.startswith(">2P5E,"): native=(h,s); continue
        if len(s)==9: pairs.append((h,s))
    return native,pairs
n1,p1=read(part1); n2,p2=read(part2)
native=n1 or n2
allpairs=p1+p2
# write canonical merged FASTA
with open("outputs/mpnn_50k/seqs/2P5E.fa","w") as o:
    o.write(native[0]+"\n"+native[1]+"\n")
    for h,s in allpairs: o.write(h+"\n"+s+"\n")
# CSV + unique
rows=[]
for h,s in allpairs:
    d=dict(re.findall(r'(\w+)=([-\d.]+)',h))
    rows.append({"peptide":s,"score":d.get("score"),"global_score":d.get("global_score"),
                 "seq_recovery":d.get("seq_recovery"),"T":d.get("T"),"sample":d.get("sample")})
with open("outputs/mpnn_50k/peptides_50k.csv","w",newline="") as o:
    w=csv.DictWriter(o,fieldnames=["peptide","score","global_score","seq_recovery","T","sample"]);w.writeheader();w.writerows(rows)
uniq=sorted(set(s for _,s in allpairs))
open("outputs/mpnn_50k/peptides_unique.txt","w").write("\n".join(uniq)+"\n")
print(f"part1={len(p1)} part2={len(p2)} total={len(allpairs)} unique={len(uniq)}")
PYEOF

echo "[$(date)] DONE part2+merge. total=$(($(wc -l < outputs/mpnn_50k/seqs/2P5E.fa)/2 - 1))" | tee -a "$LOG"
touch outputs/mpnn_50k/COMPLETE
