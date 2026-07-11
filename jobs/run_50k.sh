#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/if-mhc
PY=/home/ubuntu/miniforge3/envs/esmcba/bin/python
OUT=outputs/mpnn_50k
mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date)] START ProteinMPNN 50K | design chain C (peptide) | context=full complex A+B+D+E | batch=48" | tee "$OUT/run.log"

$PY ProteinMPNN/protein_mpnn_run.py \
   --jsonl_path outputs/mpnn/parsed_chains.jsonl \
   --chain_id_jsonl outputs/mpnn/assigned_chains.jsonl \
   --out_folder "$OUT" \
   --num_seq_per_target 50000 \
   --batch_size 48 \
   --sampling_temp "0.3" \
   --seed 37 \
   --model_name v_48_020 >> "$OUT/run.log" 2>&1

echo "[$(date)] MPNN done, post-processing" | tee -a "$OUT/run.log"

$PY - <<'PYEOF' 2>>"$OUT/run.log"
import csv, re
fa = "outputs/mpnn_50k/seqs/2P5E.fa"
rows=[]; native=None
with open(fa) as f:
    lines=[l.rstrip("\n") for l in f]
for i in range(0,len(lines)-1,2):
    h=lines[i]; s=lines[i+1]
    if h.startswith(">2P5E,"):
        native=s; continue
    d=dict(re.findall(r'(\w+)=([-\d.]+)', h))
    rows.append({"peptide":s,"score":d.get("score"),
                 "global_score":d.get("global_score"),
                 "seq_recovery":d.get("seq_recovery"),
                 "T":d.get("T"),"sample":d.get("sample")})
with open("outputs/mpnn_50k/peptides_50k.csv","w",newline="") as o:
    w=csv.DictWriter(o,fieldnames=["peptide","score","global_score","seq_recovery","T","sample"])
    w.writeheader(); w.writerows(rows)
uniq=sorted(set(r["peptide"] for r in rows))
with open("outputs/mpnn_50k/peptides_unique.txt","w") as o:
    o.write("\n".join(uniq)+"\n")
print(f"native={native}")
print(f"total_sequences={len(rows)}")
print(f"unique_sequences={len(uniq)}")
PYEOF

echo "[$(date)] DONE" | tee -a "$OUT/run.log"
touch "$OUT/COMPLETE"
