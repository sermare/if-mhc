#!/usr/bin/env bash
# Usage: run_mhconly.sh <model_name> <seed> <outdir>
set -euo pipefail
cd /home/ubuntu/if-mhc
MODEL="${1:-v_48_020}"; SEED="${2:-37}"; OUT="${3:-outputs/mpnn_mhconly_v20}"; BATCH="${4:-32}"
PY=/home/ubuntu/miniforge3/envs/esmcba/bin/python
mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG="$OUT/run.log"
trap 'echo "[$(date)] EXIT code $? (mhconly $MODEL)" >> "$LOG"' EXIT

echo "[$(date)] START MHC-only IF: 50048 seqs @ batch$BATCH | model=$MODEL seed=$SEED ctx=A+B" | tee "$LOG"
$PY -u ProteinMPNN/protein_mpnn_run.py \
   --jsonl_path outputs/mpnn/parsed_mhconly.jsonl \
   --chain_id_jsonl outputs/mpnn/assigned_mhconly.jsonl \
   --out_folder "$OUT" \
   --num_seq_per_target 50048 --batch_size "$BATCH" \
   --sampling_temp "0.3" --seed "$SEED" --model_name "$MODEL" >> "$LOG" 2>&1

echo "[$(date)] mhconly done, parsing" | tee -a "$LOG"
/home/ubuntu/miniforge3/bin/python - "$OUT" <<'PYEOF' 2>>"$LOG"
import re, csv, sys
out=sys.argv[1]; fa=f"{out}/seqs/2P5E_ABC.fa"
ls=open(fa).read().splitlines(); rows=[]
for i in range(0,len(ls)-1,2):
    h,s=ls[i],ls[i+1].strip()
    if h.startswith(">2P5E_ABC,") or not h.startswith(">"): continue
    if len(s)!=9: continue
    d=dict(re.findall(r'(\w+)=([-\d.]+)',h))
    rows.append({"peptide":s,"score":d.get("score"),"global_score":d.get("global_score"),
                 "seq_recovery":d.get("seq_recovery"),"T":d.get("T"),"sample":d.get("sample")})
with open(f"{out}/peptides.csv","w",newline="") as o:
    w=csv.DictWriter(o,fieldnames=["peptide","score","global_score","seq_recovery","T","sample"]);w.writeheader();w.writerows(rows)
open(f"{out}/peptides_unique.txt","w").write("\n".join(sorted(set(r['peptide'] for r in rows)))+"\n")
print(f"mhconly total={len(rows)} unique={len(set(r['peptide'] for r in rows))}")
PYEOF
echo "[$(date)] DONE mhconly $MODEL" | tee -a "$LOG"
touch "$OUT/COMPLETE"
