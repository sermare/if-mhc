#!/usr/bin/env bash
# Periodic eval during the grind: every ~30 min, MPNN the accumulated backbones and score the
# generated peptides vs DRG & GIG (and flag similar-motif / high-identity hits). Logs to TIMELINE.
set -uo pipefail
cd /home/ubuntu/if-mhc
ABS=/home/ubuntu/if-mhc
PY=/home/ubuntu/miniforge3/envs/esmcba/bin/python
OUT=outputs/grind
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2
gpu_wait(){ while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -lt "$1" ]; do sleep 15; done; }

cycle(){
  local nb=$(ls "$OUT"/pdb/*.pdb 2>/dev/null|wc -l)
  [ "$nb" -lt 4 ] && { echo "[$(date -u)] eval: only $nb backbones, skip" >> "$OUT/TIMELINE.log"; return; }
  rm -rf "$OUT/bb"; mkdir -p "$OUT/bb"; cp "$OUT"/pdb/*.pdb "$OUT/bb/" 2>/dev/null
  $PY ProteinMPNN/helper_scripts/parse_multiple_chains.py --input_path="$OUT/bb" --output_path="$OUT/parsed.jsonl" >/dev/null 2>&1
  $PY - "$OUT" >/dev/null 2>&1 <<'PYX'
import json,sys
out=sys.argv[1]; ci={}
for line in open(f"{out}/parsed.jsonl"):
    r=json.loads(line); ch=[k[-1] for k in r if k.startswith("seq_chain_")]
    pep=next((c for c in ch if 8<=len(r.get(f"seq_chain_{c}",""))<=12), ch[-1])
    ci[r["name"]]=[[pep],[c for c in ch if c!=pep]]
json.dump(ci,open(f"{out}/chain_id.jsonl","w"))
PYX
  gpu_wait 5000
  $PY ProteinMPNN/protein_mpnn_run.py --jsonl_path "$OUT/parsed.jsonl" --chain_id_jsonl "$OUT/chain_id.jsonl" \
     --out_folder "$OUT" --num_seq_per_target 50 --batch_size 8 --sampling_temp "0.1" --seed 37 --model_name v_48_020 >/dev/null 2>&1
  $PY grind_eval_report.py "$OUT" >> "$OUT/TIMELINE.log" 2>&1
}

while [ "$(date +%s)" -lt "$(cat $OUT/DEADLINE)" ] || [ ! -f "$OUT/worker0.DONE" ] || [ ! -f "$OUT/worker1.DONE" ]; do
  sleep 1800
  cycle
  [ -f "$OUT/worker0.DONE" ] && [ -f "$OUT/worker1.DONE" ] && break
done
echo "[$(date -u)] FINAL eval (higher sampling)" >> "$OUT/TIMELINE.log"
cycle
touch "$OUT/EVAL_DONE"
