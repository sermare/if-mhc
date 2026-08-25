#!/usr/bin/env bash
# One-off reordering wrapper: nat_6AMU's partial progress got wiped by protein_mpnn_run.py's
# open(...,'w') truncation on every resume attempt (46895/50000 lost at the 2026-07-13 21:00
# auto-resume). Do nat_6AM5 (completely untouched, 0/50000) FIRST so it can't suffer the same
# fate, then fall through to the normal run script which will only have nat_6AMU left to redo.
# Holds the same runner.lock/runner.pid as run_nomhc_topcross_50k.sh so the cron supervisor
# recognizes this as "the runner" and does not double-launch.
set -uo pipefail
cd /home/ubuntu/if-mhc; ABS=/home/ubuntu/if-mhc
PY=/home/ubuntu/miniforge3/envs/esmcba/bin/python
OUT="$ABS/outputs/mpnn_nomhc_topcross_50k"
LOG="$OUT/run.log"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
gpu_wait(){ while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -lt "$1" ]; do sleep 20; done; }
exec 9>"$OUT/runner.lock"; flock -n 9 || { echo "another runner holds lock — exit"; exit 0; }
echo $$ >"$OUT/runner.pid"; trap 'rm -f "$OUT/runner.pid"' EXIT

count_seqs(){ [ -f "$OUT/seqs/$1.fa" ] && grep -c "^>" "$OUT/seqs/$1.fa" 2>/dev/null || echo 0; }

have=$(count_seqs nat_6AM5); have=${have:-0}
if [ "$have" -lt 50000 ]; then
  gpu_wait 4000
  echo "[$(date '+%F %T')] target=nat_6AM5 have=$have -> generating 50000 (T=0.1) [priority: untouched target first]" | tee -a "$LOG"
  $PY ProteinMPNN/protein_mpnn_run.py \
     --jsonl_path "$OUT/parsed_nat_6AM5.jsonl" \
     --chain_id_jsonl "$OUT/assigned_nat_6AM5.jsonl" \
     --out_folder "$OUT" \
     --num_seq_per_target 50000 \
     --batch_size 8 \
     --sampling_temp 0.1 \
     --seed 37 \
     --path_to_model_weights ProteinMPNN/nomhc_model_weights/ \
     --model_name proteinmpnn_nomhc >>"$LOG" 2>&1
  got=$(count_seqs nat_6AM5)
  echo "[$(date '+%F %T')] target=nat_6AM5 done: $got seqs" | tee -a "$LOG"
else
  echo "[$(date '+%F %T')] nat_6AM5 already at $have/50000 -- skip" | tee -a "$LOG"
fi

echo "[$(date '+%F %T')] nat_6AM5 pass complete -> handing off to normal runner for nat_6AMU (and any other stragglers)" | tee -a "$LOG"
rm -f "$OUT/runner.pid"
exec bash "$ABS/jobs/run_nomhc_topcross_50k.sh"
