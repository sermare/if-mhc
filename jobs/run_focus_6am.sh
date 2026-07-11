#!/usr/bin/env bash
set -uo pipefail
cd /home/ubuntu/if-mhc
PY=/home/ubuntu/miniforge3/envs/esmcba/bin/python
OUT=outputs/focus_6am
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2
LOG="$OUT/run.log"; trap 'echo "[$(date)] EXIT $? (focus_6am)" >>"$LOG"' EXIT
gpu_wait(){ while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -lt "$1" ]; do sleep 20; done; }
gpu_wait 6000
echo "[$(date)] focus_6am UNCONSTRAINED recovery (all 20 AA), T=0.1, 5000/structure" | tee "$LOG"
$PY ProteinMPNN/protein_mpnn_run.py --jsonl_path "$OUT/parsed.jsonl" --chain_id_jsonl "$OUT/chain_id.jsonl" \
   --out_folder "$OUT" --num_seq_per_target 5000 --batch_size 8 --sampling_temp "0.1" --seed 37 --model_name v_48_020 >>"$LOG" 2>&1
$PY focus_recovery.py "$OUT" 2>>"$LOG" | tee "$OUT/recovery_table.txt"
echo "[$(date)] DONE focus_6am" | tee -a "$LOG"
touch "$OUT/COMPLETE"
