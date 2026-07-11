#!/usr/bin/env bash
set -uo pipefail
cd /home/ubuntu/if-mhc
PY=/home/ubuntu/miniforge3/envs/esmcba/bin/python; NB=/home/ubuntu/miniforge3/bin/python
DP=outputs/dataset_protocol; OUT=outputs/dataset_softP; mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2
LOG="$OUT/run.log"; trap 'echo "[$(date)] EXIT $? (softP)" >>"$LOG"' EXIT
gpu_wait(){ while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -lt "$1" ]; do sleep 20; done; }
gpu_wait 6000
echo "[$(date)] softP: bias Proline -2.0 (downweight, not forbidden) + no Met@P1" | tee "$LOG"
$PY ProteinMPNN/protein_mpnn_run.py --jsonl_path "$DP/parsed.jsonl" --chain_id_jsonl "$DP/chain_id.jsonl" \
   --omit_AA_jsonl "$DP/omit.jsonl" --bias_AA_jsonl bias_P.json \
   --out_folder "$OUT" --num_seq_per_target 2000 --batch_size 8 --sampling_temp "0.1" --seed 37 --model_name v_48_020 >>"$LOG" 2>&1
$NB ds_recovery.py "$OUT" inputs/pmhc_tcr_dataset/dataset.csv 2>>"$LOG" | tee "$OUT/recovery_table.txt"
touch "$OUT/COMPLETE"
