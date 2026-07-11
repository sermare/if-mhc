#!/usr/bin/env bash
# run_shard2.sh <outdir> <weights_dir> <model_name> <tag> <nseq> <batch> [seed]
# generalized parallel ProteinMPNN shard (reuses pre-split parsed_/assigned_ jsonl in <outdir>)
set -euo pipefail
cd /home/ubuntu/if-mhc
PY=/home/ubuntu/miniforge3/envs/esmcba/bin/python
OUT=$1; WEIGHTS=$2; MODEL=$3; TAG=$4; NSEQ=$5; BATCH=$6; SEED=${7:-37}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
rm -f "$OUT/logs/$TAG.DONE"; mkdir -p "$OUT/logs"
echo "[$(date)] shard=$TAG model=$MODEL nseq=$NSEQ batch=$BATCH seed=$SEED" > "$OUT/logs/$TAG.log"
$PY ProteinMPNN/protein_mpnn_run.py \
   --jsonl_path "$OUT/parsed_$TAG.jsonl" \
   --chain_id_jsonl "$OUT/assigned_$TAG.jsonl" \
   --out_folder "$OUT" \
   --num_seq_per_target "$NSEQ" --batch_size "$BATCH" \
   --sampling_temp "0.3" --seed "$SEED" \
   --path_to_model_weights "$WEIGHTS" --model_name "$MODEL" >> "$OUT/logs/$TAG.log" 2>&1
echo "[$(date)] DONE $TAG" >> "$OUT/logs/$TAG.log"
touch "$OUT/logs/$TAG.DONE"
