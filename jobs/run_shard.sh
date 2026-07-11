#!/usr/bin/env bash
# run_shard.sh <tag> <nseq> <batch>  -- one parallel noMHC MPNN shard
set -euo pipefail
cd /home/ubuntu/if-mhc
PY=/home/ubuntu/miniforge3/envs/esmcba/bin/python
OUT=outputs/mpnn_nomhc_allbb_deep
WEIGHTS=ProteinMPNN/nomhc_model_weights/
TAG=$1; NSEQ=$2; BATCH=$3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
rm -f "$OUT/logs/$TAG.DONE"
echo "[$(date)] shard=$TAG nseq=$NSEQ batch=$BATCH" > "$OUT/logs/$TAG.log"
$PY ProteinMPNN/protein_mpnn_run.py \
   --jsonl_path "$OUT/parsed_$TAG.jsonl" \
   --chain_id_jsonl "$OUT/assigned_$TAG.jsonl" \
   --out_folder "$OUT" \
   --num_seq_per_target "$NSEQ" \
   --batch_size "$BATCH" \
   --sampling_temp "0.3" --seed 37 \
   --path_to_model_weights "$WEIGHTS" --model_name proteinmpnn_nomhc >> "$OUT/logs/$TAG.log" 2>&1
echo "[$(date)] DONE $TAG" >> "$OUT/logs/$TAG.log"
touch "$OUT/logs/$TAG.DONE"
