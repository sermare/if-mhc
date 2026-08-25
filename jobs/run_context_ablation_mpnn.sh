#!/usr/bin/env bash
# Context-ablation ProteinMPNN runs on 3HG1: how much does REMOVING the physical MHC/TCR structural
# context (not just the model's training weights) change what gets designed? Three conditions, each
# run with both vanilla and noMHC weights:
#   nocontext -- chain C (peptide) parsed completely alone, no other chains present at all
#   mhconly   -- chains A+B+C only (MHC+b2m+peptide), TCR removed
#   tcronly   -- chains C+D+E only (peptide+TCRa+TCRb), MHC removed
# (full-context data already exists: outputs/mpnn_3hg1_100k, outputs/mpnn_3hg1_T03_50k)
#
# This is a DIFFERENT axis than vanilla-vs-noMHC WEIGHTS (which is about what was in the model's
# training data) -- this ablates what's physically PRESENT at design time, regardless of weights.
# Small pilot scale (5k/condition) since this is diagnostic, not meant to match the 20-50k main
# campaigns. Usage: CONDITION=nocontext|mhconly|tcronly bash jobs/run_context_ablation_mpnn.sh
set -uo pipefail
cd /home/ubuntu/if-mhc; ABS=/home/ubuntu/if-mhc
PY=/home/ubuntu/miniforge3/envs/esmcba/bin/python
CONDITION="${CONDITION:?set CONDITION=nocontext|mhconly|tcronly}"
IN="$ABS/outputs/context_ablation_3hg1/mpnn_$CONDITION"
OUT="$IN"; mkdir -p "$OUT/seqs"
NSEQ="${NSEQ:-5000}"; BATCH="${BATCH:-8}"; TEMP="${TEMP:-0.1}"
LOG="$OUT/run.log"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
gpu_wait(){ while true; do f="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)"; if [[ "$f" =~ ^[0-9]+$ ]] && [ "$f" -ge "$1" ]; then sleep "0.$((RANDOM % 9 + 1))"; f2="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)"; [[ "$f2" =~ ^[0-9]+$ ]] && [ "$f2" -ge "$1" ] && break; fi; sleep $((15 + RANDOM % 15)); done; }
exec 9>"$OUT/runner.lock"; flock -n 9 || { echo "another runner holds lock — exit"; exit 0; }
echo $$ >"$OUT/runner.pid"; trap 'rm -f "$OUT/runner.pid"' EXIT

echo "[$(date '+%F %T')] === context_ablation/$CONDITION: 3HG1 x $NSEQ seqs, T=$TEMP, vanilla+noMHC ===" | tee -a "$LOG"

count_seqs(){ [ -f "$OUT/seqs/$1.fa" ] && grep -c "^>" "$OUT/seqs/$1.fa" 2>/dev/null || echo 0; }

declare -A WEIGHTS_DIR=( [vanilla]="" [nomhc]="ProteinMPNN/nomhc_model_weights/" )
declare -A MODEL_NAME=( [vanilla]="v_48_020" [nomhc]="proteinmpnn_nomhc" )
PDBNAME="3HG1_${CONDITION}"

for weights in vanilla nomhc; do
  tag="${weights}_${PDBNAME}"
  if [ -f "$OUT/seqs/$PDBNAME.fa" ]; then
    mv -f "$OUT/seqs/$PDBNAME.fa" "$OUT/seqs/$tag.fa"
  fi
  have=$(count_seqs "$tag"); have=${have:-0}
  if [ "$have" -ge "$NSEQ" ]; then
    echo "[$(date '+%F %T')] $tag already at $have/$NSEQ -- skip" | tee -a "$LOG"; continue
  fi
  gpu_wait 4000
  echo "[$(date '+%F %T')] target=$tag have=$have -> generating $NSEQ (T=$TEMP, weights=$weights)" | tee -a "$LOG"
  wdir="${WEIGHTS_DIR[$weights]}"
  extra_args=()
  [ -n "$wdir" ] && extra_args=(--path_to_model_weights "$wdir")
  $PY ProteinMPNN/protein_mpnn_run.py \
     --jsonl_path "$IN/parsed.jsonl" \
     --chain_id_jsonl "$IN/chain_id.jsonl" \
     --out_folder "$OUT" \
     --num_seq_per_target "$NSEQ" \
     --batch_size "$BATCH" \
     --sampling_temp "$TEMP" \
     --seed 37 \
     --model_name "${MODEL_NAME[$weights]}" \
     "${extra_args[@]}" >>"$LOG" 2>&1
  mv -f "$OUT/seqs/$PDBNAME.fa" "$OUT/seqs/$tag.fa" 2>/dev/null
  got=$(count_seqs "$tag")
  echo "[$(date '+%F %T')] target=$tag done: $got seqs" | tee -a "$LOG"
done

all_done=1
for weights in vanilla nomhc; do
  have=$(count_seqs "${weights}_${PDBNAME}"); [ "${have:-0}" -lt "$NSEQ" ] && all_done=0
done
[ "$all_done" = 1 ] && touch "$OUT/COMPLETE" && echo "[$(date '+%F %T')] === $CONDITION COMPLETE ===" | tee -a "$LOG"
