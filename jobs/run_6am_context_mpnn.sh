#!/usr/bin/env bash
# 10k ProteinMPNN + noMHC designs on 6AM5/6AMU, matching the paper's convention (10,000 seqs,
# T=0.1, peptide chain C designed), under two contexts:
#   full     A+B+C+D+E  -- MHC + b2m + peptide + TCR
#   mhconly  A+B+C      -- TCR REMOVED, MHC + b2m + peptide (the paper's ablation)
# 6AM5/6AMU are not in the paper's 20-structure panel; this adds the cross-reactive DMF5 pair on the
# same footing. Inputs from py/prep_6am_context_inputs.py.
set -uo pipefail
cd /home/ubuntu/if-mhc; ABS=/home/ubuntu/if-mhc
PY=/home/ubuntu/miniforge3/envs/esmcba/bin/python
NSEQ="${NSEQ:-10000}"; BATCH="${BATCH:-8}"; TEMP="${TEMP:-0.1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
declare -A WEIGHTS_DIR=( [vanilla]="" [nomhc]="ProteinMPNN/nomhc_model_weights/" )
declare -A MODEL_NAME=( [vanilla]="v_48_020" [nomhc]="proteinmpnn_nomhc" )
gpu_wait(){ while true; do f="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)"; if [[ "$f" =~ ^[0-9]+$ ]] && [ "$f" -ge "$1" ]; then sleep "0.$((RANDOM % 9 + 1))"; f2="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)"; [[ "$f2" =~ ^[0-9]+$ ]] && [ "$f2" -ge "$1" ] && break; fi; sleep $((15 + RANDOM % 15)); done; }
LOG="$ABS/outputs/context_6am/run.log"

for S in 6AM5 6AMU; do
  for COND in full mhconly; do
    CELL="$ABS/outputs/context_6am/${S}_${COND}"; NAME="${S}_${COND}"
    mkdir -p "$CELL/seqs"
    for W in vanilla nomhc; do
      TAG="${W}_${NAME}"
      have=$( [ -f "$CELL/seqs/$TAG.fa" ] && grep -c "^>" "$CELL/seqs/$TAG.fa" || echo 0 )
      if [ "${have:-0}" -ge "$NSEQ" ]; then
        echo "[$(date '+%F %T')] $TAG already $have/$NSEQ -- skip" | tee -a "$LOG"; continue
      fi
      gpu_wait 4000
      echo "[$(date '+%F %T')] === $TAG: $NSEQ seqs, T=$TEMP ===" | tee -a "$LOG"
      wdir="${WEIGHTS_DIR[$W]}"; extra=()
      [ -n "$wdir" ] && extra=(--path_to_model_weights "$wdir")
      "$PY" ProteinMPNN/protein_mpnn_run.py \
        --jsonl_path "$CELL/parsed.jsonl" --chain_id_jsonl "$CELL/chain_id.jsonl" \
        --out_folder "$CELL" --num_seq_per_target "$NSEQ" --batch_size "$BATCH" \
        --sampling_temp "$TEMP" --seed 37 --model_name "${MODEL_NAME[$W]}" \
        "${extra[@]}" >>"$LOG" 2>&1
      mv -f "$CELL/seqs/$NAME.fa" "$CELL/seqs/$TAG.fa" 2>/dev/null
      got=$( [ -f "$CELL/seqs/$TAG.fa" ] && grep -c "^>" "$CELL/seqs/$TAG.fa" || echo 0 )
      echo "[$(date '+%F %T')] $TAG done: $got seqs" | tee -a "$LOG"
    done
  done
done
echo "[$(date '+%F %T')] ALL DONE" | tee -a "$LOG"
