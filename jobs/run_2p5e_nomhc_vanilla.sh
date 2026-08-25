#!/usr/bin/env bash
# vanilla vs noMHC ProteinMPNN on 2P5E (1G4-c58/c61 TCR / NY-ESO-1 / SLLMWITQC), same 5-chain
# convention as 3HG1 (A=MHC, B=b2m, C=peptide designed, D=TCRa, E=TCRb fixed). 2P5E already has
# plenty of vanilla data at various temps (7temp sweep, tempsweep, 50k/50k_part2/finish50k/50k_noM,
# dataset_protocol -- all v_48_020) but ZERO noMHC data -- this campaign fills that gap and adds a
# fresh matched vanilla run at the SAME temp/scale for a clean apples-to-apples pair (rather than
# reusing the old scattered-temp vanilla runs). Usage: TEMP=0.1|0.3 NSEQ=20000 OUT=<dir> bash
# jobs/run_2p5e_nomhc_vanilla.sh
set -uo pipefail
cd /home/ubuntu/if-mhc; ABS=/home/ubuntu/if-mhc
PY=/home/ubuntu/miniforge3/envs/esmcba/bin/python
OUT="${OUT:?set OUT=outputs/mpnn_2p5e_T01_20k or mpnn_2p5e_T03_20k}"; mkdir -p "$OUT/seqs"
NSEQ="${NSEQ:-20000}"; BATCH="${BATCH:-8}"; TEMP="${TEMP:-0.1}"
LOG="$OUT/run.log"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
gpu_wait(){ while true; do f="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)"; if [[ "$f" =~ ^[0-9]+$ ]] && [ "$f" -ge "$1" ]; then sleep "0.$((RANDOM % 9 + 1))"; f2="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)"; [[ "$f2" =~ ^[0-9]+$ ]] && [ "$f2" -ge "$1" ] && break; fi; sleep $((15 + RANDOM % 15)); done; }
exec 9>"$OUT/runner.lock"; flock -n 9 || { echo "another runner holds lock — exit"; exit 0; }
echo $$ >"$OUT/runner.pid"; trap 'rm -f "$OUT/runner.pid"' EXIT

echo "[$(date '+%F %T')] === 2p5e: 1G4c58c61 TCR/NY-ESO-1 x $NSEQ seqs, T=$TEMP, vanilla+noMHC ===" | tee -a "$LOG"

count_seqs(){ [ -f "$OUT/seqs/$1.fa" ] && grep -c "^>" "$OUT/seqs/$1.fa" 2>/dev/null || echo 0; }

declare -A WEIGHTS_DIR=( [vanilla]="" [nomhc]="ProteinMPNN/nomhc_model_weights/" )
declare -A MODEL_NAME=( [vanilla]="v_48_020" [nomhc]="proteinmpnn_nomhc" )

for weights in vanilla nomhc; do
  tag="${weights}_2P5E"
  if [ -f "$OUT/seqs/2P5E.fa" ]; then
    mv -f "$OUT/seqs/2P5E.fa" "$OUT/seqs/$tag.fa"
    echo "[$(date '+%F %T')] recovered interrupted-run leftover seqs/2P5E.fa -> seqs/$tag.fa" | tee -a "$LOG"
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
     --jsonl_path "$OUT/parsed.jsonl" \
     --chain_id_jsonl "$OUT/chain_id.jsonl" \
     --out_folder "$OUT" \
     --num_seq_per_target "$NSEQ" \
     --batch_size "$BATCH" \
     --sampling_temp "$TEMP" \
     --seed 37 \
     --model_name "${MODEL_NAME[$weights]}" \
     "${extra_args[@]}" >>"$LOG" 2>&1
  mv -f "$OUT/seqs/2P5E.fa" "$OUT/seqs/$tag.fa" 2>/dev/null
  got=$(count_seqs "$tag")
  echo "[$(date '+%F %T')] target=$tag done: $got seqs" | tee -a "$LOG"
  [ "$got" -lt "$NSEQ" ] && echo "[$(date '+%F %T')] WARN $tag short of target ($got/$NSEQ) -- will retry next invocation" | tee -a "$LOG"
done

all_done=1
for weights in vanilla nomhc; do
  have=$(count_seqs "${weights}_2P5E"); [ "${have:-0}" -lt "$NSEQ" ] && all_done=0
done
if [ "$all_done" = 1 ]; then
  echo "[$(date '+%F %T')] === 2p5e ALL TARGETS REACHED $NSEQ ===" | tee -a "$LOG"
  touch "$OUT/COMPLETE"
else
  echo "[$(date '+%F %T')] === pass complete, some targets still short -- supervisor will retry ===" | tee -a "$LOG"
fi
