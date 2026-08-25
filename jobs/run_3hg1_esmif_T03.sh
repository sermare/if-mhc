#!/usr/bin/env bash
# ESM-IF1 T=0.3/20K campaign on 3HG1 (MEL5 TCR / MART-1 / ELAGIGILTV) -- companion to
# jobs/run_3hg1_esmif_pilot.sh (T=0.1/10K). Separate output dir so the T=0.1 pilot data is untouched.
# Vanilla/noMHC ProteinMPNN already have 50K at T=0.3 (outputs/mpnn_3hg1_T03_50k) -- plenty, not
# rerun. ESM-IF and LigandMPNN had no T=0.3 data before this, hence this + the LigandMPNN companion
# script. See MATCHED_TCR_TRACKING.md.
#
# gpu_wait's 12000MB floor doubles as the sequencing guard against the T=0.1 pilot possibly still
# running on the same GPU (each ESM-IF batch=8 run peaks at ~10.9GB; two concurrent instances would
# OOM on this 22GB L4) -- this script just blocks until enough memory is free rather than needing to
# be launched only after the other one finishes.
set -uo pipefail
cd /home/ubuntu/if-mhc; ABS=/home/ubuntu/if-mhc
PY=/home/ubuntu/miniforge3/envs/esmcba/bin/python
OUT="$ABS/outputs/esmif_3hg1_T03_20k"; mkdir -p "$OUT/seqs"
NSEQ="${NSEQ:-20000}"; BATCH="${BATCH:-8}"; TEMP="${TEMP:-0.3}"
LOG="$OUT/run.log"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
export TORCH_HOME="$ABS/models_cache/torch_hub"
gpu_wait(){ while true; do f="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)"; if [[ "$f" =~ ^[0-9]+$ ]] && [ "$f" -ge "$1" ]; then sleep "0.$((RANDOM % 9 + 1))"; f2="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)"; [[ "$f2" =~ ^[0-9]+$ ]] && [ "$f2" -ge "$1" ] && break; fi; sleep $((15 + RANDOM % 15)); done; }
exec 9>"$OUT/runner.lock"; flock -n 9 || { echo "another runner holds lock — exit"; exit 0; }
echo $$ >"$OUT/runner.pid"; trap 'rm -f "$OUT/runner.pid"' EXIT

echo "[$(date '+%F %T')] === esmif_3hg1_T03_20k: MEL5 TCR/MART-1 x $NSEQ seqs, T=$TEMP ===" | tee -a "$LOG"
gpu_wait 12000

$PY py/esmif_sample_panel.py \
    --pdb inputs/pmhc_tcr_dataset/3HG1.pdb \
    --chains A B C D E \
    --target-chain C \
    --num-seqs "$NSEQ" \
    --batch-size "$BATCH" \
    --temperature "$TEMP" \
    --out "$OUT/seqs/3HG1.fa" >>"$LOG" 2>&1

got=$(grep -c "^>" "$OUT/seqs/3HG1.fa" 2>/dev/null || echo 0)
echo "[$(date '+%F %T')] esmif_3hg1_T03_20k done: $got/$NSEQ seqs" | tee -a "$LOG"
if [ "${got:-0}" -ge "$NSEQ" ]; then
  touch "$OUT/COMPLETE"
  echo "[$(date '+%F %T')] === esmif_3hg1_T03_20k COMPLETE ===" | tee -a "$LOG"
else
  echo "[$(date '+%F %T')] WARN short of target ($got/$NSEQ) -- rerun to resume" | tee -a "$LOG"
fi
