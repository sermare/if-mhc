#!/usr/bin/env bash
# ESM-IF1 campaign on 2P5E (1G4-c58/c61 TCR / NY-ESO-1 / SLLMWITQC), matching 3HG1's convention
# (jobs/run_3hg1_esmif_pilot.sh, jobs/run_3hg1_esmif_T03.sh). Usage:
#   NSEQ=10000 TEMP=0.1 OUT=outputs/esmif_2p5e_pilot bash jobs/run_2p5e_esmif.sh
#   NSEQ=20000 TEMP=0.3 OUT=outputs/esmif_2p5e_T03_20k bash jobs/run_2p5e_esmif.sh
set -uo pipefail
cd /home/ubuntu/if-mhc; ABS=/home/ubuntu/if-mhc
PY=/home/ubuntu/miniforge3/envs/esmcba/bin/python
OUT="${OUT:?set OUT=outputs/esmif_2p5e_pilot or esmif_2p5e_T03_20k}"; mkdir -p "$OUT/seqs"
NSEQ="${NSEQ:-10000}"; BATCH="${BATCH:-8}"; TEMP="${TEMP:-0.1}"
LOG="$OUT/run.log"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
export TORCH_HOME="$ABS/models_cache/torch_hub"
gpu_wait(){ while true; do f="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)"; if [[ "$f" =~ ^[0-9]+$ ]] && [ "$f" -ge "$1" ]; then sleep "0.$((RANDOM % 9 + 1))"; f2="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)"; [[ "$f2" =~ ^[0-9]+$ ]] && [ "$f2" -ge "$1" ] && break; fi; sleep $((15 + RANDOM % 15)); done; }
exec 9>"$OUT/runner.lock"; flock -n 9 || { echo "another runner holds lock — exit"; exit 0; }
echo $$ >"$OUT/runner.pid"; trap 'rm -f "$OUT/runner.pid"' EXIT

echo "[$(date '+%F %T')] === esmif 2P5E: 1G4c58c61/NY-ESO-1 x $NSEQ seqs, T=$TEMP ===" | tee -a "$LOG"
gpu_wait 12000

$PY py/esmif_sample_panel.py \
    --pdb inputs/pmhc_tcr_dataset/2P5E.pdb \
    --chains A B C D E \
    --target-chain C \
    --num-seqs "$NSEQ" \
    --batch-size "$BATCH" \
    --temperature "$TEMP" \
    --out "$OUT/seqs/2P5E.fa" >>"$LOG" 2>&1

got=$(grep -c "^>" "$OUT/seqs/2P5E.fa" 2>/dev/null || echo 0)
echo "[$(date '+%F %T')] esmif 2P5E done: $got/$NSEQ seqs" | tee -a "$LOG"
if [ "${got:-0}" -ge "$NSEQ" ]; then
  touch "$OUT/COMPLETE"
  echo "[$(date '+%F %T')] === esmif 2P5E COMPLETE ($OUT) ===" | tee -a "$LOG"
else
  echo "[$(date '+%F %T')] WARN short of target ($got/$NSEQ) -- rerun to resume" | tee -a "$LOG"
fi
