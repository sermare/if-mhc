#!/usr/bin/env bash
# LigandMPNN campaign on 2P5E (1G4-c58/c61 TCR / NY-ESO-1 / SLLMWITQC), matching 3HG1's convention
# (jobs/run_3hg1_ligandmpnn_pilot.sh, jobs/run_3hg1_ligandmpnn_T03.sh). BATCH=32 + gpu_wait 6000
# default (not 64/3000) given this GPU is shared with other unrelated processes -- see
# MATCHED_TCR_TRACKING.md for the OOM history at batch=64 that motivated this. Usage:
#   NSEQ=10000 TEMP=0.1 OUT=outputs/ligandmpnn_2p5e_pilot bash jobs/run_2p5e_ligandmpnn.sh
#   NSEQ=20000 TEMP=0.3 OUT=outputs/ligandmpnn_2p5e_T03_20k bash jobs/run_2p5e_ligandmpnn.sh
set -uo pipefail
cd /home/ubuntu/if-mhc; ABS=/home/ubuntu/if-mhc
PY=/home/ubuntu/miniforge3/envs/esmcba/bin/python
OUT="${OUT:?set OUT=outputs/ligandmpnn_2p5e_pilot or ligandmpnn_2p5e_T03_20k}"
# resolve to absolute -- the script does `cd LigandMPNN` before invoking run.py, so a relative $OUT
# (as passed by every caller, e.g. OUT=outputs/ligandmpnn_2p5e_pilot) breaks the `>>"$LOG"` redirect
# at that point (bash tries to open it relative to LigandMPNN/, not $ABS/) -- confirmed root cause of
# repeated silent zero-output failures (2026-07-29): the redirect setup fails BEFORE run.py ever
# executes, so nothing gets logged and the failure looks instantaneous.
[[ "$OUT" = /* ]] || OUT="$ABS/$OUT"
mkdir -p "$OUT"
NSEQ="${NSEQ:-10000}"; BATCH="${BATCH:-32}"; TEMP="${TEMP:-0.1}"
NBATCHES=$(( (NSEQ + BATCH - 1) / BATCH ))
LOG="$OUT/run.log"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
# gpu_wait re-checks memory is STILL free after a random jitter -- with 13+ jobs all polling at once,
# several can see the same "free" reading in the same instant and all proceed together (a thundering
# herd), each grabbing memory the others assumed was theirs -> instant CUDA crash with zero flushed
# output (confirmed 2026-07-29: manual replay with free memory succeeds every time; production runs
# died silently exactly when several other queued jobs also unblocked at once). The re-check + random
# sleep breaks the herd up.
gpu_wait(){ while true; do f="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)"; if [[ "$f" =~ ^[0-9]+$ ]] && [ "$f" -ge "$1" ]; then sleep "0.$((RANDOM % 9 + 1))"; f2="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)"; [[ "$f2" =~ ^[0-9]+$ ]] && [ "$f2" -ge "$1" ] && break; fi; sleep $((15 + RANDOM % 15)); done; }
exec 9>"$OUT/runner.lock"; flock -n 9 || { echo "another runner holds lock — exit"; exit 0; }
echo $$ >"$OUT/runner.pid"; trap 'rm -f "$OUT/runner.pid"' EXIT

count_seqs(){ [ -f "$OUT/seqs/2P5E.fa" ] && grep -c "^>" "$OUT/seqs/2P5E.fa" 2>/dev/null || echo 0; }
have=$(count_seqs); have=${have:-0}
if [ "$have" -ge "$NSEQ" ]; then
  echo "[$(date '+%F %T')] already at $have/$NSEQ -- skip" | tee -a "$LOG"
  touch "$OUT/COMPLETE"
  exit 0
fi

echo "[$(date '+%F %T')] === ligandmpnn 2P5E: 1G4c58c61/NY-ESO-1 x $NSEQ seqs, T=$TEMP (have=$have) ===" | tee -a "$LOG"
gpu_wait 6000

cd LigandMPNN
"$PY" run.py \
    --model_type "ligand_mpnn" \
    --checkpoint_ligand_mpnn "./model_params/ligandmpnn_v_32_010_25.pt" \
    --pdb_path "$ABS/inputs/pmhc_tcr_dataset/2P5E.pdb" \
    --parse_these_chains_only "A,B,C,D,E" \
    --chains_to_design "C" \
    --out_folder "$OUT" \
    --batch_size "$BATCH" \
    --number_of_batches "$NBATCHES" \
    --temperature "$TEMP" \
    --seed 41 >>"$LOG" 2>&1
cd "$ABS"

got=$(count_seqs)
echo "[$(date '+%F %T')] ligandmpnn 2P5E done: $got seqs (requested $((BATCH*NBATCHES)), target $NSEQ)" | tee -a "$LOG"
if [ "${got:-0}" -ge "$NSEQ" ]; then
  touch "$OUT/COMPLETE"
  echo "[$(date '+%F %T')] === ligandmpnn 2P5E COMPLETE ($OUT) ===" | tee -a "$LOG"
fi

rm -rf "$OUT/backbones" "$OUT/packed"
echo "[$(date '+%F %T')] cleaned up backbones/ + packed/ (redundant per-design backbone dumps)" | tee -a "$LOG"
