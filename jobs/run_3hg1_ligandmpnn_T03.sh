#!/usr/bin/env bash
# LigandMPNN T=0.3/20K campaign on 3HG1 (MEL5 TCR / MART-1 / ELAGIGILTV) -- companion to
# jobs/run_3hg1_ligandmpnn_pilot.sh (T=0.1/10K). Separate output dir so the T=0.1 pilot data is
# untouched. See jobs/run_3hg1_esmif_T03.sh / MATCHED_TCR_TRACKING.md for why only ESM-IF and
# LigandMPNN get a new T=0.3 run (vanilla/noMHC ProteinMPNN already have 50K each at T=0.3).
#
# BATCH=32 + gpu_wait 6000 (down from 64/3000) after a first attempt OOM'd: this GPU is shared with
# OTHER unrelated jobs not ours to control (pmhc's sweep_nyeso1_*.py, ~340MB each) plus whatever ESM-IF
# T=0.3 run is doing concurrently (~11.6GB) -- 3000MB of headroom wasn't enough margin against that.
set -uo pipefail
cd /home/ubuntu/if-mhc; ABS=/home/ubuntu/if-mhc
PY=/home/ubuntu/miniforge3/envs/esmcba/bin/python
OUT="$ABS/outputs/ligandmpnn_3hg1_T03_20k"; mkdir -p "$OUT"
NSEQ="${NSEQ:-20000}"; BATCH="${BATCH:-32}"; TEMP="${TEMP:-0.3}"
NBATCHES=$(( (NSEQ + BATCH - 1) / BATCH ))
LOG="$OUT/run.log"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
gpu_wait(){ while true; do f="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)"; if [[ "$f" =~ ^[0-9]+$ ]] && [ "$f" -ge "$1" ]; then sleep "0.$((RANDOM % 9 + 1))"; f2="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)"; [[ "$f2" =~ ^[0-9]+$ ]] && [ "$f2" -ge "$1" ] && break; fi; sleep $((15 + RANDOM % 15)); done; }
exec 9>"$OUT/runner.lock"; flock -n 9 || { echo "another runner holds lock — exit"; exit 0; }
echo $$ >"$OUT/runner.pid"; trap 'rm -f "$OUT/runner.pid"' EXIT

count_seqs(){ [ -f "$OUT/seqs/3HG1.fa" ] && grep -c "^>" "$OUT/seqs/3HG1.fa" 2>/dev/null || echo 0; }
have=$(count_seqs); have=${have:-0}
if [ "$have" -ge "$NSEQ" ]; then
  echo "[$(date '+%F %T')] already at $have/$NSEQ -- skip" | tee -a "$LOG"
  touch "$OUT/COMPLETE"
  exit 0
fi

echo "[$(date '+%F %T')] === ligandmpnn_3hg1_T03_20k: MEL5 TCR/MART-1 x $NSEQ seqs, T=$TEMP (have=$have) ===" | tee -a "$LOG"
gpu_wait 6000

cd LigandMPNN
"$PY" run.py \
    --model_type "ligand_mpnn" \
    --checkpoint_ligand_mpnn "./model_params/ligandmpnn_v_32_010_25.pt" \
    --pdb_path "$ABS/inputs/pmhc_tcr_dataset/3HG1.pdb" \
    --parse_these_chains_only "A,B,C,D,E" \
    --chains_to_design "C" \
    --out_folder "$OUT" \
    --batch_size "$BATCH" \
    --number_of_batches "$NBATCHES" \
    --temperature "$TEMP" \
    --seed 41 >>"$LOG" 2>&1
cd "$ABS"

got=$(count_seqs)
echo "[$(date '+%F %T')] ligandmpnn_3hg1_T03_20k done: $got seqs (requested $((BATCH*NBATCHES)), target $NSEQ)" | tee -a "$LOG"
if [ "${got:-0}" -ge "$NSEQ" ]; then
  touch "$OUT/COMPLETE"
  echo "[$(date '+%F %T')] === ligandmpnn_3hg1_T03_20k COMPLETE ===" | tee -a "$LOG"
fi

rm -rf "$OUT/backbones" "$OUT/packed"
echo "[$(date '+%F %T')] cleaned up backbones/ + packed/ (redundant per-design backbone dumps)" | tee -a "$LOG"
