#!/usr/bin/env bash
# LigandMPNN context-ablation on 3HG1 -- same 3 conditions, via --parse_these_chains_only directly
# on the original PDB (no filtered-file prep needed, unlike ProteinMPNN's parse_multiple_chains.py
# which needs the subset physically absent from the input file).
# Usage: CONDITION=nocontext|mhconly|tcronly bash jobs/run_context_ablation_ligandmpnn.sh
set -uo pipefail
cd /home/ubuntu/if-mhc; ABS=/home/ubuntu/if-mhc
PY=/home/ubuntu/miniforge3/envs/esmcba/bin/python
CONDITION="${CONDITION:?set CONDITION=nocontext|mhconly|tcronly}"
declare -A CHAINS=( [nocontext]="C" [mhconly]="A,B,C" [tcronly]="C,D,E" )
OUT="$ABS/outputs/context_ablation_3hg1/ligandmpnn_$CONDITION"; mkdir -p "$OUT"
NSEQ="${NSEQ:-5000}"; BATCH="${BATCH:-32}"; TEMP="${TEMP:-0.1}"
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
  touch "$OUT/COMPLETE"; exit 0
fi

echo "[$(date '+%F %T')] === ligandmpnn context_ablation/$CONDITION: 3HG1 x $NSEQ seqs, T=$TEMP, chains=${CHAINS[$CONDITION]} ===" | tee -a "$LOG"
gpu_wait 6000

cd LigandMPNN
"$PY" run.py \
    --model_type "ligand_mpnn" \
    --checkpoint_ligand_mpnn "./model_params/ligandmpnn_v_32_010_25.pt" \
    --pdb_path "$ABS/inputs/pmhc_tcr_dataset/3HG1.pdb" \
    --parse_these_chains_only "${CHAINS[$CONDITION]}" \
    --chains_to_design "C" \
    --out_folder "$OUT" \
    --batch_size "$BATCH" \
    --number_of_batches "$NBATCHES" \
    --temperature "$TEMP" \
    --seed 41 >>"$LOG" 2>&1
cd "$ABS"

got=$(count_seqs)
echo "[$(date '+%F %T')] ligandmpnn context_ablation/$CONDITION done: $got seqs" | tee -a "$LOG"
[ "${got:-0}" -ge "$NSEQ" ] && touch "$OUT/COMPLETE" && echo "[$(date '+%F %T')] === COMPLETE ===" | tee -a "$LOG"
rm -rf "$OUT/backbones" "$OUT/packed"
