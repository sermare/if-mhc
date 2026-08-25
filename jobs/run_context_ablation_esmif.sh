#!/usr/bin/env bash
# ESM-IF context-ablation on 3HG1 -- same 3 conditions as jobs/run_context_ablation_mpnn.sh
# (nocontext: chain C alone; mhconly: A+B+C; tcronly: C+D+E), using the ORIGINAL 3HG1.pdb directly
# (no filtered-PDB prep needed -- esmif_sample.py's --chains already controls what gets loaded/
# concatenated as context, verified: coords dict built from exactly the requested chains).
# Usage: CONDITION=nocontext|mhconly|tcronly bash jobs/run_context_ablation_esmif.sh
set -uo pipefail
cd /home/ubuntu/if-mhc; ABS=/home/ubuntu/if-mhc
PY=/home/ubuntu/miniforge3/envs/esmcba/bin/python
CONDITION="${CONDITION:?set CONDITION=nocontext|mhconly|tcronly}"
declare -A CHAINS=( [nocontext]="C" [mhconly]="A B C" [tcronly]="C D E" )
OUT="$ABS/outputs/context_ablation_3hg1/esmif_$CONDITION"; mkdir -p "$OUT/seqs"
NSEQ="${NSEQ:-5000}"; BATCH="${BATCH:-8}"; TEMP="${TEMP:-0.1}"
LOG="$OUT/run.log"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
export TORCH_HOME="$ABS/models_cache/torch_hub"
gpu_wait(){ while true; do f="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)"; if [[ "$f" =~ ^[0-9]+$ ]] && [ "$f" -ge "$1" ]; then sleep "0.$((RANDOM % 9 + 1))"; f2="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)"; [[ "$f2" =~ ^[0-9]+$ ]] && [ "$f2" -ge "$1" ] && break; fi; sleep $((15 + RANDOM % 15)); done; }
exec 9>"$OUT/runner.lock"; flock -n 9 || { echo "another runner holds lock — exit"; exit 0; }
echo $$ >"$OUT/runner.pid"; trap 'rm -f "$OUT/runner.pid"' EXIT

echo "[$(date '+%F %T')] === esmif context_ablation/$CONDITION: 3HG1 x $NSEQ seqs, T=$TEMP, chains=${CHAINS[$CONDITION]} ===" | tee -a "$LOG"
gpu_wait 8000

$PY py/esmif_sample_panel.py \
    --pdb inputs/pmhc_tcr_dataset/3HG1.pdb \
    --chains ${CHAINS[$CONDITION]} \
    --target-chain C \
    --num-seqs "$NSEQ" \
    --batch-size "$BATCH" \
    --temperature "$TEMP" \
    --out "$OUT/seqs/3HG1.fa" >>"$LOG" 2>&1

got=$(grep -c "^>" "$OUT/seqs/3HG1.fa" 2>/dev/null || echo 0)
echo "[$(date '+%F %T')] esmif context_ablation/$CONDITION done: $got/$NSEQ seqs" | tee -a "$LOG"
[ "${got:-0}" -ge "$NSEQ" ] && touch "$OUT/COMPLETE" && echo "[$(date '+%F %T')] === COMPLETE ===" | tee -a "$LOG"
