#!/usr/bin/env bash
# ESM-IF1 on one panel structure/condition, reading the pre-filtered/renamed A=MHC,B=b2m,C=peptide,
# D=TCRa,E=TCRb PDB built by py/prep_panel_structures.py.
# Usage: PDB=1QSF CONDITION=full|mhconly bash jobs/run_panel_esmif.sh
set -uo pipefail
cd /home/ubuntu/if-mhc; ABS=/home/ubuntu/if-mhc
PY=/home/ubuntu/miniforge3/envs/esmcba/bin/python
PDB="${PDB:?set PDB=<panel pdb id>}"
CONDITION="${CONDITION:?set CONDITION=full|mhconly}"
declare -A CHAINS=( [full]="A B C D E" [mhconly]="A B C" )
IN_PDB="$ABS/outputs/panel_prep/$PDB/pdbs/$CONDITION/$PDB.pdb"
[ -f "$IN_PDB" ] || { echo "no prep pdb for $PDB/$CONDITION at $IN_PDB"; exit 1; }
OUT="$ABS/outputs/panel/$PDB/$CONDITION/esmif"; mkdir -p "$OUT/seqs"
NSEQ="${NSEQ:-10000}"; BATCH="${BATCH:-8}"; TEMP="${TEMP:-0.1}"
LOG="$OUT/run.log"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
export TORCH_HOME="$ABS/models_cache/torch_hub"
gpu_wait(){ while true; do f="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)"; if [[ "$f" =~ ^[0-9]+$ ]] && [ "$f" -ge "$1" ]; then sleep "0.$((RANDOM % 9 + 1))"; f2="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)"; [[ "$f2" =~ ^[0-9]+$ ]] && [ "$f2" -ge "$1" ] && break; fi; sleep $((15 + RANDOM % 15)); done; }
exec 9>"$OUT/runner.lock"; flock -n 9 || { echo "another runner holds lock — exit"; exit 0; }
echo $$ >"$OUT/runner.pid"; trap 'rm -f "$OUT/runner.pid"' EXIT

count_seqs(){ [ -f "$OUT/seqs/$PDB.fa" ] && grep -c "^>" "$OUT/seqs/$PDB.fa" 2>/dev/null || echo 0; }
have=$(count_seqs); have=${have:-0}
if [ "$have" -ge "$NSEQ" ]; then
  echo "[$(date '+%F %T')] already at $have/$NSEQ -- skip" | tee -a "$LOG"
  touch "$OUT/COMPLETE"; exit 0
fi

echo "[$(date '+%F %T')] === panel esmif: $PDB/$CONDITION x $NSEQ seqs, T=$TEMP, chains=${CHAINS[$CONDITION]} ===" | tee -a "$LOG"
gpu_wait 8000

$PY py/esmif_sample_panel.py \
    --pdb "$IN_PDB" \
    --chains ${CHAINS[$CONDITION]} \
    --target-chain C \
    --num-seqs "$NSEQ" \
    --batch-size "$BATCH" \
    --temperature "$TEMP" \
    --out "$OUT/seqs/$PDB.fa" >>"$LOG" 2>&1

got=$(count_seqs)
echo "[$(date '+%F %T')] panel esmif $PDB/$CONDITION done: $got/$NSEQ seqs" | tee -a "$LOG"
[ "${got:-0}" -ge "$NSEQ" ] && touch "$OUT/COMPLETE" && echo "[$(date '+%F %T')] === COMPLETE ===" | tee -a "$LOG"
