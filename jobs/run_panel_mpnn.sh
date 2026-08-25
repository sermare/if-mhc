#!/usr/bin/env bash
# ProteinMPNN (vanilla + noMHC) on one panel structure/condition, using the pre-filtered/renamed
# A=MHC,B=b2m,C=peptide,D=TCRa,E=TCRb PDBs + parsed.jsonl/chain_id.jsonl built by
# py/prep_panel_structures.py (outputs/panel_prep/{PDB}/{full,mhconly}_{parsed,chain_id}.jsonl).
# Usage: PDB=1QSF CONDITION=full|mhconly bash jobs/run_panel_mpnn.sh
set -uo pipefail
cd /home/ubuntu/if-mhc; ABS=/home/ubuntu/if-mhc
PY=/home/ubuntu/miniforge3/envs/esmcba/bin/python
PDB="${PDB:?set PDB=<panel pdb id>}"
CONDITION="${CONDITION:?set CONDITION=full|mhconly}"
PREP="$ABS/outputs/panel_prep/$PDB"
[ -f "$PREP/${CONDITION}_parsed.jsonl" ] || { echo "no prep for $PDB/$CONDITION"; exit 1; }
OUT="$ABS/outputs/panel/$PDB/$CONDITION/mpnn"; mkdir -p "$OUT/seqs"
NSEQ="${NSEQ:-10000}"; BATCH="${BATCH:-8}"; TEMP="${TEMP:-0.1}"
LOG="$OUT/run.log"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
gpu_wait(){ while true; do f="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)"; if [[ "$f" =~ ^[0-9]+$ ]] && [ "$f" -ge "$1" ]; then sleep "0.$((RANDOM % 9 + 1))"; f2="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)"; [[ "$f2" =~ ^[0-9]+$ ]] && [ "$f2" -ge "$1" ] && break; fi; sleep $((15 + RANDOM % 15)); done; }
exec 9>"$OUT/runner.lock"; flock -n 9 || { echo "another runner holds lock — exit"; exit 0; }
echo $$ >"$OUT/runner.pid"; trap 'rm -f "$OUT/runner.pid"' EXIT

echo "[$(date '+%F %T')] === panel mpnn: $PDB/$CONDITION x $NSEQ seqs, T=$TEMP, vanilla+noMHC ===" | tee -a "$LOG"

count_seqs(){ [ -f "$OUT/seqs/$1.fa" ] && grep -c "^>" "$OUT/seqs/$1.fa" 2>/dev/null || echo 0; }

declare -A WEIGHTS_DIR=( [vanilla]="" [nomhc]="ProteinMPNN/nomhc_model_weights/" )
declare -A MODEL_NAME=( [vanilla]="v_48_020" [nomhc]="proteinmpnn_nomhc" )

for weights in vanilla nomhc; do
  tag="${weights}_${PDB}"
  if [ -f "$OUT/seqs/$PDB.fa" ]; then
    mv -f "$OUT/seqs/$PDB.fa" "$OUT/seqs/$tag.fa"
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
     --jsonl_path "$PREP/${CONDITION}_parsed.jsonl" \
     --chain_id_jsonl "$PREP/${CONDITION}_chain_id.jsonl" \
     --out_folder "$OUT" \
     --num_seq_per_target "$NSEQ" \
     --batch_size "$BATCH" \
     --sampling_temp "$TEMP" \
     --seed 37 \
     --model_name "${MODEL_NAME[$weights]}" \
     "${extra_args[@]}" >>"$LOG" 2>&1
  mv -f "$OUT/seqs/$PDB.fa" "$OUT/seqs/$tag.fa" 2>/dev/null
  got=$(count_seqs "$tag")
  echo "[$(date '+%F %T')] target=$tag done: $got seqs" | tee -a "$LOG"
done

all_done=1
for weights in vanilla nomhc; do
  have=$(count_seqs "${weights}_${PDB}"); [ "${have:-0}" -lt "$NSEQ" ] && all_done=0
done
[ "$all_done" = 1 ] && touch "$OUT/COMPLETE" && echo "[$(date '+%F %T')] === $PDB/$CONDITION mpnn COMPLETE ===" | tee -a "$LOG"
