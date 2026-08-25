#!/usr/bin/env bash
# 3HG1 = MEL5 TCR (MART-1-specific) / HLA-A2 / ELAGIGILTV, the one structure in
# inputs/pmhc_tcr_dataset found to overlap an Adimab TCR clone (CAB60174_G01, CDR3a+CDR3b match --
# see outputs/analysis/adimab_tcr_overlap.csv). Same 5-chain convention as the native 6AM5/6AMU
# campaigns (A=MHC, B=b2m, C=peptide designed, D=TCRa, E=TCRb fixed). 40K sequences EACH, vanilla
# (v_48_020) and noMHC (proteinmpnn_nomhc), T=0.1 (matching the project's primary recovery-campaign
# temperature, not the T=0.3 exploration run). Originally scoped to 100K/target -- reduced to 40K
# (plenty for recovery/distribution stats). OUT dir keeps its original name.
#
# chain_id.jsonl gotcha: ProteinMPNN/helper_scripts/assign_fixed_chains.py's --chain_list argument
# despite the script's name is the DESIGNED chain list, not the fixed one (see its own docstring
# output example: {"5TTA": [["A"], ["B"]]} = [designed, fixed]). An earlier run of this campaign
# passed --chain_list "A B D E" intending to fix those chains, which actually designed the whole
# MHC+TCR framework and fixed the peptide instead -- caught via the reference-sequence line in the
# output fasta not matching ELAGIGILTV, and via load_fasta silently producing 0 rows since the
# "peptide" records were 824 residues, not 10. That entire first attempt (44,127 vanilla + partial
# noMHC sequences) was discarded; chain_id.jsonl here is corrected to
# {"3HG1": [["C"], ["A", "B", "D", "E"]]} (peptide designed, complex fixed) and verified with a
# 4-sequence test batch before relaunching. Resumable per weights via flock + per-target skip, same
# convention as every other campaign script.
set -uo pipefail
cd /home/ubuntu/if-mhc; ABS=/home/ubuntu/if-mhc
PY=/home/ubuntu/miniforge3/envs/esmcba/bin/python
OUT="$ABS/outputs/mpnn_3hg1_100k"; mkdir -p "$OUT/seqs"
NSEQ="${NSEQ:-40000}"; BATCH="${BATCH:-8}"; TEMP="${TEMP:-0.1}"
LOG="$OUT/run.log"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2
gpu_wait(){ while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -lt "$1" ]; do sleep 20; done; }
exec 9>"$OUT/runner.lock"; flock -n 9 || { echo "another runner holds lock — exit"; exit 0; }
echo $$ >"$OUT/runner.pid"; trap 'rm -f "$OUT/runner.pid"' EXIT

echo "[$(date '+%F %T')] === 3hg1_100k: MEL5 TCR/MART-1 x $NSEQ seqs, T=$TEMP, vanilla+noMHC ===" | tee -a "$LOG"

count_seqs(){ [ -f "$OUT/seqs/$1.fa" ] && grep -c "^>" "$OUT/seqs/$1.fa" 2>/dev/null || echo 0; }

declare -A WEIGHTS_DIR=( [vanilla]="" [nomhc]="ProteinMPNN/nomhc_model_weights/" )
declare -A MODEL_NAME=( [vanilla]="v_48_020" [nomhc]="proteinmpnn_nomhc" )

for weights in vanilla nomhc; do
  tag="${weights}_3HG1"
  # leftover from an interrupted run (killed before the post-run mv) -- recover before checking progress
  if [ -f "$OUT/seqs/3HG1.fa" ]; then
    mv -f "$OUT/seqs/3HG1.fa" "$OUT/seqs/$tag.fa"
    echo "[$(date '+%F %T')] recovered interrupted-run leftover seqs/3HG1.fa -> seqs/$tag.fa" | tee -a "$LOG"
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
  mv -f "$OUT/seqs/3HG1.fa" "$OUT/seqs/$tag.fa" 2>/dev/null
  got=$(count_seqs "$tag")
  echo "[$(date '+%F %T')] target=$tag done: $got seqs" | tee -a "$LOG"
  [ "$got" -lt "$NSEQ" ] && echo "[$(date '+%F %T')] WARN $tag short of target ($got/$NSEQ) -- will retry next invocation" | tee -a "$LOG"
done

all_done=1
for weights in vanilla nomhc; do
  have=$(count_seqs "${weights}_3HG1"); [ "${have:-0}" -lt "$NSEQ" ] && all_done=0
done
if [ "$all_done" = 1 ]; then
  echo "[$(date '+%F %T')] === 3hg1_100k ALL TARGETS REACHED $NSEQ ===" | tee -a "$LOG"
  touch "$OUT/COMPLETE"
else
  echo "[$(date '+%F %T')] === pass complete, some targets still short -- supervisor will retry ===" | tee -a "$LOG"
fi
