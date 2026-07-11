#!/usr/bin/env bash
# Scale-up campaign: generate de-novo backbones on the PROMISING contact-conditions (L5_max>L4>L3),
# 2 parallel workers, weighted toward L5_max, until TARGET total or DEADLINE. Resumable (timestamps).
set -uo pipefail
cd /home/ubuntu/if-mhc; ABS=/home/ubuntu/if-mhc
WID=${1:-0}; NW=${2:-2}; CRYST=${3:-}   # optional 3rd arg: restrict to one crystal (6AMU|6AM5)
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh; conda activate SE3nv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2
export LD_LIBRARY_PATH="$(cat $ABS/RFdiffusion/SE3NV_LDPATH.txt 2>/dev/null):${LD_LIBRARY_PATH:-}"
OUT=outputs/promising; mkdir -p "$OUT/pdb"; LOG=$OUT/gen.log
TARGET=$(cat $OUT/TARGET 2>/dev/null || echo 240)
gpu_wait(){ while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -lt "$1" ]; do sleep 15; done; }
# weighted condition list: L5_max x3, L4_expanded x2, L3_nterm_t2 x1; optionally one crystal only
declare -a JOBS
while IFS=$'\t' read -r pid name nc contig hot; do
  [ -n "$CRYST" ] && [ "$pid" != "$CRYST" ] && continue
  case "$name" in L5_max) w=3;; L4_expanded) w=2;; *) w=1;; esac
  for ((k=0;k<w;k++)); do JOBS+=("$pid	$name	$contig	$hot"); done
done < inputs/focus_6am/promising_spec.tsv
N=${#JOBS[@]}
cd "$ABS/RFdiffusion"; i=0
while [ "$(ls $ABS/$OUT/pdb/*.pdb 2>/dev/null|grep -vc _split)" -lt "$TARGET" ] && [ "$(date +%s)" -lt "$(cat $ABS/$OUT/DEADLINE)" ]; do
  IFS=$'\t' read -r pid name contig hot <<< "${JOBS[$((i % N))]}"; i=$((i+1))
  stamp=$(date +%s%N | tail -c 8)
  gpu_wait 5000
  python run_inference.py inference.input_pdb="$ABS/inputs/focus_6am/${pid}_trim.pdb" \
     "contigmap.contigs=[$contig]" "ppi.hotspot_res=[$hot]" inference.num_designs=2 diffuser.T=30 \
     inference.ckpt_override_path=models/Complex_base_ckpt.pt \
     inference.output_prefix="$ABS/$OUT/pdb/${pid}_${name}_w${WID}_${stamp}" >>"$ABS/$LOG" 2>&1 || true
  echo "[$(date -u)] w$WID $pid/$name total=$(ls $ABS/$OUT/pdb/*.pdb 2>/dev/null|grep -vc _split)" >> "$ABS/$OUT/TIMELINE.log"
done
touch "$ABS/$OUT/worker${WID}.DONE"
