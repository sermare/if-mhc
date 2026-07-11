#!/usr/bin/env bash
# Grind worker: cycles its slice of combos, generating de-novo peptides with RFdiffusion
# until the 8h DEADLINE. Resumable (unique timestamped prefixes). Args: <wid> <nworkers>

## Other peptides in the paper: strong = ['ELAGIGILTV', 'SMLGIGIVPV', 'NMGGLGIMPV', 'NLSNLGILV', 'ILEDRGFNQV' , 'LMFDRGMSLL', 'MMWDRGLGMM']
# weaker = ['MMWDRGMGLL', 'SMAGIGIVDV', 'IMEDVGWLNV', 'SMAGIGIVDV']

set -uo pipefail
cd /home/ubuntu/if-mhc
ABS=/home/ubuntu/if-mhc
WID=${1:-0}; NW=${2:-2}
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2
OUT=outputs/grind; LOG=$OUT/gen.log
gpu_wait(){ while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -lt "$1" ]; do sleep 15; done; }
conda activate SE3nv
export LD_LIBRARY_PATH="$(cat $ABS/RFdiffusion/SE3NV_LDPATH.txt 2>/dev/null):${LD_LIBRARY_PATH:-}"
mapfile -t COMBOS < "$OUT/combos.tsv"
N=${#COMBOS[@]}
cd "$ABS/RFdiffusion"
i=$WID
while [ "$(date +%s)" -lt "$(cat $ABS/$OUT/DEADLINE)" ]; do
  line="${COMBOS[$((i % N))]}"; i=$((i+NW))
  IFS=$'\t' read -r pid cond L contig hot <<< "$line"
  stamp=$(date +%s)
  gpu_wait 4500
  python run_inference.py inference.input_pdb="$ABS/inputs/focus_6am/${pid}_trim.pdb" \
     "contigmap.contigs=[$contig]" "ppi.hotspot_res=[$hot]" inference.num_designs=2 diffuser.T=30 \
     inference.ckpt_override_path=models/Complex_base_ckpt.pt \
     inference.output_prefix="$ABS/$OUT/pdb/${pid}_${cond}_L${L}_w${WID}_${stamp}" >>"$ABS/$LOG" 2>&1 || true
  echo "[$(date -u)] w$WID $pid/$cond/L$L  total_bb=$(ls $ABS/$OUT/pdb/*.pdb 2>/dev/null|wc -l)" >> "$ABS/$OUT/TIMELINE.log"
done
touch "$ABS/$OUT/worker${WID}.DONE"
echo "[$(date -u)] worker $WID reached 8h deadline, stopped" >> "$ABS/$OUT/TIMELINE.log"
