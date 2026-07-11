#!/usr/bin/env bash
# N-anchor sliding-window arm (motif SCAFFOLD, full diffusion -- NOT partial).
# Hold an N-terminal peptide window fixed, BUILD the rest of the peptide de novo, guided by the
# A/B-pocket N-anchor hotspot. Windows slide over the first 2-3 positions: P1-2 / P2-3 / P1-3.
# Validated: N-window-only scaffold stays in-groove (smoke toGIG=2.7). Output layout differs from
# the partial arms: peptide = chain A, receptor = chain B.
#
# Required env: OUT ; Optional: HOT (A-pocket set), WINDOWS, NDES_CRYSTAL(4), NDES_RELAXED(1), ONLY, GPU_FREE_MIN
set -uo pipefail
ABS=/home/ubuntu/if-mhc
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
conda activate SE3nv
export LD_LIBRARY_PATH="$(cat $ABS/RFdiffusion/SE3NV_LDPATH.txt):${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2

MAN=$ABS/outputs/rfdiff_q30/seeds_manifest.tsv
OUT="${OUT:-$ABS/outputs/rfdiff_q30_nslide}"
HOT="${HOT:-A7,A63,A66,A70,A99,A159,A171}"
WINDOWS="${WINDOWS:-W12 W23 W13}"
NDES_CRYSTAL="${NDES_CRYSTAL:-4}"; NDES_RELAXED="${NDES_RELAXED:-1}"; ONLY="${ONLY:-}"
GPU_FREE_MIN="${GPU_FREE_MIN:-11000}"
CKPT=models/Complex_base_ckpt.pt
mkdir -p "$OUT/logs"
# peptide contig per window: hold C<window> native, build the rest (10-mer total)
declare -A PEP=( [W12]="C1-2/8-8" [W23]="1-1/C2-3/7-7" [W13]="C1-3/7-7" )
gpu_wait(){ while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -lt "$GPU_FREE_MIN" ]; do sleep 15; done; }

worker(){
  local wpid=$1
  local log="$OUT/logs/${wpid}.log"
  local hot="$HOT"                                   # epitope-specific override if provided
  [ "$wpid" = "6AM5" ] && hot="${GIG_HOT:-$HOT}"
  [ "$wpid" = "6AMU" ] && hot="${DRG_HOT:-$HOT}"
  echo "[$(date)] nslide worker=$wpid hot=[$hot] windows=[$WINDOWS]" >"$log"
  cd "$ABS/RFdiffusion"
  tail -n +2 "$MAN" | while IFS=$'\t' read -r name pid input contig provide _hot; do
    [ "$pid" = "$wpid" ] || continue
    [ -n "$ONLY" ] && [[ "$name" != *"$ONLY"* ]] && continue
    local ndes=$NDES_RELAXED; [[ "$name" == *_trim ]] && ndes=$NDES_CRYSTAL
    for W in $WINDOWS; do
      local pepc=${PEP[$W]}
      local newcontig=$(echo "$contig" | sed -E "s#C[0-9]+-[0-9]+#${pepc}#")   # swap C-segment
      local pref="$OUT/${name}_${W}"
      ls "${pref}_$((ndes-1)).pdb" >/dev/null 2>&1 && { echo "[$(date)] $name $W done, skip" >>"$log"; continue; }
      gpu_wait
      echo "[$(date)] $name $W ndes=$ndes contig=[$newcontig]" >>"$log"
      python run_inference.py \
        inference.input_pdb="$input" \
        inference.output_prefix="$pref" \
        inference.num_designs="$ndes" \
        "contigmap.contigs=[$newcontig]" \
        "ppi.hotspot_res=[$hot]" \
        inference.ckpt_override_path="$CKPT" >>"$log" 2>&1 || echo "[$(date)] ERR $name $W" >>"$log"
    done
  done
  echo "[$(date)] DONE $wpid" >>"$log"; touch "$OUT/logs/${wpid}.DONE"
}
rm -f "$OUT/logs/6AM5.DONE" "$OUT/logs/6AMU.DONE"
worker 6AM5 & worker 6AMU & wait
echo "[$(date)] nslide complete: $(ls $OUT/*_W*_*.pdb 2>/dev/null | grep -v _split | wc -l) designs" | tee "$OUT/RFDIFF_COMPLETE"
