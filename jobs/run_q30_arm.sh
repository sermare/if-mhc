#!/usr/bin/env bash
# Generalized Q30 partial-diffusion arm: epitope-specific hotspots + own output dir.
# Reuses outputs/rfdiff_q30/seeds_manifest.tsv for per-seed contig/provide_seq; OVERRIDES the
# hotspot column with GIG_HOT (6AM5 seeds) / DRG_HOT (6AMU seeds). Two concurrent epitope workers.
#
# Required env: OUT, GIG_HOT, DRG_HOT
# Optional:     TS ("5 10 20 40"), NDES_CRYSTAL (6), NDES_RELAXED (1), ONLY (name filter)
set -uo pipefail
ABS=/home/ubuntu/if-mhc
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
conda activate SE3nv
export LD_LIBRARY_PATH="$(cat $ABS/RFdiffusion/SE3NV_LDPATH.txt):${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2

MAN=$ABS/outputs/rfdiff_q30/seeds_manifest.tsv
OUT="${OUT:?set OUT}"; GIG_HOT="${GIG_HOT:?set GIG_HOT}"; DRG_HOT="${DRG_HOT:?set DRG_HOT}"
TS="${TS:-5 10 20 40}"; NDES_CRYSTAL="${NDES_CRYSTAL:-6}"; NDES_RELAXED="${NDES_RELAXED:-1}"; ONLY="${ONLY:-}"
GPU_FREE_MIN="${GPU_FREE_MIN:-6000}"   # min free MiB to admit a design; raise to cap cross-arm concurrency
CKPT=models/Complex_base_ckpt.pt
mkdir -p "$OUT/logs"
gpu_wait(){ while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -lt "$GPU_FREE_MIN" ]; do sleep 15; done; }

worker(){
  local wpid=$1
  local log="$OUT/logs/${wpid}.log"
  local hot=$GIG_HOT; [ "$wpid" = "6AMU" ] && hot=$DRG_HOT
  echo "[$(date)] arm=$OUT worker=$wpid hot=[$hot] TS=[$TS]" >"$log"
  cd "$ABS/RFdiffusion"
  tail -n +2 "$MAN" | while IFS=$'\t' read -r name pid input contig provide _hot; do
    [ "$pid" = "$wpid" ] || continue
    [ -n "$ONLY" ] && [[ "$name" != *"$ONLY"* ]] && continue
    local ndes=$NDES_RELAXED; [[ "$name" == *_trim ]] && ndes=$NDES_CRYSTAL
    for T in $TS; do
      local pref="$OUT/${name}_T${T}"
      ls "${pref}_$((ndes-1)).pdb" >/dev/null 2>&1 && { echo "[$(date)] $name T$T done, skip" >>"$log"; continue; }
      gpu_wait 6000
      echo "[$(date)] $name partial_T=$T ndes=$ndes${INPAINT_STR:+ inpaint_str=[$INPAINT_STR]}" >>"$log"
      python run_inference.py \
        inference.input_pdb="$input" \
        inference.output_prefix="$pref" \
        inference.num_designs="$ndes" \
        "contigmap.contigs=[$contig]" \
        "contigmap.provide_seq=[$provide]" \
        ${INPAINT_STR:+"+contigmap.inpaint_str=[$INPAINT_STR]"} \
        diffuser.partial_T="$T" \
        "ppi.hotspot_res=[$hot]" \
        inference.ckpt_override_path="$CKPT" >>"$log" 2>&1 \
        || echo "[$(date)] ERR $name T$T" >>"$log"
    done
  done
  echo "[$(date)] DONE worker $wpid" >>"$log"; touch "$OUT/logs/${wpid}.DONE"
}

rm -f "$OUT/logs/6AM5.DONE" "$OUT/logs/6AMU.DONE"
worker 6AM5 & worker 6AMU & wait
n=$(ls "$OUT"/*_T*.pdb 2>/dev/null | grep -v _split | wc -l)
echo "[$(date)] arm complete: $n designs in $OUT" | tee "$OUT/RFDIFF_COMPLETE"
