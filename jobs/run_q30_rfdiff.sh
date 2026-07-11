#!/usr/bin/env bash
# Q30 + F-pocket dual-hotspot PARTIAL DIFFUSION (local RFdiffusion v1).
# Re-diffuse ONLY the peptide (C) inside the fixed complex, guided by the dual hotspot
# [D30 (Q30 N-term) + floor F-pocket A77,A84,A116,A123 (C-term)], across a partial_T sweep.
# Seeds + per-seed contig/provide_seq come from outputs/rfdiff_q30/seeds_manifest.tsv
# (py/prep_q30_seeds.py). Two workers (6AM5/GIG, 6AMU/DRG) run concurrently on the L4.
#
# Env knobs:
#   TS            partial_T levels           (default "5 10 20 40")
#   NDES_CRYSTAL  designs/T for *_trim seeds  (default 6)
#   NDES_RELAXED  designs/T for snap seeds     (default 1)
#   ONLY          substring filter on seed name (e.g. "trim" for a crystal-only smoke run)
# Smoke/validation gate:  TS=20 NDES_CRYSTAL=3 ONLY=trim bash jobs/run_q30_rfdiff.sh
set -uo pipefail
ABS=/home/ubuntu/if-mhc
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
conda activate SE3nv
export LD_LIBRARY_PATH="$(cat $ABS/RFdiffusion/SE3NV_LDPATH.txt):${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2

OUT=$ABS/outputs/rfdiff_q30
MAN=$OUT/seeds_manifest.tsv
TS="${TS:-5 10 20 40}"
NDES_CRYSTAL="${NDES_CRYSTAL:-6}"
NDES_RELAXED="${NDES_RELAXED:-1}"
ONLY="${ONLY:-}"
CKPT=models/Complex_base_ckpt.pt
mkdir -p "$OUT/logs"
gpu_wait(){ while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -lt "$1" ]; do sleep 15; done; }

worker(){
  local wpid=$1
  local log="$OUT/logs/${wpid}.log"
  echo "[$(date)] start Q30 dual-hotspot PD worker=$wpid  TS=[$TS] ndes(crystal=$NDES_CRYSTAL,relaxed=$NDES_RELAXED) only='$ONLY'" >"$log"
  cd "$ABS/RFdiffusion"
  # rows for this pid, skipping header
  tail -n +2 "$MAN" | while IFS=$'\t' read -r name pid input contig provide hot; do
    [ "$pid" = "$wpid" ] || continue
    [ -n "$ONLY" ] && [[ "$name" != *"$ONLY"* ]] && continue
    local ndes=$NDES_RELAXED; [[ "$name" == *_trim ]] && ndes=$NDES_CRYSTAL
    for T in $TS; do
      local pref="$OUT/${name}_T${T}"
      if ls "${pref}_$((ndes-1)).pdb" >/dev/null 2>&1; then
        echo "[$(date)] $name T$T done ($ndes), skip" >>"$log"; continue
      fi
      gpu_wait 6000
      echo "[$(date)] $name partial_T=$T ndes=$ndes hot=[$hot]" >>"$log"
      python run_inference.py \
        inference.input_pdb="$input" \
        inference.output_prefix="$pref" \
        inference.num_designs="$ndes" \
        "contigmap.contigs=[$contig]" \
        "contigmap.provide_seq=[$provide]" \
        diffuser.partial_T="$T" \
        "ppi.hotspot_res=[$hot]" \
        inference.ckpt_override_path="$CKPT" >>"$log" 2>&1 \
        || echo "[$(date)] ERR $name T$T (see log)" >>"$log"
    done
  done
  echo "[$(date)] DONE worker $wpid" >>"$log"; touch "$OUT/logs/${wpid}.DONE"
}

rm -f "$OUT/logs/6AM5.DONE" "$OUT/logs/6AMU.DONE"
worker 6AM5 &
worker 6AMU &
wait
n=$(ls "$OUT"/*_T*.pdb 2>/dev/null | grep -v _split | wc -l)
echo "[$(date)] Q30 PD complete: $n design pdbs in $OUT" | tee "$OUT/RFDIFF_COMPLETE"
