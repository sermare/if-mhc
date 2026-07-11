#!/usr/bin/env bash
# FAIR-N equalization: the 2.06 A "crossing" hit (promising/6AM5/L4_expanded, n=26) was shown to be a
# best-of-k SAMPLING draw, not a property of the conditioning. To test that fairly, bring every OTHER
# "best" de-novo conditioning up to the SAME N=26 draws, so best-of-N is comparable across conditionings.
# Fully de-novo peptide (10-10, nothing templated), full T=30, withTCR context. Resumable, round-robin.
set -uo pipefail
cd /home/ubuntu/if-mhc; ABS=/home/ubuntu/if-mhc
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
OUT="$ABS/outputs/fairn"; mkdir -p "$OUT"
SPEC="$ABS/jobs/fairn_spec.tsv"; TARGET="${TARGET:-26}"; HOURS="${HOURS:-8}"
LOG="$OUT/run.log"
if [ -f "$OUT/deadline" ]; then DEADLINE=$(cat "$OUT/deadline"); else DEADLINE=$(( $(date +%s)+HOURS*3600 )); echo "$DEADLINE">"$OUT/deadline"; fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2
conda activate SE3nv
export LD_LIBRARY_PATH="$(cat "$ABS/RFdiffusion/SE3NV_LDPATH.txt" 2>/dev/null):${LD_LIBRARY_PATH:-}"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
# count existing designs already present in the target cell (across ALL campaigns that share this cond)
count_cell(){ # $1=pid $2=cond $3=outdir
  local pid="$1" cond="$2" od="$3" c=0
  c=$(ls "$ABS/$od/${pid}_${cond}"_*.pdb 2>/dev/null | grep -vc _split || true)
  echo "${c:-0}"
}
exec 9>"$OUT/runner.lock"; flock -n 9 || { log "another runner holds lock — exit"; exit 0; }
echo $$ >"$OUT/runner.pid"; trap 'rm -f "$OUT/runner.pid"' EXIT
cd "$ABS/RFdiffusion"
log "=== fairn start TARGET=$TARGET/cell deadline=$(date -d @"$DEADLINE" '+%F %T') ==="
progress=1
while [ "$progress" = 1 ]; do
  progress=0
  while IFS=$'\t' read -r pid cond outdir contig hot; do
    [ -z "${pid:-}" ] && continue
    now=$(date +%s); [ "$now" -ge "$DEADLINE" ] && { log deadline; exit 0; }
    have=$(count_cell "$pid" "$cond" "$outdir"); have=${have:-0}
    [ "$have" -ge "$TARGET" ] && continue
    progress=1
    # naming: maxcond uses sequential {pid}_{cond}_{idx}; promising uses {pid}_{cond}_wF_{stamp}
    if [[ "$outdir" == *rfd_maxcond* ]]; then
      prefix="$ABS/$outdir/${pid}_${cond}"; start="$have"
    else
      stamp=$(date +%s%N | tail -c 8); prefix="$ABS/$outdir/${pid}_${cond}_wF_${stamp}"; start=0
    fi
    log "cell=${pid}_${cond} have=$have -> +1 (prefix $(basename "$prefix"))"
    args=( inference.input_pdb="$ABS/inputs/focus_6am/${pid}_trim.pdb" "contigmap.contigs=[$contig]"
      inference.num_designs=1 inference.design_startnum="$start" diffuser.T=30
      inference.ckpt_override_path=models/Complex_base_ckpt.pt inference.output_prefix="$prefix"
      "ppi.hotspot_res=[$hot]" )
    timeout 1200 python run_inference.py "${args[@]}" >>"$LOG" 2>&1 || log "  WARN ${pid}_${cond} failed"
  done < "$SPEC"
done
log "=== fairn ALL cells at TARGET=$TARGET ==="; touch "$OUT/COMPLETE"
