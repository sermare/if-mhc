#!/usr/bin/env bash
# ALLCOND300 campaign — top up EVERY conditioning in jobs/all_conditionings.tsv by +300 designs each,
# in BATCHes of 30 per visit: cell 1 gets 30, then cell 2 gets 30, ... round-robin across all 16 cells
# (5 contact-ladder levels x2 crystals + 3 maxcond levels x2 crystals) until each reaches target.
# Absolute per-cell targets were snapshotted at launch into jobs/allcond300_spec.tsv (have+300), so this
# is resumable/idempotent: re-running never overshoots even if restarted mid-campaign.
# Fully de-novo peptide (10-10, nothing templated), full T=30, withTCR context, MAX hotspots per cell.
set -uo pipefail
cd /home/ubuntu/if-mhc; ABS=/home/ubuntu/if-mhc
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
OUT="$ABS/outputs/allcond300"; mkdir -p "$OUT"
SPEC="$ABS/jobs/allcond300_spec.tsv"; TABLE="$ABS/jobs/all_conditionings.tsv"; HOURS="${HOURS:-72}"
BATCH="${BATCH:-30}"
LOG="$OUT/run.log"
if [ -f "$OUT/deadline" ]; then DEADLINE=$(cat "$OUT/deadline"); else DEADLINE=$(( $(date +%s)+HOURS*3600 )); echo "$DEADLINE">"$OUT/deadline"; fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2
conda activate SE3nv
export LD_LIBRARY_PATH="$(cat "$ABS/RFdiffusion/SE3NV_LDPATH.txt" 2>/dev/null):${LD_LIBRARY_PATH:-}"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
exec 9>"$OUT/runner.lock"; flock -n 9 || { log "another allcond300 runner holds lock — exit"; exit 0; }
echo $$ >"$OUT/runner.pid"; trap 'rm -f "$OUT/runner.pid"' EXIT

count_cell(){ local dir="$1" pid="$2" cond="$3"; ls "$ABS/$dir/${pid}_${cond}"_*.pdb 2>/dev/null | grep -vc _split || true; }
cd "$ABS/RFdiffusion"
log "=== allcond300 start (16 cells, +300 each) deadline=$(date -d @"$DEADLINE" '+%F %T') ==="
progress=1
while [ "$progress" = 1 ]; do
  progress=0
  while IFS=$'\t' read -r pid cond target style; do
    [ -z "${pid:-}" ] && continue
    now=$(date +%s); [ "$now" -ge "$DEADLINE" ] && { log deadline; exit 0; }
    row=$(awk -F'\t' -v p="$pid" -v c="$cond" '$1==p && $2==c {print; exit}' "$TABLE")
    [ -z "$row" ] && { log "SKIP $pid $cond (not in table)"; continue; }
    IFS=$'\t' read -r _p _c _style contig hot <<< "$row"
    case "$style" in maxcond) od="outputs/rfd_maxcond/pdb";; *) od="outputs/promising/pdb";; esac
    mkdir -p "$ABS/$od"
    have=$(count_cell "$od" "$pid" "$cond"); have=${have:-0}
    [ "$have" -ge "$target" ] && continue
    progress=1
    batch_end=$(( have+BATCH < target ? have+BATCH : target ))
    log "cell=${pid}_${cond} have=$have/$target -> batch to $batch_end"
    while [ "$have" -lt "$batch_end" ]; do
      now=$(date +%s); [ "$now" -ge "$DEADLINE" ] && { log deadline; exit 0; }
      if [ "$style" = maxcond ]; then prefix="$ABS/$od/${pid}_${cond}"; start="$have"
      else stamp=$(date +%s%N | tail -c 8); prefix="$ABS/$od/${pid}_${cond}_wA_${stamp}"; start=0; fi
      args=( inference.input_pdb="$ABS/inputs/focus_6am/${pid}_trim.pdb" "contigmap.contigs=[$contig]"
        inference.num_designs=1 inference.design_startnum="$start" diffuser.T=30
        inference.ckpt_override_path=models/Complex_base_ckpt.pt inference.output_prefix="$prefix" )
      [ -n "$(echo "$hot"|tr -d '[:space:]')" ] && args+=( "ppi.hotspot_res=[$hot]" )
      timeout 1200 python run_inference.py "${args[@]}" >>"$LOG" 2>&1 || log "  WARN ${pid}_${cond} failed"
      have=$(count_cell "$od" "$pid" "$cond"); have=${have:-0}
    done
  done < "$SPEC"
done
log "=== allcond300 ALL cells at target ==="; touch "$OUT/COMPLETE"
