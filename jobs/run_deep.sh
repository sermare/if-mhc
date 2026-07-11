#!/usr/bin/env bash
# DEEP campaign — extend the best-of-k sampling curve on the top conditionings.
# Waits for the fair-N run to finish (GPU is compute-bound; no parallel speedup), then tops each
# cell in jobs/deep_spec.tsv (pid, cond, ABSOLUTE target) up to target. Order = top-to-bottom, so
# 6AM5 L4_expanded (+100, the "hit" conditioning) fills FIRST. Contig/hotspots looked up from
# jobs/all_conditionings.tsv. Fully de-novo (10-10), full T=30, withTCR. Resumable + idempotent
# (absolute targets). Re-score anytime: bash jobs/evaluate.sh
set -uo pipefail
ABS=/home/ubuntu/if-mhc; cd "$ABS"
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
OUT="$ABS/outputs/deep"; mkdir -p "$OUT"
SPEC="$ABS/jobs/deep_spec.tsv"; TABLE="$ABS/jobs/all_conditionings.tsv"; HOURS="${HOURS:-20}"
LOG="$OUT/run.log"
if [ -f "$OUT/deadline" ]; then DEADLINE=$(cat "$OUT/deadline"); else DEADLINE=$(( $(date +%s)+HOURS*3600 )); echo "$DEADLINE">"$OUT/deadline"; fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2
conda activate SE3nv
export LD_LIBRARY_PATH="$(cat "$ABS/RFdiffusion/SE3NV_LDPATH.txt" 2>/dev/null):${LD_LIBRARY_PATH:-}"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
exec 9>"$OUT/runner.lock"; flock -n 9 || { log "another deep runner holds lock — exit"; exit 0; }
echo $$ >"$OUT/runner.pid"; trap 'rm -f "$OUT/runner.pid"' EXIT

# ---- wait for fair-N to finish so we don't split the GPU ----
fairn_busy(){ local p; p=$(cat "$ABS/outputs/fairn/runner.pid" 2>/dev/null||true); [ -n "$p" ] && kill -0 "$p" 2>/dev/null && [ ! -f "$ABS/outputs/fairn/COMPLETE" ]; }
while fairn_busy; do
  now=$(date +%s); [ "$now" -ge "$DEADLINE" ] && { log "deadline before fairn done"; exit 0; }
  log "waiting for fair-N to finish before deep run..."; sleep 60
done
log "=== fair-N clear; deep start deadline=$(date -d @"$DEADLINE" '+%F %T') ==="

count_cell(){ ls "$ABS/$1/${2}_${3}"_*.pdb 2>/dev/null | grep -vc _split || true; }
cd "$ABS/RFdiffusion"
while IFS=$'\t' read -r pid cond target; do
  [ -z "${pid:-}" ] && continue
  row=$(awk -F'\t' -v p="$pid" -v c="$cond" '$1==p && $2==c {print; exit}' "$TABLE")
  [ -z "$row" ] && { log "SKIP $pid $cond (not in table)"; continue; }
  IFS=$'\t' read -r _p _c style contig hot <<< "$row"
  case "$style" in maxcond) od="outputs/rfd_maxcond/pdb";; *) od="outputs/promising/pdb";; esac
  mkdir -p "$ABS/$od"
  log ">>> cell $pid $cond target=$target style=$style dir=$od"
  while :; do
    now=$(date +%s); [ "$now" -ge "$DEADLINE" ] && { log deadline; exit 0; }
    have=$(count_cell "$od" "$pid" "$cond"); have=${have:-0}
    [ "$have" -ge "$target" ] && { log "  $pid $cond reached $have/$target"; break; }
    if [ "$style" = maxcond ]; then prefix="$ABS/$od/${pid}_${cond}"; start="$have"
    else stamp=$(date +%s%N | tail -c 8); prefix="$ABS/$od/${pid}_${cond}_wG${stamp}"; start=0; fi
    log "  $pid $cond $((have+1))/$target -> $(basename "$prefix")"
    timeout 1200 python run_inference.py \
      inference.input_pdb="$ABS/inputs/focus_6am/${pid}_trim.pdb" \
      "contigmap.contigs=[$contig]" "ppi.hotspot_res=[$hot]" \
      inference.num_designs=1 inference.design_startnum="$start" diffuser.T=30 \
      inference.ckpt_override_path=models/Complex_base_ckpt.pt \
      inference.output_prefix="$prefix" >>"$LOG" 2>&1 || log "  WARN $pid $cond failed"
  done
done < "$SPEC"
log "=== deep ALL cells at target ==="; touch "$OUT/COMPLETE"
