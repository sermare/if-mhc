#!/usr/bin/env bash
# RFD1 de-novo 10-mer campaign — round-robin over all cells in rfd_denovo30_spec.tsv.
# Each cell = (crystal x context {withTCR,noTCR} x conditioning). Peptide chain fully removed (contig
# ...10-10), full diffuser.T=30, RFD1 (Complex_base_ckpt). Breadth-first: advance every cell a small
# BATCH at a time, cap at TARGET/cell, stop at DEADLINE. Resumable (counts existing pdbs, uses
# inference.design_startnum so new designs never overwrite). Env: SE3nv + SE3NV_LDPATH.
set -uo pipefail
cd /home/ubuntu/if-mhc
ABS=/home/ubuntu/if-mhc
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
OUT="$ABS/outputs/rfd_denovo30"; PDB="$OUT/pdb"; mkdir -p "$PDB"
SPEC="$ABS/jobs/rfd_denovo30_spec.tsv"
TARGET="${TARGET:-30}"; BATCH="${BATCH:-2}"; HOURS="${HOURS:-10}"
LOG="$OUT/run.log"
# DEADLINE persists across supervisor relaunches
if [ -f "$OUT/deadline" ]; then DEADLINE=$(cat "$OUT/deadline"); else DEADLINE=$(( $(date +%s) + HOURS*3600 )); echo "$DEADLINE" > "$OUT/deadline"; fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2
conda activate SE3nv
export LD_LIBRARY_PATH="$(cat "$ABS/RFdiffusion/SE3NV_LDPATH.txt" 2>/dev/null):${LD_LIBRARY_PATH:-}"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
# match ONLY {cell}_<number>.pdb (RFd output index) so prefix-sharing cells (mhc vs mhc_tcr1) don't cross-count
count_cell(){ ls "$PDB/$1"_[0-9]*.pdb 2>/dev/null | grep -vc _split || true; }
# single-instance guard + liveness marker (supervisor reads this PID, not a fragile name match)
exec 9>"$OUT/runner.lock"
flock -n 9 || { log "another runner holds the lock — exiting"; exit 0; }
echo $$ > "$OUT/runner.pid"
trap 'rm -f "$OUT/runner.pid"' EXIT

log "=== rfd_denovo30 start | TARGET=$TARGET/cell BATCH=$BATCH deadline=$(date -d @"$DEADLINE" '+%F %T') ==="
cd "$ABS/RFdiffusion"
pass=0
while :; do
  now=$(date +%s); [ "$now" -ge "$DEADLINE" ] && { log "deadline reached — stopping"; break; }
  made=0; pending=0
  while IFS=$'\t' read -r cell pid ctx cond contig hot; do
    [ -z "${cell:-}" ] && continue
    now=$(date +%s); [ "$now" -ge "$DEADLINE" ] && break
    have=$(count_cell "$cell"); have=${have:-0}
    [ "$have" -ge "$TARGET" ] && continue
    pending=$((pending+1))
    b=$(( TARGET - have )); [ "$b" -gt "$BATCH" ] && b=$BATCH
    args=( inference.input_pdb="$ABS/inputs/focus_6am/${pid}_trim.pdb"
           "contigmap.contigs=[$contig]"
           inference.num_designs="$b" inference.design_startnum="$have" diffuser.T=30
           inference.ckpt_override_path=models/Complex_base_ckpt.pt
           inference.output_prefix="$PDB/${cell}" )
    # only add hotspots if non-empty (empty => free / unconstrained peptide)
    hot_trim="$(echo "$hot" | tr -d '[:space:]')"
    [ -n "$hot_trim" ] && args+=( "ppi.hotspot_res=[$hot_trim]" )
    log "cell=$cell have=$have -> +$b (pass $pass)"
    # timeout guard: withTCR design ~4 min, noTCR ~2.5 min; 15 min/design ceiling so a hang can't idle GPU
    timeout $(( 900 * b )) python run_inference.py "${args[@]}" >>"$LOG" 2>&1 && made=$((made+b)) || log "  WARN: $cell run failed/timeout (rc=$?)"
  done < "$SPEC"
  log "pass $pass done: +$made designs this pass, $pending cells still under target"
  [ "$pending" -eq 0 ] && { log "ALL cells at TARGET=$TARGET — campaign complete"; touch "$OUT/COMPLETE"; break; }
  pass=$((pass+1))
done
log "=== rfd_denovo30 exiting (pass $pass) ==="
