#!/usr/bin/env bash
# Max-interaction conditioning ladder — RFD1 de-novo 10-mer, full diffuser.T=30, withTCR scaffold.
# Hotspots = native peptide contacts (MHC+TCR), top-k per level: max(all) -> k24..k2. MAX FIRST.
# Goal: find the MINIMUM conditioning level that reproduces the SELF (cognate) peptide at full T.
# Parallelism: PAR concurrent designs per cell (L4 has headroom). Resumable via design_startnum.
set -uo pipefail
cd /home/ubuntu/if-mhc; ABS=/home/ubuntu/if-mhc
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
OUT="$ABS/outputs/rfd_maxcond"; PDB="$OUT/pdb"; mkdir -p "$PDB"
SPEC="$ABS/jobs/maxcond_spec.tsv"
TARGET="${TARGET:-20}"; PAR="${PAR:-2}"; HOURS="${HOURS:-10}"
LOG="$OUT/run.log"
if [ -f "$OUT/deadline" ]; then DEADLINE=$(cat "$OUT/deadline"); else DEADLINE=$(( $(date +%s)+HOURS*3600 )); echo "$DEADLINE">"$OUT/deadline"; fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2
conda activate SE3nv
export LD_LIBRARY_PATH="$(cat "$ABS/RFdiffusion/SE3NV_LDPATH.txt" 2>/dev/null):${LD_LIBRARY_PATH:-}"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
count_cell(){ ls "$PDB/$1"_[0-9]*.pdb 2>/dev/null | grep -vc _split || true; }
# single-instance guard + liveness pid
exec 9>"$OUT/runner.lock"; flock -n 9 || { log "another runner holds lock — exit"; exit 0; }
echo $$ >"$OUT/runner.pid"; trap 'rm -f "$OUT/runner.pid"' EXIT

# level order: MAX first, then descending; both crystals per level (interleaved)
ORDER="max k24 k18 k14 k12 k9 k6 k4 k3 k2"
run_one(){ # cell pid contig hot startnum
  local cell=$1 pid=$2 contig=$3 hot=$4 sn=$5
  local args=( inference.input_pdb="$ABS/inputs/focus_6am/${pid}_trim.pdb"
    "contigmap.contigs=[$contig]" inference.num_designs=1 inference.design_startnum="$sn" diffuser.T=30
    inference.ckpt_override_path=models/Complex_base_ckpt.pt inference.output_prefix="$PDB/${cell}" )
  [ -n "$(echo "$hot"|tr -d '[:space:]')" ] && args+=( "ppi.hotspot_res=[$hot]" )
  timeout 1200 python run_inference.py "${args[@]}" >>"$LOG" 2>&1 || log "  WARN $cell sn=$sn failed"
}
cd "$ABS/RFdiffusion"
log "=== maxcond start TARGET=$TARGET PAR=$PAR deadline=$(date -d @"$DEADLINE" '+%F %T') ==="
for lvl in $ORDER; do
  while IFS=$'\t' read -r cell pid level nk contig hot; do
    [ "$level" = "$lvl" ] || continue
    while :; do
      now=$(date +%s); [ "$now" -ge "$DEADLINE" ] && { log "deadline"; exit 0; }
      have=$(count_cell "$cell"); have=${have:-0}
      [ "$have" -ge "$TARGET" ] && break
      b=$(( TARGET-have )); [ "$b" -gt "$PAR" ] && b=$PAR
      log "cell=$cell lvl=$lvl nk=$nk have=$have -> +$b (par)"
      pids=()
      for w in $(seq 0 $((b-1))); do run_one "$cell" "$pid" "$contig" "$hot" "$((have+w))" & pids+=($!); done
      wait "${pids[@]}" 2>/dev/null || true
    done
  done < "$SPEC"
  log "level $lvl complete (both crystals at TARGET=$TARGET)"
done
log "=== maxcond ALL LEVELS complete ==="; touch "$OUT/COMPLETE"
