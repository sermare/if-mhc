#!/usr/bin/env bash
# ============================================================================
# gen.sh — generate MORE de-novo peptide designs for ANY named conditioning.
#
# USAGE:
#   bash jobs/gen.sh <PID> <COND> <N> [OUTDIR] [T]
#
#   PID     6AM5 | 6AMU                         (which crystal / register)
#   COND    L1_nterm L2_nterm_t1 L3_nterm_t2 L4_expanded L5_max   (contact ladder)
#           k18 k24 max                          (max-contact ladder)
#           -> see jobs/all_conditionings.tsv for the full list + exact hotspots
#   N       how many MORE designs to ADD (on top of whatever already exists)
#   OUTDIR  (optional) where to write; default routes to the campaign that
#           conditioning belongs to so the scorer tags it correctly:
#             L*  -> outputs/promising/pdb    k*/max -> outputs/rfd_maxcond/pdb
#   T       (optional) diffuser timesteps; default 30 (full de-novo)
#
# EXAMPLES:
#   bash jobs/gen.sh 6AM5 L4_expanded 100      # 100 more of the "hit" conditioning
#   bash jobs/gen.sh 6AMU L5_max 40            # 40 more, DRG crystal, max contacts
#   bash jobs/gen.sh 6AM5 max 20               # 20 more of the 37-hotspot maxcond
#
# Fully de-novo peptide (contig 10-10, NOTHING templated), full T, withTCR context.
# Resumable + single-instance per (PID,COND) via flock. Every design is a fresh
# stochastic draw — this is exactly the best-of-k sampling you are probing.
# Re-score anytime with:  bash jobs/evaluate.sh
# ============================================================================
set -uo pipefail
ABS=/home/ubuntu/if-mhc; cd "$ABS"
PID="${1:?PID (6AM5|6AMU)}"; COND="${2:?COND (see jobs/all_conditionings.tsv)}"; N="${3:?N designs to ADD}"
TCOND="${5:-30}"
TABLE="$ABS/jobs/all_conditionings.tsv"

# ---- look up contig + hotspots for this (PID,COND) ----
row=$(awk -F'\t' -v p="$PID" -v c="$COND" '$1==p && $2==c {print; exit}' "$TABLE")
[ -z "$row" ] && { echo "ERROR: no conditioning '$PID $COND' in $TABLE"; echo "available:"; awk -F'\t' '{print "  "$1" "$2}' "$TABLE"; exit 1; }
IFS=$'\t' read -r _pid _cond style contig hot <<< "$row"

# ---- default output dir + naming style routes to the right campaign source ----
if [ -n "${4:-}" ]; then OUTDIR="$4"; else
  case "$style" in maxcond) OUTDIR="outputs/rfd_maxcond/pdb";; *) OUTDIR="outputs/promising/pdb";; esac
fi
mkdir -p "$ABS/$OUTDIR"
LOGD="$ABS/outputs/gen"; mkdir -p "$LOGD"; LOG="$LOGD/${PID}_${COND}.log"
LOCK="$LOGD/${PID}_${COND}.lock"

# ---- count existing designs for this cell (both naming styles) ----
count_cell(){ ls "$ABS/$OUTDIR/${PID}_${COND}"_*.pdb 2>/dev/null | grep -vc _split || true; }

exec 9>"$LOCK"; flock -n 9 || { echo "another gen.sh already running for $PID $COND — exit"; exit 0; }
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh; conda activate SE3nv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2
export LD_LIBRARY_PATH="$(cat "$ABS/RFdiffusion/SE3NV_LDPATH.txt" 2>/dev/null):${LD_LIBRARY_PATH:-}"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

start_have=$(count_cell); start_have=${start_have:-0}
TARGET=$(( start_have + N ))
log "=== gen $PID $COND : have $start_have -> target $TARGET (ADD $N)  style=$style T=$TCOND ==="
log "    contig=[$contig]"
log "    hotspots=[$hot]"
cd "$ABS/RFdiffusion"
while :; do
  have=$(count_cell); have=${have:-0}
  [ "$have" -ge "$TARGET" ] && break
  if [ "$style" = maxcond ]; then
    prefix="$ABS/$OUTDIR/${PID}_${COND}"; start="$have"       # sequential {pid}_{cond}_{idx}
  else
    stamp=$(date +%s%N | tail -c 8); prefix="$ABS/$OUTDIR/${PID}_${COND}_wG${stamp}"; start=0
  fi
  log "  design $((have+1))/$TARGET -> $(basename "$prefix")"
  timeout 1200 python run_inference.py \
    inference.input_pdb="$ABS/inputs/focus_6am/${PID}_trim.pdb" \
    "contigmap.contigs=[$contig]" "ppi.hotspot_res=[$hot]" \
    inference.num_designs=1 inference.design_startnum="$start" diffuser.T="$TCOND" \
    inference.ckpt_override_path=models/Complex_base_ckpt.pt \
    inference.output_prefix="$prefix" >>"$LOG" 2>&1 || log "  WARN design failed"
done
log "=== gen $PID $COND DONE: $(count_cell) designs present ==="
