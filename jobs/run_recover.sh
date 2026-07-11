#!/usr/bin/env bash
# Reverse-engineering recovery ladder — RFD1 de-novo peptide, full diffuser.T=30, withTCR scaffold,
# MAX interaction hotspots on every level, and native peptide backbone fixed as MOTIF from most->none:
#   fixall (whole peptide) -> fix8 -> fix6 -> fix4 (anchors P1-2,P9-10) -> fix2 (P2,P9) -> fix0 (none).
# Question: how much peptide geometry must be pinned for RFd to recover native at full T? (then relax).
set -uo pipefail
cd /home/ubuntu/if-mhc; ABS=/home/ubuntu/if-mhc
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
OUT="$ABS/outputs/rfd_recover"; PDB="$OUT/pdb"; mkdir -p "$PDB"
SPEC="$ABS/jobs/recover_spec.tsv"; TARGET="${TARGET:-15}"; PAR="${PAR:-1}"; HOURS="${HOURS:-12}"
LOG="$OUT/run.log"
if [ -f "$OUT/deadline" ]; then DEADLINE=$(cat "$OUT/deadline"); else DEADLINE=$(( $(date +%s)+HOURS*3600 )); echo "$DEADLINE">"$OUT/deadline"; fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2
conda activate SE3nv
export LD_LIBRARY_PATH="$(cat "$ABS/RFdiffusion/SE3NV_LDPATH.txt" 2>/dev/null):${LD_LIBRARY_PATH:-}"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
count_cell(){ ls "$PDB/$1"_[0-9]*.pdb 2>/dev/null | grep -vc _split || true; }
exec 9>"$OUT/runner.lock"; flock -n 9 || { log "another runner holds lock — exit"; exit 0; }
echo $$ >"$OUT/runner.pid"; trap 'rm -f "$OUT/runner.pid"' EXIT
ORDER="fixall fix8 fix6 fix4 fix2 fix0"
cd "$ABS/RFdiffusion"
log "=== recover start TARGET=$TARGET (round-robin) deadline=$(date -d @"$DEADLINE" '+%F %T') ==="
# ROUND-ROBIN: recovery already confirmed (fixall 0.07, fix4 2.28); now sample the whole ladder evenly
# so the recovery CURVE (to_cognate vs fixed-fraction) is mapped across all levels within the phase window.
progress=1
while [ "$progress" = 1 ]; do
  progress=0
  while IFS=$'\t' read -r cell pid level contig hot; do
    [ -z "${cell:-}" ] && continue
    now=$(date +%s); [ "$now" -ge "$DEADLINE" ] && { log deadline; exit 0; }
    have=$(count_cell "$cell"); have=${have:-0}
    [ "$have" -ge "$TARGET" ] && continue
    progress=1
    log "cell=$cell lvl=$level have=$have -> +1"
    args=( inference.input_pdb="$ABS/inputs/focus_6am/${pid}_trim.pdb" "contigmap.contigs=[$contig]"
      inference.num_designs=1 inference.design_startnum="$have" diffuser.T=30
      inference.ckpt_override_path=models/Complex_base_ckpt.pt inference.output_prefix="$PDB/${cell}" )
    [ -n "$(echo "$hot"|tr -d '[:space:]')" ] && args+=( "ppi.hotspot_res=[$hot]" )
    timeout 1200 python run_inference.py "${args[@]}" >>"$LOG" 2>&1 || log "  WARN $cell failed"
  done < "$SPEC"
done
log "=== recover ALL cells at TARGET ==="; touch "$OUT/COMPLETE"
