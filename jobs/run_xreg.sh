#!/usr/bin/env bash
# Cross-register generation — RFD1 full T=30 on the CHIMERA seeds (base scaffold carrying the OTHER
# register's grafted C-terminus). Fix the grafted other-register anchor as motif (min geometric seed),
# de-novo the rest, max interaction hotspots. Question: can RFd complete a coherent peptide SEATED in
# the non-cognate register on this structure? (uses the minimum-conditioning knowledge from rfd_recover)
set -uo pipefail
cd /home/ubuntu/if-mhc; ABS=/home/ubuntu/if-mhc
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
OUT="$ABS/outputs/rfd_xreg"; PDB="$OUT/pdb"; mkdir -p "$PDB"
SPEC="$ABS/jobs/xreg_spec.tsv"; TARGET="${TARGET:-20}"; SEEDDIR="$ABS/outputs/rfdiff_q30_v4arm2/seeds"
LOG="$OUT/run.log"; DEADLINE=$(cat "$ABS/outputs/mission/deadline" 2>/dev/null || echo $(( $(date +%s)+28800 )))
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2
conda activate SE3nv
export LD_LIBRARY_PATH="$(cat "$ABS/RFdiffusion/SE3NV_LDPATH.txt" 2>/dev/null):${LD_LIBRARY_PATH:-}"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
count_cell(){ ls "$PDB/$1"_[0-9]*.pdb 2>/dev/null | grep -vc _split || true; }
exec 9>"$OUT/runner.lock"; flock -n 9 || { log "another xreg runner holds lock"; exit 0; }
echo $$ >"$OUT/runner.pid"; trap 'rm -f "$OUT/runner.pid"' EXIT
cd "$ABS/RFdiffusion"
log "=== xreg start TARGET=$TARGET deadline=$(date -d @"$DEADLINE" '+%F %T') ==="
progress=1
while [ "$progress" = 1 ]; do
  progress=0
  while IFS=$'\t' read -r cell seed pid tgt level contig hot; do
    [ -z "${cell:-}" ] && continue
    now=$(date +%s); [ "$now" -ge "$DEADLINE" ] && { log "mission deadline"; exit 0; }
    have=$(count_cell "$cell"); have=${have:-0}
    [ "$have" -ge "$TARGET" ] && continue
    progress=1
    log "cell=$cell tgt=$tgt lvl=$level have=$have -> +1"
    args=( inference.input_pdb="$SEEDDIR/${seed}.pdb" "contigmap.contigs=[$contig]"
      inference.num_designs=1 inference.design_startnum="$have" diffuser.T=30
      inference.ckpt_override_path=models/Complex_base_ckpt.pt inference.output_prefix="$PDB/${cell}" )
    [ -n "$(echo "$hot"|tr -d '[:space:]')" ] && args+=( "ppi.hotspot_res=[$hot]" )
    timeout 1200 python run_inference.py "${args[@]}" >>"$LOG" 2>&1 || log "  WARN $cell failed"
  done < "$SPEC"
done
log "=== xreg ALL cells at TARGET ==="; touch "$OUT/COMPLETE"
