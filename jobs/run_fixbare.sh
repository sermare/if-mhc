#!/usr/bin/env bash
# fix6/fix4 with NO hotspots (bare motif) — isolates whether max-hotspots on top of geometry lowers
# FIX_MIN. Compare to recover fix6+maxhot=1.45 / fix4+maxhot=1.94. Full T=30, withTCR, round-robin.
set -uo pipefail
cd /home/ubuntu/if-mhc; ABS=/home/ubuntu/if-mhc
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
OUT="$ABS/outputs/rfd_fixbare"; PDB="$OUT/pdb"; mkdir -p "$PDB"
SPEC="$ABS/jobs/fixbare_spec.tsv"; TARGET="${TARGET:-6}"
DEADLINE=$(cat "$ABS/outputs/mission/deadline" 2>/dev/null || echo $(( $(date +%s)+7200 )))
LOG="$OUT/run.log"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2
conda activate SE3nv
export LD_LIBRARY_PATH="$(cat "$ABS/RFdiffusion/SE3NV_LDPATH.txt" 2>/dev/null):${LD_LIBRARY_PATH:-}"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
count_cell(){ ls "$PDB/$1"_[0-9]*.pdb 2>/dev/null | grep -vc _split || true; }
exec 9>"$OUT/runner.lock"; flock -n 9 || { log "locked"; exit 0; }
echo $$ >"$OUT/runner.pid"; trap 'rm -f "$OUT/runner.pid"' EXIT
cd "$ABS/RFdiffusion"; log "=== fixbare start TARGET=$TARGET ==="
progress=1
while [ "$progress" = 1 ]; do
  progress=0
  while IFS=$'\t' read -r cell pid level contig hot; do
    [ -z "${cell:-}" ] && continue
    now=$(date +%s); [ "$now" -ge "$DEADLINE" ] && { log deadline; exit 0; }
    have=$(count_cell "$cell"); have=${have:-0}
    [ "$have" -ge "$TARGET" ] && continue
    progress=1; log "cell=$cell have=$have -> +1 (no hotspots)"
    timeout 1200 python run_inference.py inference.input_pdb="$ABS/inputs/focus_6am/${pid}_trim.pdb" \
      "contigmap.contigs=[$contig]" inference.num_designs=1 inference.design_startnum="$have" diffuser.T=30 \
      inference.ckpt_override_path=models/Complex_base_ckpt.pt inference.output_prefix="$PDB/${cell}" \
      >>"$LOG" 2>&1 || log "  WARN $cell failed"
  done < "$SPEC"
done
log "=== fixbare complete ==="; touch "$OUT/COMPLETE"
