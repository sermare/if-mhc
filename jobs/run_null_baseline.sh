#!/usr/bin/env bash
# Null/background baseline campaign — TRUE zero-conditioning RFdiffusion draws: no ppi.hotspot_res
# at all, no peptide-backbone motif template (contig 10-10), full T=30, MHC/B2M/TCR held fixed as
# structural context only. This is distinct from the existing "fix0" cells in recover_spec.tsv,
# which (despite the name) still carry the full rich hotspot list -- only the motif template is
# absent there. Purpose: establish the background rate at which an arbitrary groove-compatible Cα
# backbone falls inside the mean+3sigma acceptance envelope of either native register by chance
# alone, addressing the reviewer-identified gap that 5/1,305 "hits" cannot otherwise be
# distinguished from a null. RFdiffusion backbone generation ONLY -- no ProteinMPNN step.
set -uo pipefail
cd /home/ubuntu/if-mhc; ABS=/home/ubuntu/if-mhc
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
OUT="$ABS/outputs/rfd_null"; PDB="$OUT/pdb"; mkdir -p "$PDB"
TARGET="${TARGET:-85}"; HOURS="${HOURS:-14}"
LOG="$OUT/run.log"
DEADLINE=$(( $(date +%s)+HOURS*3600 )); echo "$DEADLINE" > "$OUT/deadline"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2
conda activate SE3nv
export LD_LIBRARY_PATH="$(cat "$ABS/RFdiffusion/SE3NV_LDPATH.txt" 2>/dev/null):${LD_LIBRARY_PATH:-}"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
count_cell(){ ls "$PDB/$1"_[0-9]*.pdb 2>/dev/null | grep -vc _split || true; }
exec 9>"$OUT/runner.lock"; flock -n 9 || { log "another null runner holds lock — exit"; exit 0; }
echo $$ >"$OUT/runner.pid"; trap 'rm -f "$OUT/runner.pid"' EXIT
cd "$ABS/RFdiffusion"
log "=== null baseline start TARGET=$TARGET per crystal (zero hotspots, zero template) deadline=$(date -d @"$DEADLINE" '+%F %T') ==="

declare -A CONTIG
CONTIG[6AM5]="A1-180/0 B1-100/0 D1-115/0 E1-120/0 10-10"
CONTIG[6AMU]="A2-180/0 B0-99/0 D2-115/0 E4-120/0 10-10"

progress=1
while [ "$progress" = 1 ]; do
  progress=0
  for pid in 6AM5 6AMU; do
    now=$(date +%s); [ "$now" -ge "$DEADLINE" ] && { log deadline; exit 0; }
    cell="${pid}_null0"
    contig="${CONTIG[$pid]}"
    have=$(count_cell "$cell"); have=${have:-0}
    [ "$have" -ge "$TARGET" ] && continue
    progress=1
    log "cell=$cell have=$have -> +1 (target $TARGET) contig=[$contig] hotspots=NONE"
    timeout 1200 python run_inference.py \
      inference.input_pdb="$ABS/inputs/focus_6am/${pid}_trim.pdb" \
      "contigmap.contigs=[$contig]" \
      inference.num_designs=1 inference.design_startnum="$have" diffuser.T=30 \
      inference.ckpt_override_path=models/Complex_base_ckpt.pt \
      inference.output_prefix="$PDB/${cell}" >>"$LOG" 2>&1 || log "  WARN $cell failed"
  done
done
log "=== null baseline: all cells reached TARGET=$TARGET ==="; touch "$OUT/COMPLETE"
