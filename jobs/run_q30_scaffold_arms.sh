#!/usr/bin/env bash
# Scaffold-mode joint-conditioning arms, run SEQUENTIALLY (one arm's 2 workers at a time) to avoid the
# gpu_wait over-admission race. Each arm: hold P1-2, rebuild P3-10 under its joint hotspot set (W12).
# The valid test of basin steering (partial diffusion is frozen). Idempotent/resumable per seed.
set -uo pipefail
ABS=/home/ubuntu/if-mhc
run(){  # $1=arm suffix  $2=GIG hotspots  $3=DRG hotspots
  local d="$ABS/outputs/rfdiff_q30_$1"
  [ -f "$d/RFDIFF_COMPLETE" ] && { echo "[$(date)] $1 already complete, skip"; return; }
  echo "[$(date)] === scaffold arm $1 ==="
  OUT="$d" WINDOWS=W12 NDES_CRYSTAL=4 NDES_RELAXED=1 GPU_FREE_MIN=7000 \
    GIG_HOT="$2" DRG_HOT="$3" bash "$ABS/jobs/run_q30_nslide.sh"
}
run sc_tcr     "D30,E97"                              "D30,E100"
run sc_apoc    "D30,E97,A7,A63,A66,A70,A99,A159,A171" "D30,E100,A7,A63,A66,A70,A99,A159,A171"
run sc_fpocket "D30,A77,A84,A116,A123"                "D30,A77,A84,A116,A123"
echo "[$(date)] all scaffold arms done" | tee "$ABS/outputs/rfdiff_q30_scaffold_COMPLETE"
