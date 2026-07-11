#!/usr/bin/env bash
# Idempotent, flock-guarded supervisor for the three Q30 conditioning arms (TCR / A-pocket / N-slide).
# Survives session teardown via cron: if no diffusion is running and an arm is incomplete, it
# (re)launches that arm's run script (which idempotently skips finished designs). Self-uninstalls
# from cron once all three arms have written RFDIFF_COMPLETE. Balanced GPU_FREE_MIN=6500 so all
# three share the L4 (~3 concurrent workers) instead of starving the slow scaffold arm.
set -uo pipefail
ABS=/home/ubuntu/if-mhc
exec 9>/tmp/resume_q30_arms.lock; flock -n 9 || exit 0   # single instance

TCR=$ABS/outputs/rfdiff_q30_tcr
APOC=$ABS/outputs/rfdiff_q30_apoc
NSL=$ABS/outputs/rfdiff_q30_nslide
done_flag(){ [ -f "$1/RFDIFF_COMPLETE" ]; }

# all done -> uninstall self from cron and stop
if done_flag "$TCR" && done_flag "$APOC" && done_flag "$NSL"; then
  crontab -l 2>/dev/null | grep -v 'resume_q30_arms.sh' | crontab - || true
  echo "[$(date)] all arms complete; cron removed" >> "$ABS/outputs/rfdiff_q30_supervisor.log"
  exit 0
fi

# diffusion already running -> nothing to do
if pgrep -f run_inference.py >/dev/null 2>&1; then exit 0; fi

echo "[$(date)] no diffusion running; (re)launching incomplete arms" >> "$ABS/outputs/rfdiff_q30_supervisor.log"
if ! done_flag "$TCR"; then
  OUT="$TCR" GIG_HOT="D30,E97" DRG_HOT="D30,E100" GPU_FREE_MIN=6500 \
    setsid bash "$ABS/jobs/run_q30_arm.sh" </dev/null >>"$TCR/supervisor.log" 2>&1 &
fi
if ! done_flag "$APOC"; then
  OUT="$APOC" GIG_HOT="D30,E97,A7,A63,A66,A70,A99,A159,A171" DRG_HOT="D30,E100,A7,A63,A66,A70,A99,A159,A171" GPU_FREE_MIN=6500 \
    setsid bash "$ABS/jobs/run_q30_arm.sh" </dev/null >>"$APOC/supervisor.log" 2>&1 &
fi
if ! done_flag "$NSL"; then
  OUT="$NSL" ONLY=trim NDES_CRYSTAL=4 GPU_FREE_MIN=6500 \
    setsid bash "$ABS/jobs/run_q30_nslide.sh" </dev/null >>"$NSL/supervisor.log" 2>&1 &
fi
exit 0
