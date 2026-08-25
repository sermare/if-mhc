#!/usr/bin/env bash
set -uo pipefail
ABS=/home/ubuntu/if-mhc; OUT="$ABS/outputs/mpnn_3hg1_100k"

# Queue gate: do not start until the T=0.3/10K native campaign is fully done -- explicitly requested
# to run after, not concurrently with, that job.
GATE="$ABS/outputs/native_T03_10k/COMPLETE"
if [ ! -f "$GATE" ]; then
  echo "$(date '+%F %T') waiting on $GATE -- native_T03_10k not complete yet" >> "$OUT/supervise.log"
  exit 0
fi

[ -f "$OUT/COMPLETE" ] && { ( crontab -l 2>/dev/null | grep -v supervise_3hg1_100k.sh | crontab - ) || true; exit 0; }
pid=$(cat "$OUT/runner.pid" 2>/dev/null || true)
if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then exit 0; fi
echo "$(date '+%F %T') runner not alive -> relaunch" >> "$OUT/supervise.log"
setsid bash "$ABS/jobs/run_3hg1_100k.sh" >> "$OUT/run.log" 2>&1 < /dev/null &
