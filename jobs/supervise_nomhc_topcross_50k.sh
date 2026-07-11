#!/usr/bin/env bash
set -uo pipefail
ABS=/home/ubuntu/if-mhc; OUT="$ABS/outputs/mpnn_nomhc_topcross_50k"
[ -f "$OUT/COMPLETE" ] && { ( crontab -l 2>/dev/null | grep -v supervise_nomhc_topcross_50k.sh | crontab - ) || true; exit 0; }
pid=$(cat "$OUT/runner.pid" 2>/dev/null || true)
if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then exit 0; fi
echo "$(date '+%F %T') runner not alive -> relaunch" >> "$OUT/supervise.log"
setsid bash "$ABS/jobs/run_nomhc_topcross_50k.sh" >> "$OUT/run.log" 2>&1 < /dev/null &
