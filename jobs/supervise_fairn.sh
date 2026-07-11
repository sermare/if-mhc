#!/usr/bin/env bash
# Cron */2 supervisor: relaunch run_fairn if not alive and not COMPLETE and before deadline.
set -uo pipefail
ABS=/home/ubuntu/if-mhc; OUT="$ABS/outputs/fairn"; mkdir -p "$OUT"
[ -f "$OUT/COMPLETE" ] && exit 0
[ -f "$OUT/deadline" ] && [ "$(date +%s)" -ge "$(cat "$OUT/deadline")" ] && { echo "$(date '+%F %T') deadline reached, stopping cron"; ( crontab -l 2>/dev/null | grep -v supervise_fairn.sh | crontab - ) || true; exit 0; }
pid=$(cat "$OUT/runner.pid" 2>/dev/null || true)
if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then exit 0; fi
echo "$(date '+%F %T') runner not alive -> relaunch" >> "$OUT/supervise.log"
setsid bash "$ABS/jobs/run_fairn.sh" >> "$OUT/run.log" 2>&1 < /dev/null &
