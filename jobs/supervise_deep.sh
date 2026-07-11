#!/usr/bin/env bash
# Cron */3 supervisor: relaunch run_deep if not alive, not COMPLETE, before deadline.
set -uo pipefail
ABS=/home/ubuntu/if-mhc; OUT="$ABS/outputs/deep"; mkdir -p "$OUT"
[ -f "$OUT/COMPLETE" ] && { ( crontab -l 2>/dev/null | grep -v supervise_deep.sh | crontab - ) || true; exit 0; }
[ -f "$OUT/deadline" ] && [ "$(date +%s)" -ge "$(cat "$OUT/deadline")" ] && { echo "$(date '+%F %T') deadline; stop cron"; ( crontab -l 2>/dev/null | grep -v supervise_deep.sh | crontab - ) || true; exit 0; }
pid=$(cat "$OUT/runner.pid" 2>/dev/null || true)
if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then exit 0; fi
echo "$(date '+%F %T') deep runner not alive -> relaunch" >> "$OUT/supervise.log"
setsid bash "$ABS/jobs/run_deep.sh" >> "$OUT/run.log" 2>&1 < /dev/null &
