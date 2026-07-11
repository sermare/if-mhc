#!/usr/bin/env bash
set -uo pipefail
ABS=/home/ubuntu/if-mhc; OUT="$ABS/outputs/allcond300"; mkdir -p "$OUT"
now=$(date +%s); DEADLINE=$(cat "$OUT/deadline" 2>/dev/null || echo 0)
if [ -f "$OUT/COMPLETE" ] || { [ "$DEADLINE" -gt 0 ] && [ "$now" -ge "$DEADLINE" ]; }; then
  crontab -l 2>/dev/null | grep -v 'supervise_allcond300.sh' | crontab - || true
  echo "[$(date '+%F %T')] stop"; exit 0; fi
alive=0; [ -f "$OUT/runner.pid" ] && kill -0 "$(cat "$OUT/runner.pid" 2>/dev/null)" 2>/dev/null && alive=1
if [ "$alive" = 1 ]; then echo "[$(date '+%F %T')] alive $(cat "$OUT/runner.pid")"
else echo "[$(date '+%F %T')] relaunch"; HOURS="${HOURS:-72}" setsid bash "$ABS/jobs/run_allcond300.sh" </dev/null >>"$OUT/run.log" 2>&1 & fi
