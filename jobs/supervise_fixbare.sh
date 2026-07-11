#!/usr/bin/env bash
set -uo pipefail
ABS=/home/ubuntu/if-mhc; OUT="$ABS/outputs/rfd_fixbare"; mkdir -p "$OUT"
MD=$(cat "$ABS/outputs/mission/deadline" 2>/dev/null||echo 0); now=$(date +%s)
if [ -f "$OUT/COMPLETE" ] || { [ "$MD" -gt 0 ] && [ "$now" -ge "$MD" ]; }; then
  crontab -l 2>/dev/null | grep -v 'supervise_fixbare.sh' | crontab - || true; exit 0; fi
[ -f "$OUT/runner.pid" ] && kill -0 "$(cat "$OUT/runner.pid" 2>/dev/null)" 2>/dev/null && { echo "[$(date '+%F %T')] alive"; exit 0; }
echo "[$(date '+%F %T')] relaunch"; TARGET="${TARGET:-6}" setsid bash "$ABS/jobs/run_fixbare.sh" </dev/null >>"$OUT/run.log" 2>&1 &
