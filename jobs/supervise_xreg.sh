#!/usr/bin/env bash
set -uo pipefail
ABS=/home/ubuntu/if-mhc; OUT="$ABS/outputs/rfd_xreg"; mkdir -p "$OUT"
MD=$(cat "$ABS/outputs/mission/deadline" 2>/dev/null || echo 0); now=$(date +%s)
if [ -f "$OUT/COMPLETE" ] || { [ "$MD" -gt 0 ] && [ "$now" -ge "$MD" ]; }; then
  crontab -l 2>/dev/null | grep -v 'supervise_xreg.sh' | crontab - || true; echo "[$(date '+%F %T')] xreg stop"; exit 0; fi
alive=0; [ -f "$OUT/runner.pid" ] && kill -0 "$(cat "$OUT/runner.pid" 2>/dev/null)" 2>/dev/null && alive=1
if [ "$alive" = 1 ]; then echo "[$(date '+%F %T')] xreg alive"; else echo "[$(date '+%F %T')] xreg relaunch"; TARGET="${TARGET:-20}" setsid bash "$ABS/jobs/run_xreg.sh" </dev/null >>"$OUT/run.log" 2>&1 & fi
