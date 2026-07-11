#!/usr/bin/env bash
# Cron supervisor for run_maxcond.sh — relaunch-if-dead (PID file), self-remove cron at deadline/COMPLETE.
set -uo pipefail
ABS=/home/ubuntu/if-mhc; OUT="$ABS/outputs/rfd_maxcond"; mkdir -p "$OUT"
now=$(date +%s); DEADLINE=$(cat "$OUT/deadline" 2>/dev/null || echo 0)
if [ -f "$OUT/COMPLETE" ] || { [ "$DEADLINE" -gt 0 ] && [ "$now" -ge "$DEADLINE" ]; }; then
  crontab -l 2>/dev/null | grep -v 'supervise_maxcond.sh' | crontab - || true
  echo "[$(date '+%F %T')] supervisor: stop (complete/deadline), cron removed"; exit 0
fi
alive=0
if [ -f "$OUT/runner.pid" ] && kill -0 "$(cat "$OUT/runner.pid" 2>/dev/null)" 2>/dev/null; then alive=1; fi
if [ "$alive" = 1 ]; then echo "[$(date '+%F %T')] supervisor: runner alive ($(cat "$OUT/runner.pid"))"
else echo "[$(date '+%F %T')] supervisor: relaunch"; TARGET="${TARGET:-20}" PAR="${PAR:-1}" HOURS="${HOURS:-12}" \
  setsid bash "$ABS/jobs/run_maxcond.sh" </dev/null >>"$OUT/run.log" 2>&1 & fi
