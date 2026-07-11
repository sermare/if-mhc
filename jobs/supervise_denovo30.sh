#!/usr/bin/env bash
# Idempotent cron supervisor for run_rfd_denovo30.sh — relaunches the runner if it died (detached jobs
# do not survive Claude Code session teardown here). Self-removes the cron line at deadline / COMPLETE.
# Install:  ( crontab -l 2>/dev/null; echo "*/10 * * * * flock -n /tmp/denovo30.lock bash /home/ubuntu/if-mhc/jobs/supervise_denovo30.sh >> /home/ubuntu/if-mhc/outputs/rfd_denovo30/cron.log 2>&1" ) | crontab -
set -uo pipefail
ABS=/home/ubuntu/if-mhc; OUT="$ABS/outputs/rfd_denovo30"; mkdir -p "$OUT"
now=$(date +%s)
DEADLINE=$(cat "$OUT/deadline" 2>/dev/null || echo 0)
# stop condition: complete OR past deadline -> remove our cron line and exit
if [ -f "$OUT/COMPLETE" ] || { [ "$DEADLINE" -gt 0 ] && [ "$now" -ge "$DEADLINE" ]; }; then
  crontab -l 2>/dev/null | grep -v 'supervise_denovo30.sh' | crontab - || true
  echo "[$(date '+%F %T')] supervisor: stopping (complete or past deadline), cron removed"
  exit 0
fi
# relaunch runner if not currently running — use the PID file (kill -0), not a fragile name match
alive=0
if [ -f "$OUT/runner.pid" ] && kill -0 "$(cat "$OUT/runner.pid" 2>/dev/null)" 2>/dev/null; then alive=1; fi
if [ "$alive" = 1 ]; then
  echo "[$(date '+%F %T')] supervisor: runner alive (pid $(cat "$OUT/runner.pid"))"
else
  echo "[$(date '+%F %T')] supervisor: runner down -> relaunching"
  setsid bash "$ABS/jobs/run_rfd_denovo30.sh" </dev/null >>"$OUT/run.log" 2>&1 &
fi
