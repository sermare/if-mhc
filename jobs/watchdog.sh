#!/usr/bin/env bash
# Cron-run every minute (survives session teardown). Keeps the promising supervisor alive
# until the campaign hits TARGET/DEADLINE (signalled by all worker DONE files).
cd /home/ubuntu/if-mhc
OUT=/home/ubuntu/if-mhc/outputs/promising
# stop once campaign complete (all 4 workers DONE)
if [ -f "$OUT/worker0.DONE" ] && [ -f "$OUT/worker1.DONE" ] && [ -f "$OUT/worker2.DONE" ] && [ -f "$OUT/worker3.DONE" ]; then
  exit 0
fi
# relaunch supervisor if not running
if pgrep -f "promising_supervisor.sh" >/dev/null 2>&1; then
  echo "[$(date -u)] watchdog: supervisor already running" >> "$OUT/watchdog.log"
else
  echo "[$(date -u)] watchdog: launching supervisor" >> "$OUT/watchdog.log"
  setsid bash /home/ubuntu/if-mhc/promising_supervisor.sh </dev/null >/dev/null 2>&1
fi
