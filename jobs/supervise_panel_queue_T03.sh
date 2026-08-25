#!/usr/bin/env bash
set -uo pipefail
ABS=/home/ubuntu/if-mhc; OUT="$ABS/outputs/panel_T03"
mkdir -p "$OUT"

[ -f "$OUT/ALL_COMPLETE" ] && { ( crontab -l 2>/dev/null | grep -v supervise_panel_queue_T03.sh | crontab - ) || true; exit 0; }
pid=$(cat "$OUT/queue.pid" 2>/dev/null || true)
if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then exit 0; fi
echo "$(date '+%F %T') runner not alive -> relaunch" >> "$OUT/supervise.log"
setsid bash -c 'echo $$ > '"$OUT"'/queue.pid; exec bash '"$ABS"'/jobs/run_panel_queue_T03.sh' >> "$OUT/supervise.log" 2>&1 < /dev/null &
