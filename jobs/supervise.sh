#!/usr/bin/env bash
# Robust supervisor: tracks the PID it launches per job and checks it with kill -0
# (no pgrep -> immune to zombie/defunct false-positives). Keeps <=2 jobs, auto-restarts on death.
cd /home/ubuntu/if-mhc
SLOG=outputs/supervisor.log
MAXC=3
JOBS=(
  "grind_w0|run_grind_w0.sh|outputs/grind/worker0.DONE"
  "grind_w1|run_grind_w1.sh|outputs/grind/worker1.DONE"
  "grind_eval|grind_eval.sh|outputs/grind/EVAL_DONE"
)
declare -A PID
echo "[$(date)] robust supervisor start (max=$MAXC)" >> "$SLOG"
running(){ local p="${PID[$1]:-}"; [ -n "$p" ] && kill -0 "$p" 2>/dev/null; }
while true; do
  active=0; done=0
  for j in "${JOBS[@]}"; do IFS='|' read -r name script flag <<< "$j"
    [ -f "$flag" ] && { done=$((done+1)); continue; }
    running "$name" && active=$((active+1))
  done
  for j in "${JOBS[@]}"; do IFS='|' read -r name script flag <<< "$j"
    [ -f "$flag" ] && continue
    running "$name" && continue
    if [ "$active" -lt "$MAXC" ]; then
      setsid bash ./"$script" >/dev/null 2>&1 &
      PID[$name]=$!
      echo "[$(date)] launched $name pid ${PID[$name]}" >> "$SLOG"
      active=$((active+1)); sleep 8
    fi
  done
  echo "[$(date)] done=$done active=$active" >> "$SLOG"
  alldone=1; for j in "${JOBS[@]}"; do IFS='|' read -r n s flag <<< "$j"; [ -f "$flag" ] || alldone=0; done
  [ "$alldone" = 1 ] && { echo "[$(date)] ALL COMPLETE" >> "$SLOG"; break; }
  sleep 45
done
