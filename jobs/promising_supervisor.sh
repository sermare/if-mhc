#!/usr/bin/env bash
# Keeps 2 promising workers alive (worker0->6AMU, worker1->6AM5) until TARGET or DEADLINE.
# Restarts DEAD workers AND HUNG workers (no new pdb in STALL_S seconds -> RFdiffusion hang).
ABS=/home/ubuntu/if-mhc; cd "$ABS"
OUT=$ABS/outputs/promising; NW=2; STALL_S=1200   # 20 min with no new backbone => treat as hung
CRYST=(6AMU 6AM5)
SLOG=$OUT/supervisor.log
pkill -9 -f run_promising_worker.sh 2>/dev/null; pkill -9 -f "run_inference.py.*promising" 2>/dev/null
sleep 2
echo "[$(date -u)] supervisor start (NW=$NW, per-crystal, hang-detect ${STALL_S}s) — cleaned strays" >> "$SLOG"
declare -A PID
running(){ local p="${PID[$1]:-}"; [ -n "$p" ] && kill -0 "$p" 2>/dev/null; }
newest_pdb_age(){ # seconds since newest backbone pdb (or 999999 if none)
  local f; f=$(ls -t $OUT/pdb/*.pdb 2>/dev/null | grep -v _split | head -1)
  [ -z "$f" ] && { echo 999999; return; }
  echo $(( $(date +%s) - $(stat -c %Y "$f") ))
}
launch(){ local w=$1
  setsid bash "$ABS/run_promising_worker.sh" "$w" "$NW" "${CRYST[$w]}" </dev/null >>"$OUT/worker${w}.log" 2>&1 &
  PID[$w]=$!
  echo "[$(date -u)] launched worker $w (${CRYST[$w]}) pid ${PID[$w]}" >> "$SLOG"
}
while true; do
  cnt=$(ls $OUT/pdb/*.pdb 2>/dev/null | grep -vc _split)
  tgt=$(cat $OUT/TARGET 2>/dev/null || echo 300)
  dl=$(cat $OUT/DEADLINE 2>/dev/null || echo 0)
  if [ "$cnt" -ge "$tgt" ] || [ "$(date +%s)" -ge "$dl" ]; then
    echo "[$(date -u)] target/deadline reached (cnt=$cnt tgt=$tgt) — stopping" >> "$SLOG"
    for w in $(seq 0 $((NW-1))); do touch "$OUT/worker${w}.DONE"; done
    break
  fi
  # HANG DETECTION: if no new pdb in STALL_S, kill workers+rfdiff and relaunch fresh
  if [ "$(newest_pdb_age)" -gt "$STALL_S" ]; then
    echo "[$(date -u)] HANG detected (no new pdb in >${STALL_S}s, cnt=$cnt) — killing+restarting workers" >> "$SLOG"
    pkill -9 -f run_promising_worker.sh 2>/dev/null; pkill -9 -f "run_inference.py.*promising" 2>/dev/null
    sleep 3; for w in $(seq 0 $((NW-1))); do unset 'PID[$w]'; done
  fi
  for w in $(seq 0 $((NW-1))); do running "$w" || launch "$w"; done
  sleep 60
done
echo "[$(date -u)] supervisor exit" >> "$SLOG"
