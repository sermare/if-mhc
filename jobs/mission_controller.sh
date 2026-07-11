#!/usr/bin/env bash
# 8-hour mission controller: Phase RECOVER -> (determine min conditioning) -> Phase XREG (cross-register).
# Single cron owns the transition. Recover has its own supervisor cron during phase 1.
set -uo pipefail
ABS=/home/ubuntu/if-mhc; M="$ABS/outputs/mission"; mkdir -p "$M"
PY=/home/ubuntu/miniforge3/envs/esmfold2/bin/python
now=$(date +%s); MD=$(cat "$M/deadline" 2>/dev/null || echo 0)
STATE=$(cat "$M/state" 2>/dev/null || echo recover)
log(){ echo "[$(date '+%F %T')] $*" >> "$M/controller.log"; }
# global mission stop
if [ "$MD" -gt 0 ] && [ "$now" -ge "$MD" ]; then
  for s in supervise_recover supervise_xreg mission_controller; do crontab -l 2>/dev/null | grep -v "$s.sh" | crontab - || true; done
  touch "$ABS/outputs/rfd_recover/COMPLETE" "$ABS/outputs/rfd_xreg/COMPLETE" 2>/dev/null || true
  echo done > "$M/state"; log "MISSION DEADLINE reached -> all stopped"; exit 0
fi
if [ "$STATE" = recover ]; then
  # readiness: every recover cell >= MINN, OR recover COMPLETE, OR >3h into mission
  MINN=8; RP="$ABS/outputs/rfd_recover/pdb"
  ncells=$(cut -f1 "$ABS/jobs/recover_spec.tsv" | grep -c .)
  ready=$(for c in $(cut -f1 "$ABS/jobs/recover_spec.tsv"); do n=$(ls "$RP/${c}"_[0-9]*.pdb 2>/dev/null|grep -vc _split||echo 0); [ "$n" -ge "$MINN" ] && echo 1; done | grep -c 1)
  START=$(cat "$M/start" 2>/dev/null || echo "$now"); elapsed=$(( now - START ))
  if [ -f "$ABS/outputs/rfd_recover/COMPLETE" ] || [ "$ready" -ge "$ncells" ] || [ "$elapsed" -ge 10800 ]; then
    log "recover ready (ready=$ready/$ncells elapsed=${elapsed}s) -> scoring + transition to xreg"
    $PY "$ABS/py/score_denovo_designs.py" "$ABS/outputs/rfd_recover/pdb" >> "$M/recover_score.txt" 2>&1 || true
    # determine minimum fix level with median to_cognate < 1.0 A (fixall>fix8>fix6>fix4>fix2>fix0)
    $PY - >> "$M/fixmin.txt" 2>&1 <<'PYEOF' || true
import pandas as pd,os
f="/home/ubuntu/if-mhc/outputs/denovo_scores/per_design.csv"
if os.path.exists(f):
    d=pd.read_csv(f); d=d[(d.source=="recover")|(d.file.str.contains("rfd_recover"))] if "source" in d else d
    d=d[d.to_cognate.notna()]
    order=["fixall","fix8","fix6","fix4","fix2","fix0"]
    import re
    d["lvl"]=d.file.str.extract(r"_(fixall|fix\d)_")
    g=d.groupby("lvl").to_cognate.median()
    ok=[l for l in order if l in g.index and g[l]<1.0]
    print("median to_cognate by level:",{l:round(float(g[l]),2) for l in order if l in g.index})
    print("FIX_MIN (smallest fixed set that recovers, median<1.0):", ok[-1] if ok else "NONE recovered")
PYEOF
    crontab -l 2>/dev/null | grep -v 'supervise_recover.sh' | crontab - || true
    touch "$ABS/outputs/rfd_recover/COMPLETE"; kill "$(cat "$ABS/outputs/rfd_recover/runner.pid" 2>/dev/null)" 2>/dev/null || true
    pkill -9 -f "run_inference.py.*rfd_recover" 2>/dev/null || true; sleep 3
    echo xreg > "$M/state"
    ( crontab -l 2>/dev/null | grep -v 'supervise_xreg.sh'; echo "*/2 * * * * flock -n /tmp/xreg.lock env TARGET=20 bash $ABS/jobs/supervise_xreg.sh >> $ABS/outputs/rfd_xreg/cron.log 2>&1" ) | crontab -
    setsid bash "$ABS/jobs/run_xreg.sh" </dev/null >> "$ABS/outputs/rfd_xreg/run.log" 2>&1 &
    log "launched xreg"
  else log "recover phase (ready=$ready/$ncells elapsed=${elapsed}s) — waiting"; fi
else
  log "phase=$STATE"
fi
