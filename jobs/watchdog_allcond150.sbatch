#!/bin/bash
# Redundant always-on watchdog for the allcond150 campaign (savio2 CPU, co_nilah lowprio).
# Submitted as --array=0-1 (two instances) with hot-standby locking: whichever grabs the flock
# is ACTIVE; the other stays alive retrying the lock, so if the active one is preempted the
# standby takes over within ~90s instead of the fleet stalling (the failure mode we hit before).
# The ACTIVE instance keeps the worker fleet topped up to CAP and writes COMPLETE when done.
# Each instance self-resubmits before its walltime; --requeue covers preemption. This is a
# BACKSTOP -- the primary engine is a large standing worker array that self-drains without it.
#SBATCH --account=co_nilah
#SBATCH --partition=savio3_htc
#SBATCH --qos=savio_lowprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --requeue
#SBATCH --time=03:00:00
#SBATCH --job-name=ac150wd
#SBATCH --output=/global/scratch/users/sergiomar10/if-mhc/outputs/allcond150/logs/ac150wd_%A_%a.out
#SBATCH --error=/global/scratch/users/sergiomar10/if-mhc/outputs/allcond150/logs/ac150wd_%A_%a.err
set -uo pipefail
ABS=/global/scratch/users/sergiomar10/if-mhc
OUT="$ABS/outputs/allcond150"; mkdir -p "$OUT/logs"
SELF="$ABS/jobs/watchdog_allcond150.sbatch"; SBATCH="$ABS/jobs/allcond150.sbatch"
CAP="${CAP:-200}"; LOWWATER="${LOWWATER:-60}"; SLEEP="${SLEEP:-120}"; RESUB_AT=9600
log(){ echo "[$(date '+%F %T')] wd${SLURM_ARRAY_TASK_ID:-0} $*"; }

# ---- hot-standby: only the flock holder acts; others stay alive and retry ----
exec 9>"$OUT/watchdog.lock"
while ! flock -n 9; do
  [ -f "$OUT/COMPLETE" ] && exit 0
  if [ "$SECONDS" -ge "$RESUB_AT" ]; then log "standby near walltime -> hand off replacement"; sbatch --array=0-0 "$SELF" >/dev/null 2>&1 || true; exit 0; fi
  log "standby (another watchdog holds lock)"; sleep 90
done
log "ACTIVE on $(hostname) CAP=$CAP LOWWATER=$LOWWATER"

while true; do
  [ -f "$OUT/COMPLETE" ] && { log "COMPLETE flag present -> stop"; exit 0; }
  python3 "$ABS/jobs/remaining_allcond150.py" > "$OUT/remaining.txt" 2>&1; rc=$?
  if [ "$rc" = 3 ]; then
    log "all cells at target -> COMPLETE"; touch "$OUT/COMPLETE"
    scancel -u "$USER" -n ac150w   2>/dev/null || true
    scancel -u "$USER" -n ac150wd  2>/dev/null || true
    exit 0
  fi
  head -1 "$OUT/remaining.txt" | sed 's/^/    /'
  inflight=$(squeue -h -r -u "$USER" -n ac150w -t PD,R 2>/dev/null | wc -l | tr -d ' ')
  log "in-flight workers=$inflight"
  if [ "${inflight:-0}" -lt "$LOWWATER" ]; then
    need=$(( CAP - inflight )); [ "$need" -lt 1 ] && need=1
    j=$(sbatch --parsable --partition=savio3_gpu --array=0-$((need-1)) "$SBATCH" 2>&1); log "refill savio3_gpu x$need -> $j"
  fi
  if [ "$SECONDS" -ge "$RESUB_AT" ]; then
    log "near walltime -> resubmit self and exit"; sbatch --array=0-0 "$SELF" >/dev/null 2>&1 || true; exit 0
  fi
  sleep "$SLEEP"
done
