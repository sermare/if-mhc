#!/bin/bash
# CPU keeper for the SKEMPI inverse-folding sweep. Runs on savio3_htc (schedules
# in seconds, costs no GPU), keeps CAP short GPU workers in flight, sweeps claims
# orphaned by preemption, and stops the fleet once every unit has a part file.
# Self-resubmits before its own walltime; --requeue covers its preemption.
#SBATCH --job-name=skkeep
#SBATCH --account=co_nilah
#SBATCH --partition=savio3_htc
#SBATCH --qos=savio_lowprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --requeue
#SBATCH --time=04:00:00
#SBATCH --output=/global/scratch/users/sergiomar10/if-mhc/outputs/skempi_if/logs/keeper_%j.out
#SBATCH --error=/global/scratch/users/sergiomar10/if-mhc/outputs/skempi_if/logs/keeper_%j.out
set -uo pipefail
ABS=/global/scratch/users/sergiomar10/if-mhc; cd "$ABS"
source jobs/skempi_env.sh
SELF="$ABS/jobs/skempi_keeper.sbatch"; WORKER="$ABS/jobs/skempi_worker.sbatch"
export SK_DATASET="${SK_DATASET:-skempi}" SK_TEMP="${SK_TEMP:-0.1}"
PHASE_ENV="ALL,SK_DATASET=$SK_DATASET,SK_TEMP=$SK_TEMP"
CAP="${CAP:-120}"          # target workers in flight
LOWWATER="${LOWWATER:-50}"  # refill once the fleet drops below this
SLEEP="${SLEEP:-120}"
RESUB_AT=$((3*3600 + 1800))
log(){ echo "[$(date '+%F %T')] keeper $*"; }

# only one keeper acts; a second stays hot and takes over if this one is preempted
exec 9>"$OUTDIR/keeper.${SK_DATASET:-skempi}.T${SK_TEMP:-0.1}.lock"
while ! flock -n 9; do
  [ -f "$OUTDIR/COMPLETE.${SK_DATASET:-skempi}.T${SK_TEMP:-0.1}" ] && exit 0
  if [ "$SECONDS" -ge "$RESUB_AT" ]; then sbatch --array=0-1 --export="$PHASE_ENV" "$SELF" >/dev/null 2>&1 || true; exit 0; fi
  log "standby (another keeper holds the lock)"; sleep 90
done
log "ACTIVE on $(hostname) CAP=$CAP LOWWATER=$LOWWATER"

while true; do
  [ -f "$OUTDIR/COMPLETE.${SK_DATASET:-skempi}.T${SK_TEMP:-0.1}" ] && { log "COMPLETE -> stop"; exit 0; }

  $PY py/skempi_units.py --sweep --stale-min 45 | sed 's/^/    /'
  $PY py/skempi_units.py --status > "$OUTDIR/status.$SK_DATASET.T$SK_TEMP.txt" 2>&1; rc=$?
  tail -1 "$OUTDIR/status.$SK_DATASET.T$SK_TEMP.txt" | sed 's/^/    /'

  if [ "$rc" = 3 ]; then
    log "every unit has a part file -> COMPLETE for ${SK_DATASET:-skempi}/T${SK_TEMP:-0.1}"
    touch "$OUTDIR/COMPLETE.${SK_DATASET:-skempi}.T${SK_TEMP:-0.1}"
    scancel -u "$USER" -n skw 2>/dev/null || true
    log "merging phase"
    $PY py/skempi_merge.py 2>&1 | tail -25

    # hand off to the next line of the run plan, so the whole plan drains unattended
    nxt=$(awk -v d="${SK_DATASET:-skempi}" -v t="${SK_TEMP:-0.1}" '
      /^[^#]/ && NF==2 { if (found) { print; exit } ; if ($1==d && $2==t) found=1 }
    ' "$ABS/jobs/skempi_phases.txt")
    if [ -n "$nxt" ]; then
      nd=$(echo "$nxt" | awk '{print $1}'); nt=$(echo "$nxt" | awk '{print $2}')
      log "next phase: $nd/T$nt -> launching keeper + workers"
      sbatch --array=0-1 --export=ALL,SK_DATASET="$nd",SK_TEMP="$nt" "$SELF" >/dev/null 2>&1 || true
      sbatch --array=0-119 --export=ALL,SK_DATASET="$nd",SK_TEMP="$nt" "$WORKER" >/dev/null 2>&1 || true
    else
      log "run plan finished -- no further phases"
    fi
    exit 0
  fi

  inflight=$(squeue -h -r -u "$USER" -n skw -t PD,R 2>/dev/null | wc -l | tr -d ' ')
  log "workers in flight = ${inflight:-0}"
  if [ "${inflight:-0}" -lt "$LOWWATER" ]; then
    need=$(( CAP - inflight )); [ "$need" -lt 1 ] && need=1
    j=$(sbatch --parsable --array=0-$((need-1)) --export="$PHASE_ENV" "$WORKER" 2>&1)
    log "submitted $need workers -> $j"
  fi

  if [ "$SECONDS" -ge "$RESUB_AT" ]; then
    log "near walltime -> hand off to a fresh keeper"
    sbatch --array=0-1 --export="$PHASE_ENV" "$SELF" >/dev/null 2>&1 || true
    exit 0
  fi
  sleep "$SLEEP"
done
