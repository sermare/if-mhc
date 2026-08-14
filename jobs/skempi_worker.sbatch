#!/bin/bash
# Self-draining GPU worker for the SKEMPI inverse-folding sweep.
# Short walltime so it schedules fast; claims work units atomically (mkdir), runs
# them until the walltime reserve is hit, then exits. Idempotent: a unit whose
# part file exists is skipped, and a claim left behind by a preempted worker is
# swept by the keeper, so the fleet converges without bookkeeping.
#SBATCH --job-name=skw
#SBATCH --account=co_nilah
#SBATCH --partition=savio3_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --requeue
#SBATCH --time=00:30:00
#SBATCH --output=/global/scratch/users/sergiomar10/if-mhc/outputs/skempi_if/logs/w_%A_%a.out
#SBATCH --error=/global/scratch/users/sergiomar10/if-mhc/outputs/skempi_if/logs/w_%A_%a.out
set -uo pipefail
cd /global/scratch/users/sergiomar10/if-mhc
source jobs/skempi_env.sh

RESERVE=${RESERVE:-360}          # stop starting units with under 6 min left
BUDGET=$((30*60 - RESERVE))
CLAIMS="$OUTDIR/${SK_DATASET:-skempi}/T${SK_TEMP:-0.1}/claims"; mkdir -p "$CLAIMS" "$OUTDIR/logs"
log(){ echo "[$(date '+%F %T')] w${SLURM_ARRAY_TASK_ID:-0} $*"; }
log "start on $(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

held=""
cleanup(){ [ -n "$held" ] && rmdir "$held" 2>/dev/null; }   # release an unfinished claim
trap cleanup EXIT

did=0
while [ "$SECONDS" -lt "$BUDGET" ]; do
  got=0
  while read -r model arm cplx chunk nseq dset temp; do
    [ -z "${model:-}" ] && continue
    [ "$SECONDS" -ge "$BUDGET" ] && break
    claim="$CLAIMS/${model}__${arm}__${cplx}__c$(printf %02d "$chunk")"
    mkdir "$claim" 2>/dev/null || continue      # atomic claim; someone else has it
    held="$claim"
    log "run $dset/T$temp $model/$arm/$cplx/c$chunk n=$nseq"
    if $PY py/skempi_run_unit.py --model "$model" --arm "$arm" --complex "$cplx" \
           --chunk "$chunk" --nseq "$nseq" --temp "$temp" --dataset "$dset"; then
      did=$((did+1)); held=""                   # part file written; claim stays as a marker
    else
      log "FAILED $model/$arm/$cplx/c$chunk -- releasing claim"
      rmdir "$claim" 2>/dev/null; held=""
    fi
    got=1
    break
  done < <($PY py/skempi_units.py --list | head -40)

  if [ "$got" = 0 ]; then
    if $PY py/skempi_units.py --status >/dev/null 2>&1; then
      log "no free units right now -- waiting"; sleep 30
    else
      log "all units complete"; break
    fi
  fi
done
log "exit after $did units, ${SECONDS}s"
