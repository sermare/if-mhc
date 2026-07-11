#!/usr/bin/env bash
# One-shot launcher: queue the whole allcond150 fleet + the always-on watchdog, in one go.
#   - GPU worker array on savio3_gpu   (co_nilah / savio_lowprio, <1h each, ~30 designs each)
#   - GPU worker array on savio2_1080ti (same)
#   - CPU watchdog on savio2 that keeps the fleet topped up and resubmits dead jobs.
set -uo pipefail
ABS=/global/scratch/users/sergiomar10/if-mhc
SB="$ABS/jobs/allcond150.sbatch"; WD="$ABS/jobs/watchdog_allcond150.sbatch"
mkdir -p "$ABS/outputs/allcond150/logs"
N3="${N3:-96}"; N2="${N2:-48}"     # initial array sizes per partition

echo "== submitting allcond150 fleet (one go) =="
j3=$(sbatch --parsable --partition=savio3_gpu   --array=0-$((N3-1)) "$SB"); echo "savio3_gpu    array ($N3): $j3"
j2=$(sbatch --parsable --partition=savio2_1080ti --array=0-$((N2-1)) "$SB"); echo "savio2_1080ti array ($N2): $j2"
jw=$(sbatch --parsable "$WD");                                             echo "savio2 watchdog:          $jw"
echo "== done. monitor: squeue -u \$USER -n ac150w,ac150wd | cat ; tail -f outputs/allcond150/remaining.txt =="
