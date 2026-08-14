#!/bin/bash
#SBATCH --job-name=skpkg --account=co_nilah --partition=savio3_htc --qos=savio_lowprio
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --time=00:30:00
#SBATCH --output=/global/scratch/users/sergiomar10/if-mhc/outputs/skempi_if/logs/package_%j.out
#SBATCH --error=/global/scratch/users/sergiomar10/if-mhc/outputs/skempi_if/logs/package_%j.out
set -uo pipefail; cd /global/scratch/users/sergiomar10/if-mhc; source jobs/skempi_env.sh
SK_DATASET=${DS:-skempi} SK_TEMP=${TP:-0.1} $PY py/skempi_merge.py 2>&1 | tail -15
echo "--- packaging ---"
$PY py/skempi_package.py --dataset ${DS:-skempi} --temp ${TP:-0.1}
echo "=== PACKAGE DONE ==="
