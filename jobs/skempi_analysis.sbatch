#!/bin/bash
# CPU analysis pass over whatever the sweep has produced so far.
#SBATCH --job-name=skrep --account=co_nilah --partition=savio3_htc --qos=savio_lowprio
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --time=00:30:00
#SBATCH --output=/global/scratch/users/sergiomar10/if-mhc/outputs/skempi_if/logs/analysis_%j.out
#SBATCH --error=/global/scratch/users/sergiomar10/if-mhc/outputs/skempi_if/logs/analysis_%j.out
set -uo pipefail; cd /global/scratch/users/sergiomar10/if-mhc; source jobs/skempi_env.sh
$PY py/skempi_posprofile.py
echo "=== ANALYSIS DONE ==="
