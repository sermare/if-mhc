#!/bin/bash
# Full analysis for one (dataset, temperature) phase. Reads finished part files
# only -- submits no work -- so it sets SKEMPI_ANALYSIS=1 to pass the fleet HALT
# guard. Drive with SK_DATASET / SK_TEMP.
#SBATCH --job-name=an_phase
#SBATCH --account=co_nilah
#SBATCH --partition=savio3_htc
#SBATCH --qos=savio_lowprio
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/global/scratch/users/sergiomar10/if-mhc/outputs/skempi_if/logs/anphase_%j.out
#SBATCH --error=/global/scratch/users/sergiomar10/if-mhc/outputs/skempi_if/logs/anphase_%j.out
set -uo pipefail
export SKEMPI_ANALYSIS=1
cd /global/scratch/users/sergiomar10/if-mhc
source jobs/skempi_env.sh
export SK_DATASET="${SK_DATASET:-skempi}" SK_TEMP="${SK_TEMP:-0.3}"
echo "################ ANALYSIS $SK_DATASET / T=$SK_TEMP ################"
echo "### 1. panel conclusions"; $PY py/panel_replication.py
echo; echo "### 2. follow-ups"; $PY py/panel_replication2.py
echo; echo "### 3. per-position profile"; $PY py/skempi_posprofile.py
echo; echo "### 4. build peptide table for external scoring"; $PY py/score_peptides_build.py
echo "=== ANPHASE DONE ==="
