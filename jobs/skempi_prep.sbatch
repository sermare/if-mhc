#!/bin/bash
# CPU prep for the SKEMPI sweep: stage both arms from the raw PDBs, then build
# and verify the ProteinMPNN input jsons. Fast, no GPU, schedules in seconds.
#SBATCH --job-name=skprep
#SBATCH --account=co_nilah
#SBATCH --partition=savio3_htc
#SBATCH --qos=savio_lowprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=/global/scratch/users/sergiomar10/if-mhc/outputs/skempi_if/logs/prep_%j.out
#SBATCH --error=/global/scratch/users/sergiomar10/if-mhc/outputs/skempi_if/logs/prep_%j.out
set -uo pipefail
cd /global/scratch/users/sergiomar10/if-mhc
source jobs/skempi_env.sh
mkdir -p "$OUTDIR/logs"

echo "=== stage arms from raw PDBs ==="
$PY py/skempi_prep.py || exit 1

echo
echo "=== build + verify ProteinMPNN inputs ==="
for arm in full notcr; do
  $PY py/mpnn_prep.py --arm $arm --python "$PY" || exit 1
done

echo
echo "=== unit ledger ==="
$PY py/skempi_units.py --status
echo "=== PREP DONE $(date) ==="
