#!/bin/bash
# CPU prep for the phase-2 pmhc25 dataset: stage both arms from
# inputs/pmhc_tcr_dataset, then build and verify the ProteinMPNN input jsons.
#SBATCH --job-name=p25prep
#SBATCH --account=co_nilah
#SBATCH --partition=savio3_htc
#SBATCH --qos=savio_lowprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=/global/scratch/users/sergiomar10/if-mhc/outputs/skempi_if/logs/p25prep_%j.out
#SBATCH --error=/global/scratch/users/sergiomar10/if-mhc/outputs/skempi_if/logs/p25prep_%j.out
set -uo pipefail
cd /global/scratch/users/sergiomar10/if-mhc
source jobs/skempi_env.sh

echo "=== stage pmhc25 arms ==="
$PY py/pmhc25_prep.py || exit 1

echo
echo "=== build + verify ProteinMPNN inputs for pmhc25 ==="
for arm in full notcr; do
  $PY py/mpnn_prep.py --dataset pmhc25 --arm $arm --python "$PY" || exit 1
done

echo
echo "=== phase-2 ledgers ==="
for t in 0.3; do
  for dsname in skempi pmhc25; do
    SK_DATASET=$dsname SK_TEMP=$t $PY py/skempi_units.py --status 2>/dev/null | tail -2
  done
done
echo "=== P25 PREP DONE $(date) ==="
