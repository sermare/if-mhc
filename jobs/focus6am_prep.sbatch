#!/bin/bash
# CPU prep for the 6AM5 / 6AMU reference crystals: stage both arms and build the
# ProteinMPNN inputs. Runs alongside the live SKEMPI fleet; touches nothing it uses.
#SBATCH --job-name=f6amprep
#SBATCH --account=co_nilah
#SBATCH --partition=savio3_htc
#SBATCH --qos=savio_lowprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=/global/scratch/users/sergiomar10/if-mhc/outputs/skempi_if/logs/f6amprep_%j.out
#SBATCH --error=/global/scratch/users/sergiomar10/if-mhc/outputs/skempi_if/logs/f6amprep_%j.out
set -uo pipefail
cd /global/scratch/users/sergiomar10/if-mhc
source jobs/skempi_env.sh

echo "=== stage focus6am arms (6AM5, 6AMU) ==="
$PY py/pmhc25_prep.py --dataset focus6am || exit 1

echo
echo "=== build + verify ProteinMPNN inputs ==="
for arm in full notcr; do
  $PY py/mpnn_prep.py --dataset focus6am --arm $arm --python "$PY" || exit 1
done

echo
echo "=== ledgers ==="
for t in 0.1 0.3; do
  SK_DATASET=focus6am SK_TEMP=$t $PY py/skempi_units.py --status 2>/dev/null | tail -2
done
echo "=== F6AM PREP DONE $(date) ==="
