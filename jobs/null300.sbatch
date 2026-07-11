#!/bin/bash
# +300 EXTRA negative controls (150 6AM5 + 150 6AMU), pure null: no hotspots, no template,
# de-novo 10-mer, T=30. Independent output dir so this is a clean +300 ON TOP of the null0
# cell already inside allcond150. Same short idempotent worker, savio3_gpu / co_nilah lowprio.
#SBATCH --account=co_nilah
#SBATCH --partition=savio3_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --time=00:50:00
#SBATCH --job-name=null300
#SBATCH --output=/global/scratch/users/sergiomar10/if-mhc/outputs/null_extra300/logs/null300_%A_%a.out
#SBATCH --error=/global/scratch/users/sergiomar10/if-mhc/outputs/null_extra300/logs/null300_%A_%a.err
export SPEC=/global/scratch/users/sergiomar10/if-mhc/jobs/null_extra300_spec.tsv
export OUT_DIR=/global/scratch/users/sergiomar10/if-mhc/outputs/null_extra300
export BATCH="${BATCH:-30}" SOFT_MIN="${SOFT_MIN:-42}"
bash /global/scratch/users/sergiomar10/if-mhc/jobs/allcond150_worker.sh
