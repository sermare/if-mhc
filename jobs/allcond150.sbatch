#!/bin/bash
# Short (<1h) idempotent GPU worker array for the allcond150 top-up campaign.
# Partition + array range are supplied on the sbatch command line (see submit_allcond150.sh),
# so this one file serves both savio3_gpu and savio2_1080ti.
#SBATCH --account=co_nilah
#SBATCH --qos=savio_lowprio
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --time=00:50:00
#SBATCH --job-name=ac150w
#SBATCH --output=/global/scratch/users/sergiomar10/if-mhc/outputs/allcond150/logs/ac150w_%A_%a.out
#SBATCH --error=/global/scratch/users/sergiomar10/if-mhc/outputs/allcond150/logs/ac150w_%A_%a.err

export BATCH="${BATCH:-30}" SOFT_MIN="${SOFT_MIN:-42}"
bash /global/scratch/users/sergiomar10/if-mhc/jobs/allcond150_worker.sh
