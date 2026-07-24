#!/bin/bash
# Short idempotent GPU worker array for the reviewer-driven experiment campaign.
# SPEC + OUT_DIR are passed via --export; partition + array range on the command line.
#SBATCH --account=co_nilah
#SBATCH --qos=savio_lowprio
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --time=00:50:00
#SBATCH --job-name=expw
#SBATCH --output=/global/scratch/users/sergiomar10/if-mhc/outputs/exp_logs/expw_%A_%a.out
#SBATCH --error=/global/scratch/users/sergiomar10/if-mhc/outputs/exp_logs/expw_%A_%a.err
export BATCH="${BATCH:-24}" SOFT_MIN="${SOFT_MIN:-42}"
bash /global/scratch/users/sergiomar10/if-mhc/jobs/exp_worker.sh
