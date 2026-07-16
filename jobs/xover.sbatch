#!/bin/bash
# N-terminal-templated crossover experiment: shared N-term templated, divergent C-term free.
# Own- and cross-register hotspot arms. Same short idempotent worker, savio3_gpu / co_nilah lowprio.
#SBATCH --account=co_nilah
#SBATCH --partition=savio3_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --time=00:50:00
#SBATCH --job-name=xoverw
#SBATCH --output=/global/scratch/users/sergiomar10/if-mhc/outputs/xover/logs/xoverw_%A_%a.out
#SBATCH --error=/global/scratch/users/sergiomar10/if-mhc/outputs/xover/logs/xoverw_%A_%a.err
export SPEC=/global/scratch/users/sergiomar10/if-mhc/jobs/xover_spec.tsv
export OUT_DIR=/global/scratch/users/sergiomar10/if-mhc/outputs/xover
export BATCH="${BATCH:-30}" SOFT_MIN="${SOFT_MIN:-42}"
bash /global/scratch/users/sergiomar10/if-mhc/jobs/allcond150_worker.sh
