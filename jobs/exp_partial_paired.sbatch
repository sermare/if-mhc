#!/bin/bash
#SBATCH --account=co_nilah
#SBATCH --qos=savio_lowprio
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --time=00:50:00
#SBATCH --job-name=pdpair
#SBATCH --output=/global/scratch/users/sergiomar10/if-mhc/outputs/exp_partial_paired/logs/%A_%a.out
export NPAIR="${NPAIR:-24}"
bash /global/scratch/users/sergiomar10/if-mhc/jobs/exp_partial_paired.sh
