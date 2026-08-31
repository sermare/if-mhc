#!/usr/bin/env bash
# HARD KILL-SWITCH: remove outputs/skempi_if/HALT to re-enable the fleet.
# This file is sourced (not exec'd) by every keeper and worker, so `exit` here
# stops the sourcing job before it can run work or submit anything further.
# Analysis/packaging jobs set SKEMPI_ANALYSIS=1: they only read finished part
# files and never submit work, so the fleet guard must not block them.
if [ -f /global/scratch/users/sergiomar10/if-mhc/outputs/skempi_if/HALT ] \
   && [ "${SKEMPI_ANALYSIS:-0}" != "1" ]; then
  echo "[skempi_env] HALT flag present -- refusing to start compute fleet"; exit 0
fi
# Shared environment for the SKEMPI TCR/pMHC inverse-folding runs.
# EVERYTHING caches to scratch -- nothing may be written under $HOME.
# Source this at the top of every skempi_* job script.

export ROOT=/global/scratch/users/sergiomar10/if-mhc
export SCRATCH=/global/scratch/users/sergiomar10

# keep every framework's cache off $HOME (Savio home is quota-limited)
export TORCH_HOME=$SCRATCH/torch_cache
export HF_HOME=$SCRATCH/hf_cache
export HUGGINGFACE_HUB_CACHE=$SCRATCH/hf_cache
export XDG_CACHE_HOME=$SCRATCH/.cache
export PIP_CACHE_DIR=$SCRATCH/.cache/pip
export MPLCONFIGDIR=$SCRATCH/.cache/matplotlib
export TMPDIR=$SCRATCH/tmp
mkdir -p "$TORCH_HOME" "$HF_HOME" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$TMPDIR"

export PY=/clusterfs/nilah/sergio/miniconda3/envs/tttppi/bin/python
export PMPNN=$SCRATCH/TCera/ProteinMPNN
export LMPNN=$SCRATCH/tools/LigandMPNN
export SKDIR=$ROOT/inputs/skempi
export OUTDIR=$ROOT/outputs/skempi_if

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUTDIR"
