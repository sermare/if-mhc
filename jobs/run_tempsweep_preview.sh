#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/if-mhc
PY=/home/ubuntu/miniforge3/envs/esmcba/bin/python
NB=/home/ubuntu/miniforge3/bin/python
OUT=outputs/mpnn_tempsweep
mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG="$OUT/preview.log"
trap 'echo "[$(date)] EXIT code $? (preview)" >> "$LOG"' EXIT

echo "[$(date)] PREVIEW sweep: 160 seqs x 7 temps @ batch4 (fits beside A+B)" | tee "$LOG"
$PY -u ProteinMPNN/protein_mpnn_run.py \
   --jsonl_path outputs/mpnn/parsed_chains.jsonl \
   --chain_id_jsonl outputs/mpnn/assigned_chains.jsonl \
   --out_folder "$OUT" \
   --num_seq_per_target 160 --batch_size 4 \
   --sampling_temp "0.1 0.15 0.2 0.3 0.5 0.7 1.0" \
   --seed 41 --model_name v_48_020 >> "$LOG" 2>&1

echo "[$(date)] preview generation done -> rebuilding notebook" | tee -a "$LOG"
$NB build_notebook.py >> "$LOG" 2>&1
$NB -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1800 --ExecutePreprocessor.kernel_name=python3 \
    peptide_design_analysis.ipynb >> "$LOG" 2>&1
echo "[$(date)] DONE preview (notebook now shows 7 temperatures)" | tee -a "$LOG"
touch "$OUT/PREVIEW_DONE"
