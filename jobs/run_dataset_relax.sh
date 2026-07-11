#!/usr/bin/env bash
# Relaxation-ensemble protocol across the 22 pMHC-TCR structures:
# OpenMM relax (N snapshots each) -> ProteinMPNN (no Met@P1 + no Pro) -> relaxed recovery vs single.
set -uo pipefail
cd /home/ubuntu/if-mhc
OMM=/home/ubuntu/miniforge3/envs/openmm/bin/python
PY=/home/ubuntu/miniforge3/envs/esmcba/bin/python
NB=/home/ubuntu/miniforge3/bin/python
OUT=outputs/dataset_relax; mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 OPENMM_CPU_THREADS=3
LOG="$OUT/run.log"
trap 'echo "[$(date)] EXIT $? (dataset_relax)" >> "$LOG"' EXIT
gpu_wait(){ while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -lt "$1" ]; do sleep 20; done; }

echo "[$(date)] OpenMM relax of 22 dataset structures (8 snapshots each)" | tee "$LOG"
$OMM ds_relax_openmm.py 8 >>"$LOG" 2>&1
nsnap=$(ls "$OUT"/snapshots/*.pdb 2>/dev/null | wc -l)
echo "[$(date)] snapshots: $nsnap" | tee -a "$LOG"
[ "$nsnap" -lt 1 ] && { echo "no snapshots produced"; exit 1; }

echo "[$(date)] parsing + building per-snapshot jsonls" | tee -a "$LOG"
$PY ProteinMPNN/helper_scripts/parse_multiple_chains.py --input_path="$OUT/snapshots" --output_path="$OUT/parsed.jsonl" >>"$LOG" 2>&1
$NB ds_relax_jsonls.py "$OUT/parsed.jsonl" inputs/pmhc_tcr_dataset/dataset.csv "$OUT/chain_id.jsonl" "$OUT/omit.jsonl" >>"$LOG" 2>&1

gpu_wait 6000
echo "[$(date)] ProteinMPNN over relaxed ensemble (no Met@P1, no Pro, T=0.1, 500/snapshot)" | tee -a "$LOG"
$PY ProteinMPNN/protein_mpnn_run.py --jsonl_path "$OUT/parsed.jsonl" --chain_id_jsonl "$OUT/chain_id.jsonl" \
   --omit_AA_jsonl "$OUT/omit.jsonl" --omit_AAs "PX" \
   --out_folder "$OUT" --num_seq_per_target 500 --batch_size 8 \
   --sampling_temp "0.1" --seed 37 --model_name v_48_020 >>"$LOG" 2>&1

echo "[$(date)] relaxed recovery vs single-structure" | tee -a "$LOG"
$NB ds_relax_recovery.py "$OUT" inputs/pmhc_tcr_dataset/dataset.csv 2>>"$LOG" | tee "$OUT/recovery_relaxed_table.txt"
echo "[$(date)] DONE dataset_relax" | tee -a "$LOG"
touch "$OUT/COMPLETE"
