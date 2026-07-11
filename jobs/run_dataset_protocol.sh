#!/usr/bin/env bash
# Run the IF protocol (ProteinMPNN, no Met@P1 + no Pro) on all validated pMHC-TCR structures,
# then quantify per-structure native recovery.
set -uo pipefail
cd /home/ubuntu/if-mhc
PY=/home/ubuntu/miniforge3/envs/esmcba/bin/python
NB=/home/ubuntu/miniforge3/bin/python
DDIR=inputs/pmhc_tcr_dataset
OUT=outputs/dataset_protocol; mkdir -p "$OUT" "$DDIR/valid"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG="$OUT/run.log"
trap 'echo "[$(date)] EXIT $? (dataset_protocol)" >> "$LOG"' EXIT
gpu_wait(){ while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -lt "$1" ]; do sleep 30; done; }

# stage only validated PDBs
$NB - <<'PY' 2>>"$LOG"
import csv, shutil, os
for r in csv.DictReader(open("inputs/pmhc_tcr_dataset/dataset.csv")):
    if r["valid"]=="True":
        shutil.copy(f"inputs/pmhc_tcr_dataset/{r['pdb']}.pdb", f"inputs/pmhc_tcr_dataset/valid/{r['pdb']}.pdb")
print("staged valid PDBs")
PY

echo "[$(date)] parsing dataset" | tee "$LOG"
$PY ProteinMPNN/helper_scripts/parse_multiple_chains.py --input_path="$DDIR/valid" --output_path="$OUT/parsed.jsonl" >>"$LOG" 2>&1
$NB ds_make_jsonls.py "$OUT/parsed.jsonl" "$DDIR/dataset.csv" "$OUT/chain_id.jsonl" "$OUT/omit.jsonl" >>"$LOG" 2>&1

gpu_wait 6000
echo "[$(date)] running ProteinMPNN over dataset (no Met@P1, no Pro, T=0.1, 2000/structure)" | tee -a "$LOG"
$PY ProteinMPNN/protein_mpnn_run.py \
   --jsonl_path "$OUT/parsed.jsonl" --chain_id_jsonl "$OUT/chain_id.jsonl" \
   --omit_AA_jsonl "$OUT/omit.jsonl" --omit_AAs "PX" \
   --out_folder "$OUT" --num_seq_per_target 2000 --batch_size 8 \
   --sampling_temp "0.1" --seed 37 --model_name v_48_020 >>"$LOG" 2>&1

echo "[$(date)] quantifying per-structure recovery" | tee -a "$LOG"
$NB ds_recovery.py "$OUT" "$DDIR/dataset.csv" 2>>"$LOG" | tee "$OUT/recovery_table.txt"
echo "[$(date)] DONE dataset protocol" | tee -a "$LOG"
touch "$OUT/COMPLETE"
