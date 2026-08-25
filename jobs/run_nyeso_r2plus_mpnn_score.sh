#!/usr/bin/env bash
# Batched ProteinMPNN + noMHC scoring of the 100k NY-ESO-1 R2-R4 peptides that the low-lr (NEWCNN)
# embedding covers, on the 2P5E backbone. Uses py/batch_score_peptides_mpnn.py (one model load,
# one featurization) -- protein_mpnn_run.py --score_only would reload per sequence.
set -uo pipefail
ABS=/home/ubuntu/if-mhc; cd "$ABS"
PY=/home/ubuntu/miniforge3/envs/esmcba/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PEPS="$ABS/outputs/analysis/nyeso1_r2plus_train_peps_1G4c58c61.npy"
for W in vanilla nomhc; do
  OUT="$ABS/outputs/analysis/nyeso1_r2plus_mpnn_${W}_scores_15k.npz"
  [ -f "$OUT" ] && { echo "$W already done"; continue; }
  echo "[$(date '+%F %T')] === $W ==="
  "$PY" py/batch_score_peptides_mpnn.py --weights "$W" --peptides_npy "$PEPS" \
        --out "$OUT" --batch 16 --n_orders 1 --limit 15000
done
echo "[$(date '+%F %T')] ALL DONE"
