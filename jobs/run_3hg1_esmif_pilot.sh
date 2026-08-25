#!/usr/bin/env bash
# ESM-IF1 (esm_if1_gvp4_t16_142M_UR50) pilot campaign on 3HG1 (MEL5 TCR / MART-1 / ELAGIGILTV),
# matching the existing vanilla/noMHC ProteinMPNN 3HG1 convention (chain C = peptide designed,
# A+B+D+E fixed context) so results are directly comparable -- see MATCHED_TCR_TRACKING.md.
#
# Small pilot scope (5-10k seqs, T=0.1) by design: first cross-model comparison run, not yet at the
# 40-50k scale of the ProteinMPNN campaigns.
#
# Uses py/esmif_sample_panel.py's batched sampler (B=8 default): encodes the ~866-residue complex ONCE per
# batch instead of once per sequence (the naive esm.inverse_folding.multichain_util.
# sample_sequence_in_complex loop is ~0.6s/seq; batching brings it to ~0.29s/seq). esm repo cloned to
# esm_repo/ (pip's fair-esm 2.0.0 doesn't ship the inverse_folding submodule); torch_geometric
# installed into esmcba (pure-python wheel, no CUDA-version-specific build needed for the scatter ops
# this model uses).
set -uo pipefail
cd /home/ubuntu/if-mhc; ABS=/home/ubuntu/if-mhc
PY=/home/ubuntu/miniforge3/envs/esmcba/bin/python
OUT="$ABS/outputs/esmif_3hg1_pilot"; mkdir -p "$OUT/seqs"
NSEQ="${NSEQ:-10000}"; BATCH="${BATCH:-8}"; TEMP="${TEMP:-0.1}"
LOG="$OUT/run.log"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2
export TORCH_HOME="$ABS/models_cache/torch_hub"
gpu_wait(){ while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -lt "$1" ]; do sleep 20; done; }
exec 9>"$OUT/runner.lock"; flock -n 9 || { echo "another runner holds lock — exit"; exit 0; }
echo $$ >"$OUT/runner.pid"; trap 'rm -f "$OUT/runner.pid"' EXIT

echo "[$(date '+%F %T')] === esmif_3hg1_pilot: MEL5 TCR/MART-1 x $NSEQ seqs, T=$TEMP ===" | tee -a "$LOG"
gpu_wait 12000

$PY py/esmif_sample_panel.py \
    --pdb inputs/pmhc_tcr_dataset/3HG1.pdb \
    --chains A B C D E \
    --target-chain C \
    --num-seqs "$NSEQ" \
    --batch-size "$BATCH" \
    --temperature "$TEMP" \
    --out "$OUT/seqs/3HG1.fa" >>"$LOG" 2>&1

got=$(grep -c "^>" "$OUT/seqs/3HG1.fa" 2>/dev/null || echo 0)
echo "[$(date '+%F %T')] esmif_3hg1_pilot done: $got/$NSEQ seqs" | tee -a "$LOG"
if [ "${got:-0}" -ge "$NSEQ" ]; then
  touch "$OUT/COMPLETE"
  echo "[$(date '+%F %T')] === esmif_3hg1_pilot COMPLETE ===" | tee -a "$LOG"
else
  echo "[$(date '+%F %T')] WARN short of target ($got/$NSEQ) -- rerun to resume" | tee -a "$LOG"
fi
