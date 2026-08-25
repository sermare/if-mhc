#!/usr/bin/env bash
# LigandMPNN (ligandmpnn_v_32_010_25, the package's default checkpoint) pilot campaign on 3HG1
# (MEL5 TCR / MART-1 / ELAGIGILTV), matching the existing vanilla/noMHC ProteinMPNN and ESM-IF 3HG1
# conventions (chain C = peptide designed, A+B+D+E fixed context) -- see MATCHED_TCR_TRACKING.md.
#
# Small pilot scope (10k seqs, T=0.1), matching jobs/run_3hg1_esmif_pilot.sh's scope for a fair
# first cross-model comparison.
#
# LigandMPNN repo cloned to LigandMPNN/ (github.com/dauparas/LigandMPNN), reusing esmcba's existing
# torch/numpy rather than the repo's pinned torch==2.2.1/numpy==1.23.5 (would have downgraded the
# shared env). Extra deps installed: prody, ml-collections, dm-tree. The vendored openfold/ package
# (used by sc_utils.py, imported unconditionally by run.py even though we never pack side chains) had
# 3 dead `np.int` references (removed in numpy>=1.24) patched to `np.int64` in
# openfold/np/residue_constants.py -- a plain deprecated-alias fix, not a logic change.
#
# LigandMPNN treats crystallographic HETATMs (buffer ions, glycerol, etc.) as "ligand context" by
# default (ligand_mpnn_use_atom_context=1, the default) -- 22 such atoms were found in 3HG1's PDB.
# This is expected/harmless for our purposes (protein-protein recovery), not something to suppress.
set -uo pipefail
cd /home/ubuntu/if-mhc; ABS=/home/ubuntu/if-mhc
PY=/home/ubuntu/miniforge3/envs/esmcba/bin/python
OUT="$ABS/outputs/ligandmpnn_3hg1_pilot"; mkdir -p "$OUT"
NSEQ="${NSEQ:-10000}"; BATCH="${BATCH:-64}"; TEMP="${TEMP:-0.1}"
NBATCHES=$(( (NSEQ + BATCH - 1) / BATCH ))
LOG="$OUT/run.log"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2
gpu_wait(){ while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -lt "$1" ]; do sleep 20; done; }
exec 9>"$OUT/runner.lock"; flock -n 9 || { echo "another runner holds lock — exit"; exit 0; }
echo $$ >"$OUT/runner.pid"; trap 'rm -f "$OUT/runner.pid"' EXIT

count_seqs(){ [ -f "$OUT/seqs/3HG1.fa" ] && grep -c "^>" "$OUT/seqs/3HG1.fa" 2>/dev/null || echo 0; }
have=$(count_seqs); have=${have:-0}
if [ "$have" -ge "$NSEQ" ]; then
  echo "[$(date '+%F %T')] already at $have/$NSEQ -- skip" | tee -a "$LOG"
  touch "$OUT/COMPLETE"
  exit 0
fi

echo "[$(date '+%F %T')] === ligandmpnn_3hg1_pilot: MEL5 TCR/MART-1 x $NSEQ seqs, T=$TEMP (have=$have) ===" | tee -a "$LOG"
gpu_wait 3000

cd LigandMPNN
"$PY" run.py \
    --model_type "ligand_mpnn" \
    --checkpoint_ligand_mpnn "./model_params/ligandmpnn_v_32_010_25.pt" \
    --pdb_path "$ABS/inputs/pmhc_tcr_dataset/3HG1.pdb" \
    --parse_these_chains_only "A,B,C,D,E" \
    --chains_to_design "C" \
    --out_folder "$OUT" \
    --batch_size "$BATCH" \
    --number_of_batches "$NBATCHES" \
    --temperature "$TEMP" \
    --seed 37 >>"$LOG" 2>&1
cd "$ABS"

got=$(count_seqs)
echo "[$(date '+%F %T')] ligandmpnn_3hg1_pilot done: $got seqs (requested $((BATCH*NBATCHES)), target $NSEQ)" | tee -a "$LOG"
if [ "${got:-0}" -ge "$NSEQ" ]; then
  touch "$OUT/COMPLETE"
  echo "[$(date '+%F %T')] === ligandmpnn_3hg1_pilot COMPLETE ===" | tee -a "$LOG"
fi

# run.py unconditionally dumps a full backbone PDB per design into backbones/ (identical backbone
# every time -- LigandMPNN doesn't touch it) and an empty packed/ (side-chain packing wasn't
# requested). Both are pure redundant disk cost on a disk-constrained box; the one true backbone is
# already at inputs/pmhc_tcr_dataset/3HG1.pdb. Confirmed wasteful: 10k designs -> 2.6GB of identical
# backbone copies (2026-07-29).
rm -rf "$OUT/backbones" "$OUT/packed"
echo "[$(date '+%F %T')] cleaned up backbones/ + packed/ (redundant per-design backbone dumps)" | tee -a "$LOG"
