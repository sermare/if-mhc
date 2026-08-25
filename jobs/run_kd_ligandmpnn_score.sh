#!/usr/bin/env bash
# LigandMPNN autoregressive scoring of the 51 KD-tested NY-ESO-1/1G4c58c61 peptides, each threaded
# onto the 2P5E chain-C backbone (py/thread_kd_peptides_pdb.py). Mirrors the ProteinMPNN score_only
# runs in outputs/kd_scoring/{vanilla,nomhc}. $OUT resolved to absolute BEFORE the cd into
# LigandMPNN/ -- a relative path breaks the log redirect there (see jobs/run_2p5e_ligandmpnn.sh).
set -uo pipefail
ABS=/home/ubuntu/if-mhc; cd "$ABS"
PY=/home/ubuntu/miniforge3/envs/esmcba/bin/python
OUT="$ABS/outputs/kd_scoring/ligandmpnn/score_only"
PDBS="$ABS/outputs/kd_scoring/ligandmpnn/pdbs"
LOG="$ABS/outputs/kd_scoring/ligandmpnn/score.log"
mkdir -p "$OUT"; : > "$LOG"
n=0; total=$(ls "$PDBS"/*.pdb | wc -l)
cd LigandMPNN
for f in "$PDBS"/*.pdb; do
  pep=$(basename "$f" .pdb); n=$((n+1))
  [ -f "$OUT/$pep.pt" ] && { echo "[$n/$total] $pep cached" >>"$LOG"; continue; }
  "$PY" score.py --model_type ligand_mpnn \
      --checkpoint_ligand_mpnn "./model_params/ligandmpnn_v_32_010_25.pt" \
      --pdb_path "$f" --parse_these_chains_only "A,B,C,D,E" --chains_to_design "C" \
      --out_folder "$OUT" --autoregressive_score 1 --use_sequence 1 \
      --batch_size 1 --number_of_batches 10 --seed 41 >/dev/null 2>&1
  [ -f "$OUT/$pep.pt" ] && echo "[$n/$total] $pep ok" >>"$LOG" || echo "[$n/$total] $pep FAILED" >>"$LOG"
done
cd "$ABS"
echo "done: $(ls "$OUT"/*.pt 2>/dev/null | wc -l)/$total scored" >>"$LOG"
