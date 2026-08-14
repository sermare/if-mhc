#!/usr/bin/env bash
#SBATCH --job-name=sk_smoke
#SBATCH --account=co_nilah
#SBATCH --partition=savio3_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/global/scratch/users/sergiomar10/if-mhc/outputs/skempi_if/logs/smoke_%j.out
#SBATCH --error=/global/scratch/users/sergiomar10/if-mhc/outputs/skempi_if/logs/smoke_%j.out
set -uo pipefail
cd /global/scratch/users/sergiomar10/if-mhc
source jobs/skempi_env.sh
BIG=2AK4_ABC_DE        # largest complex: 833 res, 13-mer epitope -> worst case
SMALL=2OI9_AQ_BC       # smallest: 405 res, 9-mer -- cheap enough to validate on CPU

echo "node=$(hostname)  $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# timings first: they set the chunk size, and must not be starved by the
# CPU-side validation below
echo
echo "=== 1. per-model timing probe: $BIG arm=full, 500 seqs each ==="
for m in esmif proteinmpnn proteinmpnn_nomhc ligandmpnn; do
  echo "--- $m ---"
  s=$SECONDS
  $PY py/skempi_run_unit.py --model $m --arm full --complex $BIG \
      --chunk 99 --nseq 500 --temp 0.1 2>&1 | tail -6
  echo "    [$m] 500 seqs in $((SECONDS-s))s -> 10k projected $(( (SECONDS-s)*20 ))s"
  rm -f "$OUTDIR/$m/parts/${BIG}__full__c99.csv"
done

echo
echo "=== 2. batched ESM-IF sampler vs stock sample_sequence_in_complex ($SMALL) ==="
# validation already passed (job 37455424); skip on re-probes
[ "${SKIP_VALIDATE:-0}" = 1 ] || $PY py/esmif_validate.py --complex $SMALL --arm full --n 24
echo "validate exit=$?"

echo
echo "=== 3. HOME must be untouched ==="
du -sh "$HOME/.cache/torch" 2>/dev/null || echo "no ~/.cache/torch -- good"
echo "=== SMOKE DONE $(date) ==="
