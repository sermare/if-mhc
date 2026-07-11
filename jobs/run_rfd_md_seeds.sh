#!/usr/bin/env bash
# RFD partial diffusion seeded from NATIVE MD (300/370K) frames. Receptor (MHC+b2m) fixed, peptide C
# diffused at partial_T, F-pocket hotspots. Q: does seeding from thermally-excursed MD frames (esp 370K,
# best→OTHER ~2.1Å) + conditioning bridge the register, where crystal/relaxed seeds never did?
# PAR workers concurrent (fill the GPU, as requested). Env: OUT, T(20), NDES(2), PAR(3)
set -uo pipefail
ABS=/home/ubuntu/if-mhc
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh; conda activate SE3nv
export LD_LIBRARY_PATH="$(cat $ABS/RFdiffusion/SE3NV_LDPATH.txt):${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2
OUT="${OUT:?set OUT}"; [[ "$OUT" = /* ]] || OUT="$ABS/$OUT"; mkdir -p "$OUT/logs"
T="${T:-20}"; NDES="${NDES:-2}"; PAR="${PAR:-3}"; FPHOT="A77,A80,A84,A116,A123,A143"
cd "$ABS/RFdiffusion"
i=0
for pdb in ${SEEDGLOB:-$ABS/outputs/md_seeds/*.pdb}; do
  name=$(basename "$pdb" .pdb)
  ls "$OUT/${name}_$((NDES-1)).pdb" >/dev/null 2>&1 && { echo "$name done, skip"; continue; }
  ( python run_inference.py \
      inference.input_pdb="$pdb" \
      inference.output_prefix="$OUT/${name}" \
      inference.ckpt_override_path=models/Complex_base_ckpt.pt \
      'contigmap.contigs=[A1-180/0 B1-100/0 C1-10/0]' \
      'contigmap.provide_seq=[0-279]' \
      "ppi.hotspot_res=[$FPHOT]" \
      diffuser.partial_T="$T" inference.num_designs="$NDES" \
      > "$OUT/logs/${name}.log" 2>&1 ) &
  i=$((i+1))
  [ $(( i % PAR )) -eq 0 ] && wait     # concurrency gate: PAR at a time
done
wait
echo "RFD_MD_SEEDS_DONE"
