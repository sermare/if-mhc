#!/usr/bin/env bash
# De-novo RFD peptide (peptide REMOVED, generated from noise) into the groove within each
# crystal TCR pose (6AM5 = GIG-pose, 6AMU = DRG-pose). Question: does the TCR pose alone bias
# the de-novo peptide's register (F-pocket swap)? Binder mode: fixed receptor motif + 10-10
# de-novo peptide, groove+TCR hotspots, full diffusion (no partial_T, no provide_seq).
set -uo pipefail
ABS=/home/ubuntu/if-mhc
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh; conda activate SE3nv
export LD_LIBRARY_PATH="$(cat $ABS/RFdiffusion/SE3NV_LDPATH.txt):${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2
OUT=$ABS/outputs/tcrpose_denovo; mkdir -p "$OUT/logs"
NDES="${NDES:-16}"; HOT="D30,A63,A159,A77,A116,A123"
cd "$ABS/RFdiffusion"
run(){ # name  input  contig
  local name=$1 inp=$2 contig=$3
  echo "[$(date)] de-novo $name ndes=$NDES hot=[$HOT]" >"$OUT/logs/$name.log"
  python run_inference.py \
    inference.input_pdb="$inp" \
    inference.output_prefix="$OUT/$name" \
    inference.num_designs="$NDES" \
    "contigmap.contigs=[$contig]" \
    "ppi.hotspot_res=[$HOT]" \
    inference.ckpt_override_path=models/Complex_base_ckpt.pt >>"$OUT/logs/$name.log" 2>&1 \
    && touch "$OUT/logs/$name.DONE" || echo "[$(date)] ERR $name" >>"$OUT/logs/$name.log"
}
# sequential (single GPU already shared with steered MD; two RFD workers would thrash)
run GIGpose "$OUT/6AM5_noPep.pdb" "A1-180/0 B1-100/0 10-10/0 D1-115/0 E1-120/0"
run DRGpose "$OUT/6AMU_noPep.pdb" "A2-180/0 B0-99/0 10-10/0 D2-115/0 E4-120/0"
echo "[$(date)] ALL DONE" >"$OUT/logs/ALL.DONE"
