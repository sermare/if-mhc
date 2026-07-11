#!/usr/bin/env bash
# B3 — anchor-conditioned de-novo RFdiffusion. Motif-scaffold: FIX the C-terminal anchor residue in the
# F-pocket (p10 from 6AM5 = GIG register; p9 from 6AMU = DRG register) and DE-NOVO diffuse the rest of the
# 10-mer around it (full T, not partial). Receptor MHC+b2m+TCR fixed; F-pocket hotspots. Question: can RFD
# build a COHERENT backbone around a specified anchor, and does it land at the intended register on the
# basin map? Run p9-target and p10-target separately.  Env: OUT, NDES (per target), TAG_ONLY(optional)
set -uo pipefail
ABS=/home/ubuntu/if-mhc
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh; conda activate SE3nv
export LD_LIBRARY_PATH="$(cat $ABS/RFdiffusion/SE3NV_LDPATH.txt):${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2
OUT="${OUT:?set OUT}"; [[ "$OUT" = /* ]] || OUT="$ABS/$OUT"   # make absolute (we cd into RFdiffusion)
NDES="${NDES:-250}"; CKPT=models/Complex_base_ckpt.pt
FPHOT="A77,A80,A84,A116,A123,A143"
mkdir -p "$OUT/logs"; cd "$ABS/RFdiffusion"

# target: name  input_pdb  contig (anchor fixed, rest diffused)
run_target(){
  local name="$1" input="$2" contig="$3"
  local log="$OUT/logs/${name}.log"
  echo "[$(date)] B3 $name contig=$contig hot=[$FPHOT] ndes=$NDES" >"$log"
  ls "$OUT/${name}_$((NDES-1)).pdb" >/dev/null 2>&1 && { echo "done, skip" >>"$log"; return; }
  python run_inference.py \
    inference.input_pdb="$input" \
    inference.output_prefix="$OUT/${name}" \
    inference.ckpt_override_path="$CKPT" \
    "contigmap.contigs=[$contig]" \
    "ppi.hotspot_res=[$FPHOT]" \
    inference.num_designs="$NDES" \
    denoiser.noise_scale_ca=1.0 denoiser.noise_scale_frame=1.0 \
    >>"$log" 2>&1
  echo "[$(date)] $name EXIT $?" >>"$log"
}

# TCR dropped (register is a peptide–MHC property; halves the complex → ~2-4x faster). MHC+b2m context.
# p10-target = GIG register: fix native p10 (6AM5) in F-pocket, de-novo p1-p9
P10="A1-180/0 B1-100/0 9-9/C10-10"
# p9-target = DRG register: fix native p9 (6AMU) in F-pocket, de-novo p1-p8 + p10
P9="A2-180/0 B0-99/0 8-8/C9-9/1-1"

[ "${TAG_ONLY:-}" != "p9" ]  && run_target p10_GIGanchor "$ABS/inputs/focus_6am/6AM5_trim.pdb" "$P10"
[ "${TAG_ONLY:-}" != "p10" ] && run_target p9_DRGanchor  "$ABS/inputs/focus_6am/6AMU_trim.pdb" "$P9"
echo "B3_DONE"
