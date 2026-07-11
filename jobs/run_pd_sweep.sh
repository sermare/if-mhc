#!/usr/bin/env bash
# Local RFdiffusion v1 PARTIAL-DIFFUSION noise sweep (Tamarind's PD task is broken).
# Per native seed: re-diffuse ONLY the peptide (C) inside the fixed complex at 4 partial_T
# levels x 5 designs, L4 hotspots. Contig PRESERVES input order A,B,C,D,E (partial diffusion
# forbids reindexing); provide_seq fixes the receptor sequence, peptide left free.
# Two workers (6AMU, 6AM5) run concurrently to use the L4 efficiently.
set -uo pipefail
ABS=/home/ubuntu/if-mhc
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
conda activate SE3nv
export LD_LIBRARY_PATH="$(cat $ABS/RFdiffusion/SE3NV_LDPATH.txt):${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2
OUT=$ABS/outputs/pd_local; mkdir -p "$OUT"
TS="5 10 20 40"; NDES=5
gpu_wait(){ while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -lt "$1" ]; do sleep 15; done; }

# per-seed: contig (input order preserved), provide_seq (receptor=all but peptide), L4 hotspots
declare -A CONTIG PROVIDE HOT
CONTIG[6AMU]="A2-180/0 B0-99/0 C1-10/0 D2-115/0 E4-120/0"; PROVIDE[6AMU]="0-278,289-519"; HOT[6AMU]="A159,A66,A70,A9,A77,A80,E100,D30,E99"
CONTIG[6AM5]="A1-180/0 B1-100/0 C1-10/0 D1-115/0 E1-120/0"; PROVIDE[6AM5]="0-279,290-524"; HOT[6AM5]="A159,A66,A70,A9,A77,A80,E97,D30,E96"

worker(){
  local pid=$1; local log="$OUT/${pid}.log"
  echo "[$(date)] start PD sweep $pid" >"$log"
  cd "$ABS/RFdiffusion"
  for T in $TS; do
    local pref="$OUT/${pid}_T${T}"
    ls "${pref}_$((NDES-1)).pdb" >/dev/null 2>&1 && { echo "[$(date)] $pid T$T done, skip" >>"$log"; continue; }
    gpu_wait 6000
    echo "[$(date)] $pid partial_T=$T ($NDES designs)" >>"$log"
    python run_inference.py \
      inference.input_pdb="$ABS/inputs/focus_6am/${pid}_trim.pdb" \
      inference.output_prefix="$pref" \
      inference.num_designs=$NDES \
      "contigmap.contigs=[${CONTIG[$pid]}]" \
      "contigmap.provide_seq=[${PROVIDE[$pid]}]" \
      diffuser.partial_T=$T \
      "ppi.hotspot_res=[${HOT[$pid]}]" \
      inference.ckpt_override_path=models/Complex_base_ckpt.pt >>"$log" 2>&1
  done
  echo "[$(date)] DONE $pid" >>"$log"; touch "$OUT/${pid}.DONE"
}

worker 6AMU &
worker 6AM5 &
wait
echo "[$(date)] PD sweep complete: $(ls $OUT/*_T*.pdb 2>/dev/null | grep -v _split | wc -l) designs" | tee "$OUT/COMPLETE"
