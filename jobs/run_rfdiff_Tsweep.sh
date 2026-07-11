#!/usr/bin/env bash
# RFdiffusion-ONLY noise/temperature sweep (NO ProteinMPNN tail).
# De novo 10mer peptide (matches native chain C length) in the HLA groove;
# MHC (A,B) + TCR (D,E) held as fixed motif,
# dual-interface (groove + TCR) hotspots. Built from run_rfdiff_campaign.sh's diffusion block.
# Sweeps T = noise_scale_ca = noise_scale_frame over {0.2,0.4,0.8,1.0} for 6AM5 and 6AMU.
set -uo pipefail
ABS=/home/ubuntu/if-mhc
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
conda activate SE3nv
export LD_LIBRARY_PATH="$(cat $ABS/RFdiffusion/SE3NV_LDPATH.txt 2>/dev/null):${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2
OUT=$ABS/outputs/rfdiff_Tsweep; mkdir -p "$OUT"
LOG="$OUT/campaign.log"
TS="0.2 0.4 0.8 1.0"          # noise scale (ca & frame) sweep
NDES=20                        # designs per (seed,T) condition
gpu_wait(){ while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -lt "$1" ]; do sleep 20; done; }

# per-seed: contigs (A,B,D,E fixed motif + freely generated 10mer) and dual-interface hotspots
declare -A CONTIG HOT
CONTIG[6AM5]="A1-180/0 B1-100/0 D1-115/0 E1-120/0 10-10"; HOT[6AM5]="A9,A63,A66,A77,A80,A116,A143,E97,D30"
CONTIG[6AMU]="A2-180/0 B0-99/0 D2-115/0 E4-120/0 10-10"; HOT[6AMU]="A9,A63,A66,A77,A80,A116,A143,E100,D30"

echo "[$(date)] RFDIFF T-sweep start (seeds: 6AM5 6AMU; T: $TS; ndes=$NDES)" | tee "$LOG"
cd "$ABS/RFdiffusion"
for pid in 6AM5 6AMU; do
  for T in $TS; do
    pref="$OUT/${pid}_T${T}"
    if ls "${pref}_$((NDES-1)).pdb" >/dev/null 2>&1; then
      echo "[$(date)] $pid T=$T already complete, skip" | tee -a "$LOG"; continue
    fi
    gpu_wait 6000
    echo "[$(date)] $pid T=$T ($NDES designs)" | tee -a "$LOG"
    python run_inference.py \
      inference.input_pdb="$ABS/inputs/focus_6am/${pid}_trim.pdb" \
      inference.output_prefix="$pref" \
      inference.num_designs=$NDES \
      "contigmap.contigs=[${CONTIG[$pid]}]" \
      "ppi.hotspot_res=[${HOT[$pid]}]" \
      denoiser.noise_scale_ca=$T \
      denoiser.noise_scale_frame=$T \
      inference.ckpt_override_path=models/Complex_base_ckpt.pt >>"$LOG" 2>&1
    n=$(ls "${pref}"_*.pdb 2>/dev/null | grep -v _split | wc -l)
    echo "[$(date)] $pid T=$T -> $n backbones" | tee -a "$LOG"
  done
done
tot=$(ls "$OUT"/*_T*.pdb 2>/dev/null | grep -v _split | wc -l)
echo "[$(date)] RFDIFF T-sweep DONE: $tot backbones total" | tee -a "$LOG"
touch "$OUT/COMPLETE"
