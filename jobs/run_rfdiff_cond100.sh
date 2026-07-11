#!/usr/bin/env bash
# RFdiffusion-ONLY conditioning campaign (NO ProteinMPNN tail), 100 designs/crystal/config.
# De novo 10mer peptide (native chain-C length) in the HLA groove; MHC(A,B)+TCR(D,E) fixed motif.
# Default noise (T=1.0). Two conditioning sets, two crystals (TCR residues differ per crystal):
#   condA  pkt3+TCR3  : A9,A77,A80 + 3 TCR  -> 6AMU E100,E99,D30 / 6AM5 E97,E96,D30
#   condB  Nterm+TCR1 : A159,A66,A70 (N-term anchors) + D-chain TCR -> 6AMU D30,D93 / 6AM5 D30
set -uo pipefail
ABS=/home/ubuntu/if-mhc
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
conda activate SE3nv
export LD_LIBRARY_PATH="$(cat $ABS/RFdiffusion/SE3NV_LDPATH.txt 2>/dev/null):${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2
OUT=$ABS/outputs/rfdiff_cond100; mkdir -p "$OUT"
LOG="$OUT/campaign.log"
NDES=100                       # designs per (config,crystal)
gpu_wait(){ while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -lt "$1" ]; do sleep 20; done; }

# fixed motif contigs (A,B,D,E held + freely generated 10mer), per crystal
declare -A CONTIG
CONTIG[6AM5]="A1-180/0 B1-100/0 D1-115/0 E1-120/0 10-10"
CONTIG[6AMU]="A2-180/0 B0-99/0 D2-115/0 E4-120/0 10-10"

# hotspots: [config][crystal]
declare -A HOT
HOT[condA,6AMU]="A9,A77,A80,E100,E99,D30"
HOT[condA,6AM5]="A9,A77,A80,E97,E96,D30"
HOT[condB,6AMU]="A159,A66,A70,D30,D93"
HOT[condB,6AM5]="A159,A66,A70,D30"

echo "[$(date)] RFDIFF cond100 start (configs: condA condB; crystals: 6AMU 6AM5; ndes=$NDES/each)" | tee "$LOG"
cd "$ABS/RFdiffusion"
for cfg in condA condB; do
  for pid in 6AMU 6AM5; do
    pref="$OUT/${cfg}_${pid}"
    if ls "${pref}_$((NDES-1)).pdb" >/dev/null 2>&1; then
      echo "[$(date)] $cfg/$pid already complete, skip" | tee -a "$LOG"; continue
    fi
    gpu_wait 6000
    echo "[$(date)] $cfg/$pid ($NDES designs) hotspots=${HOT[$cfg,$pid]}" | tee -a "$LOG"
    python run_inference.py \
      inference.input_pdb="$ABS/inputs/focus_6am/${pid}_trim.pdb" \
      inference.output_prefix="$pref" \
      inference.num_designs=$NDES \
      "contigmap.contigs=[${CONTIG[$pid]}]" \
      "ppi.hotspot_res=[${HOT[$cfg,$pid]}]" \
      inference.ckpt_override_path=models/Complex_base_ckpt.pt >>"$LOG" 2>&1
    n=$(ls "${pref}"_*.pdb 2>/dev/null | grep -v _split | wc -l)
    echo "[$(date)] $cfg/$pid -> $n backbones" | tee -a "$LOG"
  done
done
tot=$(ls "$OUT"/cond*_*.pdb 2>/dev/null | grep -v _split | wc -l)
echo "[$(date)] RFDIFF cond100 DONE: $tot backbones total" | tee -a "$LOG"
touch "$OUT/COMPLETE"
