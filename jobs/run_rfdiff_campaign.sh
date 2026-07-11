#!/usr/bin/env bash
# PHASE 2 campaign: RFdiffusion generates a variable-length (8-11) peptide in the HLA groove,
# with MHC (A,B) + TCR (D,E) held as fixed motif and DUAL-interface hotspots (groove + TCR),
# then ProteinMPNN designs the peptide (no Met at P1). Self-guards on env+weights, validates first.
set -uo pipefail
cd /home/ubuntu/if-mhc
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
ABS=/home/ubuntu/if-mhc
OUT=$ABS/outputs/rfdiff_campaign; mkdir -p "$OUT"
LOG="$OUT/campaign.log"
MPNN=/home/ubuntu/miniforge3/envs/esmcba/bin/python
gpu_wait(){ while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -lt "$1" ]; do sleep 30; done; }

echo "[$(date)] RFDIFF campaign: waiting for env build + weights..." | tee "$LOG"
while [ ! -f $ABS/RFdiffusion/ENV_BUILD_DONE ] || [ ! -f $ABS/RFdiffusion/models/DL_COMPLETE ]; do sleep 30; done
echo "[$(date)] env + weights ready" | tee -a "$LOG"
conda activate SE3nv
# DGL needs the bundled CUDA libs (libcusparse.so.11) on the loader path
export LD_LIBRARY_PATH="$(cat $ABS/RFdiffusion/SE3NV_LDPATH.txt 2>/dev/null):${LD_LIBRARY_PATH:-}"
cd $ABS/RFdiffusion

# --- GATE 1: does the modernized env actually run on the L4 (sm_89)? ---
gpu_wait 6000
echo "[$(date)] GATE1: validating env (tiny unconditional 60-mer)" | tee -a "$LOG"
python run_inference.py 'contigmap.contigs=[60-60]' inference.num_designs=1 \
   inference.output_prefix="$OUT/_validate/v" >>"$LOG" 2>&1
if ! ls "$OUT"/_validate/*.pdb >/dev/null 2>&1; then
   echo "[$(date)] GATE1 FAILED — env not running on GPU. Stopping for manual fix." | tee -a "$LOG"
   touch "$OUT/ENV_INVALID"; exit 1
fi
echo "[$(date)] GATE1 PASSED" | tee -a "$LOG"

# --- PILOT: 8-11mer peptide in groove; MHC+TCR fixed motif; dual-interface hotspots ---
# (NOTE: standard RFdiffusion is not optimized for <50aa peptides; rfpeptides branch is preferred.
#  This pilot tests feasibility before scaling.)
gpu_wait 6000
echo "[$(date)] PILOT: 8-11mer peptide generation (20 designs, dual hotspots)" | tee -a "$LOG"
python run_inference.py \
   inference.input_pdb=$ABS/inputs/2P5E_trimmed.pdb \
   'contigmap.contigs=[A1-180/0 B0-99/0 D0-115/0 E1-120/0 8-11]' \
   'ppi.hotspot_res=[A9,A63,A66,A70,A77,A80,A99,A116,A143,A147,D96,E96]' \
   inference.num_designs=10 \
   inference.ckpt_override_path=models/Complex_base_ckpt.pt \
   inference.output_prefix="$OUT/pep" >>"$LOG" 2>&1
nbb=$(ls "$OUT"/pep_*.pdb 2>/dev/null | wc -l)
echo "[$(date)] PILOT produced $nbb backbones" | tee -a "$LOG"
[ "$nbb" -lt 1 ] && { echo "PILOT produced nothing — review contig/hotspots/length." | tee -a "$LOG"; touch "$OUT/PILOT_FAILED"; exit 1; }

# --- ProteinMPNN on the generated peptide chain (auto-detect the 8-11-len chain), no Met@P1 ---
mkdir -p "$OUT/bb"; cp "$OUT"/pep_*.pdb "$OUT/bb/" 2>/dev/null
cd $ABS
$MPNN ProteinMPNN/helper_scripts/parse_multiple_chains.py --input_path="$OUT/bb" --output_path="$OUT/parsed.jsonl" >>"$LOG" 2>&1
$MPNN - "$OUT" >>"$LOG" 2>&1 <<'PY'
import json,sys
out=sys.argv[1]; assigned={}; omit={}
for line in open(f"{out}/parsed.jsonl"):
    r=json.loads(line); chains=[k[-1] for k in r if k.startswith("seq_chain_")]
    pep=None
    for c in chains:
        seq=r.get(f"seq_chain_{c}","")
        if 8<=len(seq)<=11: pep=c; break
    pep=pep or chains[-1]
    assigned[r["name"]]=[[pep],[c for c in chains if c!=pep]]
    omit[r["name"]]={c:([[[1],"M"]] if c==pep else []) for c in chains}
json.dump(assigned,open(f"{out}/assigned.jsonl","w"))
json.dump(omit,open(f"{out}/omit.jsonl","w"))
print("peptide chains assigned for",len(assigned),"designs")
PY
gpu_wait 8000
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $MPNN ProteinMPNN/protein_mpnn_run.py \
   --jsonl_path "$OUT/parsed.jsonl" --chain_id_jsonl "$OUT/assigned.jsonl" --omit_AA_jsonl "$OUT/omit.jsonl" --omit_AAs "PX" \
   --out_folder "$OUT" --num_seq_per_target 1000 --batch_size 16 \
   --sampling_temp "0.3" --seed 37 --model_name v_48_020 >>"$LOG" 2>&1
echo "[$(date)] RFDIFF PILOT DONE — review backbones/lengths before scaling num_designs." | tee -a "$LOG"
touch "$OUT/PILOT_COMPLETE"
