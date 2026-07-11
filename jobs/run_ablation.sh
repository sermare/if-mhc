#!/usr/bin/env bash
# HOTSPOT-ABLATION SCAN: de-novo 10-mer peptides in 6AMU/6AM5 grooves under increasing interface
# conditioning (tcr1, tcr2, mhc, mhc+tcr1, mhc+tcr2) -> ProteinMPNN -> identity vs DRG & GIG.
set -uo pipefail
cd /home/ubuntu/if-mhc
ABS=/home/ubuntu/if-mhc
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
PY=/home/ubuntu/miniforge3/envs/esmcba/bin/python
OUT=outputs/ablation; mkdir -p "$OUT/bb"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2
LOG="$OUT/run.log"; trap 'echo "[$(date)] EXIT $? (ablation)" >>"$LOG"' EXIT
gpu_wait(){ while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -lt "$1" ]; do sleep 20; done; }
NDES=8

conda activate SE3nv
export LD_LIBRARY_PATH="$(cat $ABS/RFdiffusion/SE3NV_LDPATH.txt 2>/dev/null):${LD_LIBRARY_PATH:-}"
echo "[$(date)] ABLATION scan: 2 crystals x 5 conditioning levels, $NDES designs each" | tee "$LOG"
cd "$ABS/RFdiffusion"
while IFS=$'\t' read -r pid name contig hot; do
  [ -z "$pid" ] && continue
  ls "$ABS/$OUT/${pid}_${name}_0.pdb" >/dev/null 2>&1 && continue   # resume
  gpu_wait 6000
  echo "[$(date)] $pid / $name  hotspots=[$hot]" | tee -a "$ABS/$LOG"
  python run_inference.py inference.input_pdb="$ABS/inputs/focus_6am/${pid}_trim.pdb" \
     "contigmap.contigs=[$contig]" "ppi.hotspot_res=[$hot]" inference.num_designs=$NDES \
     inference.ckpt_override_path=models/Complex_base_ckpt.pt \
     inference.output_prefix="$ABS/$OUT/${pid}_${name}" >>"$ABS/$LOG" 2>&1
done < "$ABS/inputs/focus_6am/ablation_spec.tsv"
cd "$ABS"

# ProteinMPNN (unconstrained) on every generated peptide backbone (auto-detect 10-mer chain)
for f in "$OUT"/*.pdb; do case "$f" in *_split.pdb) ;; *) cp "$f" "$OUT/bb/";; esac; done
$PY ProteinMPNN/helper_scripts/parse_multiple_chains.py --input_path="$OUT/bb" --output_path="$OUT/parsed.jsonl" >>"$LOG" 2>&1
$PY - "$OUT" >>"$LOG" 2>&1 <<'PY'
import json,sys
out=sys.argv[1]; ci={}
for line in open(f"{out}/parsed.jsonl"):
    r=json.loads(line); ch=[k[-1] for k in r if k.startswith("seq_chain_")]
    pep=next((c for c in ch if len(r.get(f"seq_chain_{c}",""))==10), ch[-1])
    ci[r["name"]]=[[pep],[c for c in ch if c!=pep]]
json.dump(ci,open(f"{out}/chain_id.jsonl","w")); print("assigned",len(ci))
PY
gpu_wait 6000
$PY ProteinMPNN/protein_mpnn_run.py --jsonl_path "$OUT/parsed.jsonl" --chain_id_jsonl "$OUT/chain_id.jsonl" \
   --out_folder "$OUT" --num_seq_per_target 200 --batch_size 8 --sampling_temp "0.1" --seed 37 --model_name v_48_020 >>"$LOG" 2>&1
$PY ablation_analysis.py "$OUT" 2>>"$LOG" | tee "$OUT/analysis.txt"
echo "[$(date)] DONE ablation" | tee -a "$LOG"
touch "$OUT/COMPLETE"
