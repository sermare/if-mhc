#!/usr/bin/env bash
# Contact-conditioning ladder: per condition, RFdiffusion (3 de-novo 10-mer structures, N-term MHC
# anchors + ranked TCR contacts, increasing # contacts) -> ProteinMPNN 2000/structure (=6000), T=0.1.
set -uo pipefail
cd /home/ubuntu/if-mhc
ABS=/home/ubuntu/if-mhc
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
PY=/home/ubuntu/miniforge3/envs/esmcba/bin/python
OUT=outputs/ladder; mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2
LOG="$OUT/run.log"; trap 'echo "[$(date)] EXIT $? (ladder)" >>"$LOG"' EXIT
gpu_wait(){ while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -lt "$1" ]; do sleep 20; done; }

# 1) RFdiffusion: 3 structures per condition
conda activate SE3nv
export LD_LIBRARY_PATH="$(cat $ABS/RFdiffusion/SE3NV_LDPATH.txt 2>/dev/null):${LD_LIBRARY_PATH:-}"
echo "[$(date)] ladder: RFdiffusion 3 structures/condition (10 conditions)" | tee "$LOG"
cd "$ABS/RFdiffusion"
while IFS=$'\t' read -r pid name nc contig hot; do
  [ -z "$pid" ] && continue
  ls "$ABS/$OUT/pdb/${pid}_${name}_2.pdb" >/dev/null 2>&1 && continue   # resume (3 designs: _0.._2)
  mkdir -p "$ABS/$OUT/pdb"
  gpu_wait 6000
  echo "[$(date)] $pid/$name (ncontacts=$nc) hot=[$hot]" | tee -a "$ABS/$LOG"
  python run_inference.py inference.input_pdb="$ABS/inputs/focus_6am/${pid}_trim.pdb" \
     "contigmap.contigs=[$contig]" "ppi.hotspot_res=[$hot]" inference.num_designs=3 diffuser.T=30 \
     inference.ckpt_override_path=models/Complex_base_ckpt.pt \
     inference.output_prefix="$ABS/$OUT/pdb/${pid}_${name}" >>"$ABS/$LOG" 2>&1 || true
done < "$ABS/inputs/focus_6am/ladder_spec.tsv"
cd "$ABS"

# 2) ProteinMPNN: 2000 seq/structure (=6000 per condition), T=0.1, unconstrained, design peptide chain A
mkdir -p "$OUT/bb"; rm -f "$OUT/bb"/*.pdb
for f in "$OUT"/pdb/*.pdb; do case "$f" in *_split.pdb) ;; *) cp "$f" "$OUT/bb/";; esac; done
$PY ProteinMPNN/helper_scripts/parse_multiple_chains.py --input_path="$OUT/bb" --output_path="$OUT/parsed.jsonl" >>"$LOG" 2>&1
$PY - "$OUT" >>"$LOG" 2>&1 <<'PY'
import json,sys
out=sys.argv[1]; ci={}
for line in open(f"{out}/parsed.jsonl"):
    r=json.loads(line); ch=[k[-1] for k in r if k.startswith("seq_chain_")]
    pep=next((c for c in ch if len(r.get(f"seq_chain_{c}",""))==10), ch[-1])
    ci[r["name"]]=[[pep],[c for c in ch if c!=pep]]
json.dump(ci,open(f"{out}/chain.jsonl","w")); print("assigned",len(ci),"structures")
PY
gpu_wait 6000
echo "[$(date)] ladder MPNN: 2000/structure T=0.1" | tee -a "$LOG"
$PY ProteinMPNN/protein_mpnn_run.py --jsonl_path "$OUT/parsed.jsonl" --chain_id_jsonl "$OUT/chain.jsonl" \
   --out_folder "$OUT" --num_seq_per_target 2000 --batch_size 25 \
   --sampling_temp "0.1" --seed 41 --model_name v_48_020 >>"$LOG" 2>&1
$PY ladder_analysis.py "$OUT" 2>>"$LOG" | tee "$OUT/analysis.txt"
echo "[$(date)] DONE ladder" | tee -a "$LOG"
touch "$OUT/COMPLETE"
