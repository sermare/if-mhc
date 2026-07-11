#!/usr/bin/env bash
# RFdiffusion partial diffusion of 6AMU/6AM5 (perturb peptide backbone at partial_T sweep) ->
# ProteinMPNN unconstrained -> own-vs-other recovery. Tests if structure-specific signal survives.
set -uo pipefail
cd /home/ubuntu/if-mhc
ABS=/home/ubuntu/if-mhc
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
PY=/home/ubuntu/miniforge3/envs/esmcba/bin/python
OUT=outputs/focus_rfdiff; mkdir -p "$OUT/bb"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2
LOG="$OUT/run.log"; trap 'echo "[$(date)] EXIT $? (focus_rfdiff)" >>"$LOG"' EXIT
gpu_wait(){ while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -lt "$1" ]; do sleep 20; done; }

conda activate SE3nv
export LD_LIBRARY_PATH="$(cat $ABS/RFdiffusion/SE3NV_LDPATH.txt 2>/dev/null):${LD_LIBRARY_PATH:-}"
echo "[$(date)] RFdiffusion partial diffusion of 6AMU/6AM5 (partial_T 10,20)" | tee "$LOG"
cd "$ABS/RFdiffusion"
while IFS=$'\t' read -r pid contig; do
  for pT in 10 20; do
    ls "$OUT/${pid}_pT${pT}_0.pdb" >/dev/null 2>&1 && continue
    gpu_wait 6000
    echo "[$(date)] $pid partial_T=$pT" | tee -a "$LOG"
    python run_inference.py inference.input_pdb="$ABS/inputs/focus_6am/${pid}_trim.pdb" \
       "contigmap.contigs=[$contig]" diffuser.partial_T=$pT inference.num_designs=15 \
       inference.ckpt_override_path=models/Complex_base_ckpt.pt \
       inference.output_prefix="$ABS/$OUT/${pid}_pT${pT}" >>"$ABS/$LOG" 2>&1
  done
done < "$ABS/inputs/focus_6am/contigs_focus.txt"
cd "$ABS"

# ProteinMPNN unconstrained on the diffused peptide (auto-detect 8-12mer chain)
for f in "$OUT"/*.pdb; do case "$f" in *_split.pdb) ;; *) cp "$f" "$OUT/bb/";; esac; done
$PY ProteinMPNN/helper_scripts/parse_multiple_chains.py --input_path="$OUT/bb" --output_path="$OUT/parsed.jsonl" >>"$LOG" 2>&1
$PY - "$OUT" >>"$LOG" 2>&1 <<'PY'
import json,sys
out=sys.argv[1]; ci={}
for line in open(f"{out}/parsed.jsonl"):
    r=json.loads(line); ch=[k[-1] for k in r if k.startswith("seq_chain_")]
    pep=next((c for c in ch if 8<=len(r.get(f"seq_chain_{c}",""))<=12), ch[-1])
    ci[r["name"]]=[[pep],[c for c in ch if c!=pep]]
json.dump(ci,open(f"{out}/chain_id.jsonl","w")); print("assigned",len(ci))
PY
gpu_wait 6000
$PY ProteinMPNN/protein_mpnn_run.py --jsonl_path "$OUT/parsed.jsonl" --chain_id_jsonl "$OUT/chain_id.jsonl" \
   --out_folder "$OUT" --num_seq_per_target 500 --batch_size 8 --sampling_temp "0.1" --seed 37 --model_name v_48_020 >>"$LOG" 2>&1
$PY focus_rfdiff_recovery.py "$OUT" 2>>"$LOG" | tee "$OUT/recovery_table.txt"
echo "[$(date)] DONE focus_rfdiff" | tee -a "$LOG"
touch "$OUT/COMPLETE"
