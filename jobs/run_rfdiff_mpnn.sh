#!/usr/bin/env bash
# ProteinMPNN over the 10 RFdiffusion backbones (no Met@P1 + no Pro), then split to clean complexes.
set -uo pipefail
cd /home/ubuntu/if-mhc
PY=/home/ubuntu/miniforge3/envs/esmcba/bin/python
NB=/home/ubuntu/miniforge3/bin/python
OUT=outputs/rfdiff_campaign
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2
LOG="$OUT/mpnn.log"
trap 'echo "[$(date)] EXIT $? (rfdiff_mpnn)" >> "$LOG"' EXIT
gpu_wait(){ while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -lt "$1" ]; do sleep 20; done; }

mkdir -p "$OUT/bb"; rm -f "$OUT/bb"/*.pdb
for f in "$OUT"/pep_*.pdb; do case "$f" in *_split.pdb) ;; *) cp "$f" "$OUT/bb/";; esac; done
echo "[$(date)] rfdiff MPNN over $(ls $OUT/bb/*.pdb|wc -l) backbones" | tee "$LOG"
$PY ProteinMPNN/helper_scripts/parse_multiple_chains.py --input_path="$OUT/bb" --output_path="$OUT/parsed.jsonl" >>"$LOG" 2>&1
$NB - "$OUT" >>"$LOG" 2>&1 <<'PY'
import json,sys
out=sys.argv[1]; assigned={}; omit={}
for line in open(f"{out}/parsed.jsonl"):
    r=json.loads(line); chains=[k[-1] for k in r if k.startswith("seq_chain_")]
    pep=next((c for c in chains if 8<=len(r.get(f"seq_chain_{c}",""))<=11), chains[-1])
    assigned[r["name"]]=[[pep],[c for c in chains if c!=pep]]
    omit[r["name"]]={c:([[[1],"M"]] if c==pep else []) for c in chains}
json.dump(assigned,open(f"{out}/assigned.jsonl","w")); json.dump(omit,open(f"{out}/omit.jsonl","w"))
print("assigned peptide chains for",len(assigned),"designs")
PY
gpu_wait 6000
$PY ProteinMPNN/protein_mpnn_run.py --jsonl_path "$OUT/parsed.jsonl" --chain_id_jsonl "$OUT/assigned.jsonl" \
   --omit_AA_jsonl "$OUT/omit.jsonl" --omit_AAs "PX" \
   --out_folder "$OUT" --num_seq_per_target 1000 --batch_size 8 \
   --sampling_temp "0.3" --seed 37 --model_name v_48_020 >>"$LOG" 2>&1
# split all backbones to clean P/A/B/D/E complexes
for f in "$OUT"/pep_*.pdb; do case "$f" in *_split.pdb) ;; *) $PY split_rfdiff_chains.py "$f" "${f%.pdb}_split.pdb" >/dev/null 2>&1;; esac; done
echo "[$(date)] DONE rfdiff MPNN" | tee -a "$LOG"
touch "$OUT/MPNN_COMPLETE"
