#!/usr/bin/env bash
set -uo pipefail
cd /home/ubuntu/if-mhc
pid="$1"; nseq="$2"
PY=/home/ubuntu/miniforge3/envs/esmcba/bin/python
BASE=outputs/grind_50k; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2
LOG=$BASE/worker_${pid}.log
gpu_wait(){ while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -lt "$1" ]; do sleep 20; done; }
for T in 0.1; do   # T!=0.1 stopped: no convergence, ~random
  od=$BASE/T${T}/${pid}
  [ -f "$od/DONE" ] && { echo "[$(date)] T=$T $pid done" >>"$LOG"; continue; }
  mkdir -p "$od/bb"; cp outputs/grind/pdb/${pid}_*.pdb "$od/bb/" 2>/dev/null
  $PY ProteinMPNN/helper_scripts/parse_multiple_chains.py --input_path="$od/bb" --output_path="$od/parsed.jsonl" >>"$LOG" 2>&1
  $PY - "$od/parsed.jsonl" "$od/chain.jsonl" >>"$LOG" 2>&1 <<'PY'
import json,sys
parsed,outp=sys.argv[1],sys.argv[2]; ci={}
for line in open(parsed):
    r=json.loads(line); ch=[k[-1] for k in r if k.startswith("seq_chain_")]
    pep=next((c for c in ch if 8<=len(r.get(f"seq_chain_{c}",""))<=12), ch[-1])
    ci[r["name"]]=[[pep],[c for c in ch if c!=pep]]
json.dump(ci,open(outp,"w"))
PY
  gpu_wait 5000
  echo "[$(date)] T=$T $pid : $nseq seq/backbone" >>"$LOG"
  $PY ProteinMPNN/protein_mpnn_run.py --jsonl_path "$od/parsed.jsonl" --chain_id_jsonl "$od/chain.jsonl" \
     --out_folder "$od" --num_seq_per_target "$nseq" --batch_size 25 \
     --sampling_temp "$T" --seed 41 --model_name v_48_020 >>"$LOG" 2>&1
  echo "[$(date)] T=$T $pid done: $(grep -h sample= $od/seqs/*.fa 2>/dev/null|wc -l)" >>"$LOG"; touch "$od/DONE"
done
touch "$BASE/worker_${pid}.DONE"
