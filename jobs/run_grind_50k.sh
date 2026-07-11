#!/usr/bin/env bash
# 50,000 MPNN sequences PER CRYSTAL PER TEMPERATURE (T=0.1/0.3/0.5/0.7), partitioned across that
# crystal's diffusion backbones. 6AMU 70bb x 715, 6AM5 36bb x 1389. Resumable (skips done crystals).
set -uo pipefail
cd /home/ubuntu/if-mhc
PY=/home/ubuntu/miniforge3/envs/esmcba/bin/python
ROOT=/home/ubuntu/if-mhc
BASE=outputs/grind_50k; mkdir -p $BASE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2
LOG=$BASE/run.log; trap 'echo "[$(date)] EXIT $? (grind_50k)" >>"$LOG"' EXIT
gpu_wait(){ while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -lt "$1" ]; do sleep 20; done; }
declare -A NSEQ=( [6AMU]=715 [6AM5]=1389 )   # ceil(50000/nbb)

echo "[$(date)] grind_50k: 50k/crystal at T=0.1,0.3,0.5,0.7 over de-novo backbones" | tee "$LOG"
for T in 0.1 0.3 0.5 0.7; do
  for pid in 6AMU 6AM5; do
    od=$BASE/T${T}/${pid}
    [ -f "$od/DONE" ] && { echo "[$(date)] T=$T $pid already done"; continue; }
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
    gpu_wait 7000
    echo "[$(date)] T=$T $pid : ${NSEQ[$pid]} seq/backbone" | tee -a "$LOG"
    $PY ProteinMPNN/protein_mpnn_run.py --jsonl_path "$od/parsed.jsonl" --chain_id_jsonl "$od/chain.jsonl" \
       --out_folder "$od" --num_seq_per_target ${NSEQ[$pid]} --batch_size 25 \
       --sampling_temp "$T" --seed 41 --model_name v_48_020 >>"$LOG" 2>&1
    n=$(grep -h sample= "$od"/seqs/*.fa 2>/dev/null|wc -l)
    echo "[$(date)] T=$T $pid done: $n seqs" | tee -a "$LOG"; touch "$od/DONE"
  done
done
echo "[$(date)] DONE grind_50k (all temps x crystals)" | tee -a "$LOG"
touch "$BASE/COMPLETE"
