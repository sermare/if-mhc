#!/usr/bin/env bash
# 10,000 MPNN sequences PER CRYSTAL at T=0.7, partitioned across that crystal's diffusion backbones.
# 6AMU: 70 bb x 143 = 10010 | 6AM5: 36 bb x 278 = 10008.
set -uo pipefail
cd /home/ubuntu/if-mhc
PY=/home/ubuntu/miniforge3/envs/esmcba/bin/python
OUT=outputs/grind_10k_T03
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2
LOG="$OUT/run.log"; trap 'echo "[$(date)] EXIT $? (grind_10k_T03)" >>"$LOG"' EXIT
gpu_wait(){ while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -lt "$1" ]; do sleep 20; done; }

declare -A NSEQ=( [6AMU]=143 [6AM5]=278 )
echo "[$(date)] grind_10k_T03: T=0.3, 10k/crystal partitioned across backbones" | tee "$LOG"
for pid in 6AMU 6AM5; do
  bb="$OUT/${pid}_bb"; cdir="$OUT/${pid}"
  $PY ProteinMPNN/helper_scripts/parse_multiple_chains.py --input_path="$bb" --output_path="$cdir.parsed.jsonl" >>"$LOG" 2>&1
  $PY - "$cdir.parsed.jsonl" "$cdir.chain.jsonl" >>"$LOG" 2>&1 <<'PY'
import json,sys
parsed,outp=sys.argv[1],sys.argv[2]; ci={}
for line in open(parsed):
    r=json.loads(line); ch=[k[-1] for k in r if k.startswith("seq_chain_")]
    pep=next((c for c in ch if 8<=len(r.get(f"seq_chain_{c}",""))<=12), ch[-1])
    ci[r["name"]]=[[pep],[c for c in ch if c!=pep]]
json.dump(ci,open(outp,"w")); print("assigned",len(ci))
PY
  mkdir -p "$cdir"
  gpu_wait 9000
  echo "[$(date)] $pid : ${NSEQ[$pid]} seq/backbone @ T=0.3" | tee -a "$LOG"
  $PY ProteinMPNN/protein_mpnn_run.py --jsonl_path "$cdir.parsed.jsonl" --chain_id_jsonl "$cdir.chain.jsonl" \
     --out_folder "$cdir" --num_seq_per_target ${NSEQ[$pid]} --batch_size 25 \
     --sampling_temp "0.3" --seed 37 --model_name v_48_020 >>"$LOG" 2>&1
  n=$(grep -h "sample=" "$cdir"/seqs/*.fa 2>/dev/null | wc -l)
  echo "[$(date)] $pid done: $n sequences" | tee -a "$LOG"
done
$PY grind_10k_report.py "$OUT" 2>>"$LOG" | tee "$OUT/report.txt"
echo "[$(date)] DONE grind_10k" | tee -a "$LOG"
touch "$OUT/COMPLETE"
