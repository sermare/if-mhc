#!/usr/bin/env bash
set -uo pipefail
cd /home/ubuntu/if-mhc
OMM=/home/ubuntu/miniforge3/envs/openmm/bin/python
PY=/home/ubuntu/miniforge3/envs/esmcba/bin/python
OUT=outputs/focus_relax; mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 OPENMM_CPU_THREADS=3
LOG="$OUT/run.log"; trap 'echo "[$(date)] EXIT $? (focus_relax)" >>"$LOG"' EXIT
gpu_wait(){ while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -lt "$1" ]; do sleep 20; done; }
echo "[$(date)] OpenMM relax of 6AMT/6AMU/6AM5 (8 snapshots each)" | tee "$LOG"
$OMM focus_relax_openmm.py >>"$LOG" 2>&1
ns=$(ls "$OUT"/snapshots/*.pdb 2>/dev/null|wc -l); echo "[$(date)] snapshots:$ns" | tee -a "$LOG"
[ "$ns" -lt 1 ] && exit 1
$PY ProteinMPNN/helper_scripts/parse_multiple_chains.py --input_path="$OUT/snapshots" --output_path="$OUT/parsed.jsonl" >>"$LOG" 2>&1
$PY - "$OUT" >>"$LOG" 2>&1 <<'PY'
import json,sys
out=sys.argv[1]; d={}
for line in open(f"{out}/parsed.jsonl"):
    r=json.loads(line); chains=[k[-1] for k in r if k.startswith("seq_chain_")]
    d[r["name"]]=[["C"],[c for c in chains if c!="C"]]   # design peptide chain C, UNCONSTRAINED
json.dump(d,open(f"{out}/chain_id.jsonl","w")); print("chain_id for",len(d),"snapshots")
PY
gpu_wait 6000
echo "[$(date)] ProteinMPNN unconstrained over relaxed ensemble (T=0.1, 1000/snap)" | tee -a "$LOG"
$PY ProteinMPNN/protein_mpnn_run.py --jsonl_path "$OUT/parsed.jsonl" --chain_id_jsonl "$OUT/chain_id.jsonl" \
   --out_folder "$OUT" --num_seq_per_target 1000 --batch_size 8 --sampling_temp "0.1" --seed 37 --model_name v_48_020 >>"$LOG" 2>&1
$PY focus_relax_recovery.py "$OUT" 2>>"$LOG" | tee "$OUT/recovery_relaxed_table.txt"
echo "[$(date)] DONE focus_relax" | tee -a "$LOG"
touch "$OUT/COMPLETE"
