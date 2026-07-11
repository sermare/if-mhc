#!/usr/bin/env bash
# Re-run ProteinMPNN over the existing 12 relax snapshots, now forbidding Proline (global) + Met@P1.
set -uo pipefail
cd /home/ubuntu/if-mhc
PY=/home/ubuntu/miniforge3/envs/esmcba/bin/python
NB=/home/ubuntu/miniforge3/bin/python
OUT=outputs/relax_campaign
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG="$OUT/mpnn_noPro.log"
trap 'echo "[$(date)] EXIT $? (relax_mpnn)" >> "$LOG"' EXIT
gpu_wait(){ while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -lt "$1" ]; do sleep 30; done; }

rm -f "$OUT"/seqs/*.fa "$OUT"/COMPLETE        # clear old proline-containing designs
echo "[$(date)] relax MPNN re-run: no Proline + no Met@P1, over $(ls $OUT/snapshots/*.pdb|wc -l) snapshots" | tee "$LOG"
$PY ProteinMPNN/helper_scripts/parse_multiple_chains.py --input_path="$OUT/snapshots" --output_path="$OUT/parsed.jsonl" >>"$LOG" 2>&1
$PY ProteinMPNN/helper_scripts/assign_fixed_chains.py --input_path="$OUT/parsed.jsonl" --output_path="$OUT/assigned.jsonl" --chain_list "C" >>"$LOG" 2>&1
$NB make_omit_M_pos1.py "$OUT/parsed.jsonl" "$OUT/omit.jsonl" C 1 M >>"$LOG" 2>&1
gpu_wait 6000
$PY ProteinMPNN/protein_mpnn_run.py --jsonl_path "$OUT/parsed.jsonl" --chain_id_jsonl "$OUT/assigned.jsonl" \
   --omit_AA_jsonl "$OUT/omit.jsonl" --omit_AAs "PX" \
   --out_folder "$OUT" --num_seq_per_target 2000 --batch_size 16 \
   --sampling_temp "0.3" --seed 37 --model_name v_48_020 >>"$LOG" 2>&1

$NB - "$OUT" >>"$LOG" 2>&1 <<'PYEOF'
import sys,glob,csv
out=sys.argv[1]; rows=[]
for fa in glob.glob(f"{out}/seqs/*.fa"):
    bb=fa.split("/")[-1].replace(".fa","")
    ls=open(fa).read().splitlines()
    for i in range(0,len(ls)-1,2):
        h,s=ls[i],ls[i+1].strip()
        if "sample=" not in h or len(s)!=9: continue
        rows.append({"backbone":bb,"peptide":s})
w=csv.DictWriter(open(f"{out}/ensemble_peptides.csv","w",newline=""),fieldnames=["backbone","peptide"]);w.writeheader();w.writerows(rows)
pro=sum(1 for r in rows if 'P' in r['peptide']); m1=sum(1 for r in rows if r['peptide'][0]=='M')
print(f"ensemble={len(rows)} unique={len(set(r['peptide'] for r in rows))} | with-Pro={pro} (should be 0) | Met@P1={m1} (should be 0)")
PYEOF
echo "[$(date)] DONE relax MPNN (no Pro)" | tee -a "$LOG"
touch "$OUT/COMPLETE"
