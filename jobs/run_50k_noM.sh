#!/usr/bin/env bash
# Another full-complex 50K ProteinMPNN IF run, but forbidding Met at peptide position 1 (chain C).
set -uo pipefail
cd /home/ubuntu/if-mhc
PY=/home/ubuntu/miniforge3/envs/esmcba/bin/python
NB=/home/ubuntu/miniforge3/bin/python
OUT=outputs/mpnn_50k_noM; mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG="$OUT/run.log"
trap 'echo "[$(date)] EXIT code $? (50k_noM)" >> "$LOG"' EXIT

# omit Met at chain C position 1
$NB make_omit_M_pos1.py outputs/mpnn/parsed_chains.jsonl "$OUT/omit.jsonl" C 1 M >>"$LOG" 2>&1

echo "[$(date)] START 50K full-complex, NO Met@P1, batch24 seed40" | tee -a "$LOG"
$PY -u ProteinMPNN/protein_mpnn_run.py \
   --jsonl_path outputs/mpnn/parsed_chains.jsonl \
   --chain_id_jsonl outputs/mpnn/assigned_chains.jsonl \
   --omit_AA_jsonl "$OUT/omit.jsonl" \
   --out_folder "$OUT" \
   --num_seq_per_target 50048 --batch_size 24 \
   --sampling_temp "0.3" --seed 40 --model_name v_48_020 >>"$LOG" 2>&1

echo "[$(date)] done, parsing + verifying no Met@P1" | tee -a "$LOG"
$NB - "$OUT" >>"$LOG" 2>&1 <<'PYEOF'
import sys,re,csv
out=sys.argv[1]; fa=f"{out}/seqs/2P5E.fa"; rows=[]
ls=open(fa).read().splitlines()
for i in range(0,len(ls)-1,2):
    h,s=ls[i],ls[i+1].strip()
    if "sample=" not in h or len(s)!=9: continue
    d=dict(re.findall(r'(\w+)=([-\d.]+)',h))
    rows.append({"peptide":s,"score":d.get("score"),"global_score":d.get("global_score"),
                 "seq_recovery":d.get("seq_recovery"),"T":d.get("T"),"sample":d.get("sample")})
csv.DictWriter(open(f"{out}/peptides_50k.csv","w",newline=""),
    fieldnames=["peptide","score","global_score","seq_recovery","T","sample"]).writerows(
    [{"peptide":"peptide","score":"score","global_score":"global_score","seq_recovery":"seq_recovery","T":"T","sample":"sample"}]+rows)
open(f"{out}/peptides_unique.txt","w").write("\n".join(sorted(set(r['peptide'] for r in rows)))+"\n")
m1=sum(1 for r in rows if r['peptide'][0]=='M')
print(f"total={len(rows)} unique={len(set(r['peptide'] for r in rows))} | Met-at-P1 count={m1} (should be 0)")
PYEOF
echo "[$(date)] DONE 50k_noM" | tee -a "$LOG"
touch "$OUT/COMPLETE"
