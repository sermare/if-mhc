#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/if-mhc
PY=/home/ubuntu/miniforge3/envs/esmcba/bin/python
OUT=outputs/mpnn_nomhc_Tsweep
IN=$OUT/pdb_in
WEIGHTS=ProteinMPNN/nomhc_model_weights/
MODEL=proteinmpnn_nomhc
mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date)] START ProteinMPNN noMHC | design chain A (peptide) | context chain B | 10 targets x 4500 = 45000 | batch=50 temp=0.3" | tee "$OUT/run.log"

# 1) parse all PDBs
$PY ProteinMPNN/helper_scripts/parse_multiple_chains.py \
   --input_path "$IN" \
   --output_path "$OUT/parsed_chains.jsonl" >> "$OUT/run.log" 2>&1

# 2) assign chain A as designed (B becomes fixed context)
$PY ProteinMPNN/helper_scripts/assign_fixed_chains.py \
   --input_path "$OUT/parsed_chains.jsonl" \
   --output_path "$OUT/assigned_chains.jsonl" \
   --chain_list "A" >> "$OUT/run.log" 2>&1

# 3) run ProteinMPNN with noMHC weights
$PY ProteinMPNN/protein_mpnn_run.py \
   --jsonl_path "$OUT/parsed_chains.jsonl" \
   --chain_id_jsonl "$OUT/assigned_chains.jsonl" \
   --out_folder "$OUT" \
   --num_seq_per_target 4500 \
   --batch_size 50 \
   --sampling_temp "0.3" \
   --seed 37 \
   --path_to_model_weights "$WEIGHTS" \
   --model_name "$MODEL" >> "$OUT/run.log" 2>&1

echo "[$(date)] MPNN done, post-processing" | tee -a "$OUT/run.log"

$PY - <<'PYEOF' 2>>"$OUT/run.log"
import csv, re, glob, os
out="outputs/mpnn_nomhc_Tsweep"
rows=[]; natives={}
for fa in sorted(glob.glob(os.path.join(out,"seqs","*.fa"))):
    target=os.path.splitext(os.path.basename(fa))[0]
    with open(fa) as f:
        lines=[l.rstrip("\n") for l in f]
    for i in range(0,len(lines)-1,2):
        h=lines[i]; s=lines[i+1]
        if h.startswith(">"+target+","):  # native line
            natives[target]=s; continue
        d=dict(re.findall(r'(\w+)=([-\d.]+)', h))
        rows.append({"target":target,"peptide":s,"score":d.get("score"),
                     "global_score":d.get("global_score"),
                     "seq_recovery":d.get("seq_recovery"),
                     "T":d.get("T"),"sample":d.get("sample")})
with open(os.path.join(out,"peptides_45k.csv"),"w",newline="") as o:
    w=csv.DictWriter(o,fieldnames=["target","peptide","score","global_score","seq_recovery","T","sample"])
    w.writeheader(); w.writerows(rows)
uniq=sorted(set(r["peptide"] for r in rows))
with open(os.path.join(out,"peptides_unique.txt"),"w") as o:
    o.write("\n".join(uniq)+"\n")
print("targets=",len(natives))
print("total_sequences=",len(rows))
print("unique_sequences=",len(uniq))
PYEOF

echo "[$(date)] DONE" | tee -a "$OUT/run.log"
touch "$OUT/COMPLETE"
