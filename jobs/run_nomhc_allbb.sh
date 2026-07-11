#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/if-mhc
PY=/home/ubuntu/miniforge3/envs/esmcba/bin/python
OUT=outputs/mpnn_nomhc_allbb
IN=$OUT/pdb_in
WEIGHTS=ProteinMPNN/nomhc_model_weights/
MODEL=proteinmpnn_nomhc
NSEQ=64
BATCH=32
mkdir -p "$OUT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date)] START noMHC MPNN across ALL core_load backbones | 614 targets x $NSEQ | per-target peptide chain | temp=0.3" | tee "$OUT/run.log"

# 1) parse every staged backbone
$PY ProteinMPNN/helper_scripts/parse_multiple_chains.py \
   --input_path "$IN" --output_path "$OUT/parsed_chains.jsonl" >> "$OUT/run.log" 2>&1

# 2) build assigned_chains.jsonl: designed = each target's peptide chain (from manifest), rest fixed
$PY - <<'PYEOF' 2>>"$OUT/run.log"
import json, csv
OUT="outputs/mpnn_nomhc_allbb"
pep={r["target"]:r["pep_chain"] for r in csv.DictReader(open(f"{OUT}/manifest.csv"))}
assigned={}
for line in open(f"{OUT}/parsed_chains.jsonl"):
    d=json.loads(line); name=d["name"]
    chains=sorted(k.split("_")[-1] for k in d if k.startswith("seq_chain_"))
    des=pep.get(name)
    if des not in chains:   # fallback: a lone 10-mer chain
        des=chains[0]
    fixed=[c for c in chains if c!=des]
    assigned[name]=[[des],fixed]
json.dump(assigned, open(f"{OUT}/assigned_chains.jsonl","w"))
print("assigned targets:",len(assigned))
PYEOF

# 3) run noMHC ProteinMPNN
$PY ProteinMPNN/protein_mpnn_run.py \
   --jsonl_path "$OUT/parsed_chains.jsonl" \
   --chain_id_jsonl "$OUT/assigned_chains.jsonl" \
   --out_folder "$OUT" \
   --num_seq_per_target "$NSEQ" \
   --batch_size "$BATCH" \
   --sampling_temp "0.3" \
   --seed 37 \
   --path_to_model_weights "$WEIGHTS" \
   --model_name "$MODEL" >> "$OUT/run.log" 2>&1

echo "[$(date)] MPNN done, post-processing" | tee -a "$OUT/run.log"

# 4) aggregate -> CSV joined to manifest metadata
$PY - <<'PYEOF' 2>>"$OUT/run.log"
import csv, re, glob, os
OUT="outputs/mpnn_nomhc_allbb"
meta={r["target"]:r for r in csv.DictReader(open(f"{OUT}/manifest.csv"))}
rows=[]
for fa in sorted(glob.glob(f"{OUT}/seqs/*.fa")):
    target=os.path.splitext(os.path.basename(fa))[0]
    m=meta.get(target,{})
    lines=open(fa).read().splitlines()
    for i in range(0,len(lines)-1,2):
        h=lines[i]
        if not h.startswith(">T="): continue
        d=dict(re.findall(r'(\w+)=([-\d.]+)', h))
        rows.append({"target":target,"pid":m.get("pid"),"group":m.get("group"),
                     "cond":m.get("cond"),"nice":m.get("nice"),"pep_chain":m.get("pep_chain"),
                     "toGIG":m.get("toGIG"),"toDRG":m.get("toDRG"),"min_native":m.get("min_native"),
                     "seated":m.get("seated"),"peptide":lines[i+1].strip(),
                     "score":d.get("score"),"global_score":d.get("global_score")})
fields=["target","pid","group","cond","nice","pep_chain","toGIG","toDRG","min_native",
        "seated","peptide","score","global_score"]
with open(f"{OUT}/designs_allbb.csv","w",newline="") as o:
    w=csv.DictWriter(o,fieldnames=fields); w.writeheader(); w.writerows(rows)
print("total sequences:",len(rows))
print("unique peptides:",len(set(r["peptide"] for r in rows)))
print("targets with output:",len(set(r["target"] for r in rows)))
PYEOF

echo "[$(date)] DONE" | tee -a "$OUT/run.log"
touch "$OUT/COMPLETE"
