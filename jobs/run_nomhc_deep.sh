#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/if-mhc
PY=/home/ubuntu/miniforge3/envs/esmcba/bin/python
SRC=outputs/mpnn_nomhc_allbb          # reuse its parsed/assigned/manifest
OUT=outputs/mpnn_nomhc_allbb_deep
WEIGHTS=ProteinMPNN/nomhc_model_weights/
MODEL=proteinmpnn_nomhc
mkdir -p "$OUT/seqs"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date)] START noMHC DEEP | natives x2048, rest x256 | reuse parsed/assigned | temp=0.3" | tee "$OUT/run.log"

# 1) split parsed + assigned jsonl into native vs rest by manifest group
$PY - <<'PYEOF' 2>>"$OUT/run.log"
import json, csv
SRC="outputs/mpnn_nomhc_allbb"; OUT="outputs/mpnn_nomhc_allbb_deep"
grp={r["target"]:r["group"] for r in csv.DictReader(open(f"{SRC}/manifest.csv"))}
nat=set(t for t,g in grp.items() if g=="native")
# parsed: one json/line keyed by "name"
pn=open(f"{OUT}/parsed_native.jsonl","w"); pr=open(f"{OUT}/parsed_rest.jsonl","w")
for line in open(f"{SRC}/parsed_chains.jsonl"):
    d=json.loads(line); (pn if d["name"] in nat else pr).write(line)
pn.close(); pr.close()
# assigned: single dict
asg=json.load(open(f"{SRC}/assigned_chains.jsonl"))
json.dump({k:v for k,v in asg.items() if k in nat}, open(f"{OUT}/assigned_native.jsonl","w"))
json.dump({k:v for k,v in asg.items() if k not in nat}, open(f"{OUT}/assigned_rest.jsonl","w"))
print("native targets:",len(nat),"| rest:",len(asg)-len(nat))
PYEOF

run_subset () {  # $1=tag $2=nseq $3=batch
  echo "[$(date)] subset=$1 nseq=$2 batch=$3" | tee -a "$OUT/run.log"
  $PY ProteinMPNN/protein_mpnn_run.py \
     --jsonl_path "$OUT/parsed_$1.jsonl" \
     --chain_id_jsonl "$OUT/assigned_$1.jsonl" \
     --out_folder "$OUT" \
     --num_seq_per_target "$2" \
     --batch_size "$3" \
     --sampling_temp "0.3" \
     --seed 37 \
     --path_to_model_weights "$WEIGHTS" \
     --model_name "$MODEL" >> "$OUT/run.log" 2>&1
}

# 2) natives first (high value), then the rest
run_subset native 2048 32
run_subset rest   256  32

echo "[$(date)] MPNN done, post-processing" | tee -a "$OUT/run.log"

# 3) aggregate -> designs_deep.csv joined to manifest metadata
$PY - <<'PYEOF' 2>>"$OUT/run.log"
import csv, re, glob, os
SRC="outputs/mpnn_nomhc_allbb"; OUT="outputs/mpnn_nomhc_allbb_deep"
meta={r["target"]:r for r in csv.DictReader(open(f"{SRC}/manifest.csv"))}
rows=[]
for fa in sorted(glob.glob(f"{OUT}/seqs/*.fa")):
    target=os.path.splitext(os.path.basename(fa))[0]; m=meta.get(target,{})
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
with open(f"{OUT}/designs_deep.csv","w",newline="") as o:
    w=csv.DictWriter(o,fieldnames=fields); w.writeheader(); w.writerows(rows)
from collections import Counter
print("total sequences:",len(rows))
print("unique peptides:",len(set(r["peptide"] for r in rows)))
print("per-structure depth:",dict(Counter(Counter(r["target"] for r in rows).values())))
PYEOF

echo "[$(date)] DONE" | tee -a "$OUT/run.log"
touch "$OUT/COMPLETE"
