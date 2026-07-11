#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/if-mhc
OUT=outputs/mpnn_nomhc_allbb_deep
SRC=outputs/mpnn_nomhc_allbb
PY=/home/ubuntu/miniforge3/envs/esmcba/bin/python
NBPY=/home/ubuntu/miniforge3/envs/esmfold2/bin

# wait for all 4 shard DONE markers
start=$(date +%s)
until [ -f "$OUT/logs/native.DONE" ] && [ -f "$OUT/logs/rest0.DONE" ] && \
      [ -f "$OUT/logs/rest1.DONE" ] && [ -f "$OUT/logs/rest2.DONE" ]; do
  sleep 60
  [ $(( $(date +%s) - start )) -gt 36000 ] && { echo "TIMEOUT" | tee -a "$OUT/run.log"; break; }
done
echo "[$(date)] all shards done, aggregating" | tee -a "$OUT/run.log"

# aggregate seqs -> designs_deep.csv joined to manifest
$PY - <<'PYEOF'
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
print("total sequences:",len(rows),"| unique:",len(set(r['peptide'] for r in rows)))
print("per-structure depth:",dict(Counter(Counter(r['target'] for r in rows).values())))
PYEOF

# rebuild + execute the analysis notebook on the deep data
$NBPY/python py/build_nomhc_conditioning_logos.py
$NBPY/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1200 nomhc_conditioning_analysis.ipynb >> "$OUT/run.log" 2>&1

echo "[$(date)] DONE finalize" | tee -a "$OUT/run.log"
touch "$OUT/COMPLETE"
