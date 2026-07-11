#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/if-mhc
OUT=outputs/mpnn_nomhc_allbb_deep
SRC=outputs/mpnn_nomhc_allbb
PY=/home/ubuntu/miniforge3/envs/esmcba/bin/python
NBPY=/home/ubuntu/miniforge3/envs/esmfold2/bin

# wait for the resubmitted remainder shard
start=$(date +%s)
until [ -f "$OUT/logs/rest0c.DONE" ]; do
  sleep 30
  [ $(( $(date +%s) - start )) -gt 14400 ] && { echo "TIMEOUT" | tee -a "$OUT/run.log"; exit 1; }
done

# safety: confirm every target reached its depth before declaring done
$PY - <<'PYEOF'
import csv, os, sys
SRC="outputs/mpnn_nomhc_allbb"; OUT="outputs/mpnn_nomhc_allbb_deep"
meta={r["target"]:r for r in csv.DictReader(open(f"{SRC}/manifest.csv"))}
miss=[t for t,m in meta.items()
      if (sum(1 for l in open(f"{OUT}/seqs/{t}.fa") if l.startswith(">T=")) if os.path.exists(f"{OUT}/seqs/{t}.fa") else 0)
         < (2048 if m["group"]=="native" else 256)]
if miss:
    print("STILL MISSING:", len(miss)); sys.exit(2)
print("all 614 complete")
PYEOF

echo "[$(date)] all complete, aggregating" | tee -a "$OUT/run.log"
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
fields=["target","pid","group","cond","nice","pep_chain","toGIG","toDRG","min_native","seated","peptide","score","global_score"]
with open(f"{OUT}/designs_deep.csv","w",newline="") as o:
    w=csv.DictWriter(o,fieldnames=fields); w.writeheader(); w.writerows(rows)
from collections import Counter
print("total sequences:",len(rows),"| unique:",len(set(r['peptide'] for r in rows)))
print("per-structure depth histogram:",dict(Counter(Counter(r['target'] for r in rows).values())))
PYEOF

for builder in build_nomhc_conditioning_logos build_nomhc_campaign_report; do
  $NBPY/python py/$builder.py >> "$OUT/run.log" 2>&1
done
for nb in nomhc_conditioning_analysis nomhc_campaign_report; do
  $NBPY/jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1800 $nb.ipynb >> "$OUT/run.log" 2>&1
done
echo "[$(date)] DONE finalize2" | tee -a "$OUT/run.log"
touch "$OUT/COMPLETE"
