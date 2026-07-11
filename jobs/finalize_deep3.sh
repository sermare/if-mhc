#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/if-mhc
D1=outputs/mpnn_nomhc_allbb_deep        # natives@2048 + rest@256 (seed37)
D2=outputs/mpnn_nomhc_allbb_deep2       # rest@768 (seed38)
SRC=outputs/mpnn_nomhc_allbb
PY=/home/ubuntu/miniforge3/envs/esmcba/bin/python
NBPY=/home/ubuntu/miniforge3/envs/esmfold2/bin

start=$(date +%s)
until [ -f "$D2/logs/r0.DONE" ] && [ -f "$D2/logs/r1.DONE" ] && [ -f "$D2/logs/r2.DONE" ]; do
  sleep 60
  [ $(( $(date +%s) - start )) -gt 36000 ] && { echo "TIMEOUT" | tee -a "$D2/run.log"; exit 1; }
done

# guard: every rest backbone must have reached 768 in deep2 before merging
$PY - <<'PYEOF' || exit 2
import json, csv, os
SRC="outputs/mpnn_nomhc_allbb"; D2="outputs/mpnn_nomhc_allbb_deep2"
grp={r["target"]:r["group"] for r in csv.DictReader(open(f"{SRC}/manifest.csv"))}
rest=[t for t,g in grp.items() if g!="native"]
miss=[t for t in rest if (sum(1 for l in open(f"{D2}/seqs/{t}.fa") if l.startswith(">T=")) if os.path.exists(f"{D2}/seqs/{t}.fa") else 0) < 768]
import sys
if miss: print("STILL INCOMPLETE:",len(miss)); sys.exit(2)
print("all 592 rest backbones at 768")
PYEOF
echo "[$(date)] deep2 complete, merging deep + deep2" | tee -a "$D2/run.log"

# merge: pool every design line from BOTH seqs dirs per target -> combined designs_deep.csv
$PY - <<'PYEOF'
import csv, re, glob, os
SRC="outputs/mpnn_nomhc_allbb"
D1="outputs/mpnn_nomhc_allbb_deep"; D2="outputs/mpnn_nomhc_allbb_deep2"
meta={r["target"]:r for r in csv.DictReader(open(f"{SRC}/manifest.csv"))}
rows=[]
def harvest(seqdir):
    for fa in glob.glob(f"{seqdir}/*.fa"):
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
harvest(f"{D1}/seqs"); harvest(f"{D2}/seqs")
fields=["target","pid","group","cond","nice","pep_chain","toGIG","toDRG","min_native","seated","peptide","score","global_score"]
out=f"{D1}/designs_deep.csv"
with open(out,"w",newline="") as o:
    w=csv.DictWriter(o,fieldnames=fields); w.writeheader(); w.writerows(rows)
from collections import Counter
dep=Counter(r["target"] for r in rows)
print("combined designs:",len(rows),"| unique:",len(set(r['peptide'] for r in rows)))
print("depth histogram:",dict(Counter(dep.values())))
PYEOF

echo "[$(date)] rebuilding notebooks on combined data" | tee -a "$D2/run.log"
for b in build_nomhc_conditioning_logos build_nomhc_campaign_report; do $NBPY/python py/$b.py >> "$D2/run.log" 2>&1; done
for nb in nomhc_conditioning_analysis nomhc_campaign_report; do
  $NBPY/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 $nb.ipynb >> "$D2/run.log" 2>&1
done
echo "[$(date)] DONE finalize3" | tee -a "$D2/run.log"
touch "$D2/COMPLETE"
