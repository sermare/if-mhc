#!/usr/bin/env bash
# Idempotent supervisor for the noMHC deep2 (rest -> 768, seed 38) run.
# Safe to run repeatedly from cron: resubmits missing work, or merges + builds
# notebooks when complete, then removes itself from cron. Survives CC teardown.
set -uo pipefail
cd /home/ubuntu/if-mhc
PY=/home/ubuntu/miniforge3/envs/esmcba/bin/python
NBPY=/home/ubuntu/miniforge3/envs/esmfold2/bin
SRC=outputs/mpnn_nomhc_allbb
D1=outputs/mpnn_nomhc_allbb_deep
D2=outputs/mpnn_nomhc_allbb_deep2
W=ProteinMPNN/nomhc_model_weights/
mkdir -p "$D2/logs"
log(){ echo "[$(date)] $*" >> "$D2/resume.log"; }

[ -f "$D2/COMPLETE" ] && { log "already COMPLETE; exiting"; exit 0; }

# count incomplete rest backbones (<768)
INCOMPLETE=$($PY - <<'PYEOF'
import json,os,csv
SRC="outputs/mpnn_nomhc_allbb"; D2="outputs/mpnn_nomhc_allbb_deep2"
grp={r["target"]:r["group"] for r in csv.DictReader(open(f"{SRC}/manifest.csv"))}
rest=[t for t,g in grp.items() if g!="native"]
miss=[t for t in rest if (sum(1 for l in open(f"{D2}/seqs/{t}.fa") if l.startswith(">T=")) if os.path.exists(f"{D2}/seqs/{t}.fa") else 0) < 768]
open(f"{D2}/_missing.txt","w").write("\n".join(miss))
print(len(miss))
PYEOF
)
log "incomplete rest backbones: $INCOMPLETE"

if [ "$INCOMPLETE" -gt 0 ]; then
    # if MPNN already running, let it continue
    if pgrep -f "protein_mpnn_run.py" >/dev/null; then log "MPNN already running; leaving it"; exit 0; fi
    # split missing into 3 shards and launch (batch 12, safe memory)
    $PY - <<'PYEOF'
import json,csv
SRC="outputs/mpnn_nomhc_allbb"; D2="outputs/mpnn_nomhc_allbb_deep2"
parsed={json.loads(l)["name"]:l for l in open(f"{SRC}/parsed_chains.jsonl")}
asg=json.load(open(f"{SRC}/assigned_chains.jsonl"))
miss=[t for t in open(f"{D2}/_missing.txt").read().split("\n") if t]
for s in range(3):
    ts=miss[s::3]
    open(f"{D2}/parsed_r{s}.jsonl","w").writelines(parsed[t] for t in ts)
    json.dump({t:asg[t] for t in ts}, open(f"{D2}/assigned_r{s}.jsonl","w"))
PYEOF
    for s in 0 1 2; do
        setsid bash jobs/run_shard2.sh "$D2" "$W" proteinmpnn_nomhc "r$s" 768 12 38 </dev/null >/dev/null 2>&1 &
    done
    log "launched 3 remainder shards (batch12)"
    exit 0
fi

# ---- all rest complete: merge deep + deep2, build notebooks, finish ----
if pgrep -f "protein_mpnn_run.py" >/dev/null; then log "waiting for stragglers"; exit 0; fi
log "all rest at 768; merging"
$PY - <<'PYEOF'
import csv, re, glob, os
SRC="outputs/mpnn_nomhc_allbb"; D1="outputs/mpnn_nomhc_allbb_deep"; D2="outputs/mpnn_nomhc_allbb_deep2"
meta={r["target"]:r for r in csv.DictReader(open(f"{SRC}/manifest.csv"))}
rows=[]
def harvest(sd):
    for fa in glob.glob(f"{sd}/*.fa"):
        t=os.path.splitext(os.path.basename(fa))[0]; m=meta.get(t,{})
        L=open(fa).read().splitlines()
        for i in range(0,len(L)-1,2):
            if not L[i].startswith(">T="): continue
            d=dict(re.findall(r'(\w+)=([-\d.]+)',L[i]))
            rows.append({"target":t,"pid":m.get("pid"),"group":m.get("group"),"cond":m.get("cond"),
                "nice":m.get("nice"),"pep_chain":m.get("pep_chain"),"toGIG":m.get("toGIG"),
                "toDRG":m.get("toDRG"),"min_native":m.get("min_native"),"seated":m.get("seated"),
                "peptide":L[i+1].strip(),"score":d.get("score"),"global_score":d.get("global_score")})
harvest(f"{D1}/seqs"); harvest(f"{D2}/seqs")
f=["target","pid","group","cond","nice","pep_chain","toGIG","toDRG","min_native","seated","peptide","score","global_score"]
import csv as C
w=C.DictWriter(open(f"{D1}/designs_deep.csv","w",newline=""),fieldnames=f); w.writeheader(); w.writerows(rows)
from collections import Counter
print("combined",len(rows),"depths",dict(Counter(Counter(r['target'] for r in rows).values())))
PYEOF
log "merged; building notebooks"
for b in build_nomhc_conditioning_logos build_nomhc_campaign_report; do $NBPY/python py/$b.py >> "$D2/resume.log" 2>&1; done
for nb in nomhc_conditioning_analysis nomhc_campaign_report; do
    $NBPY/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 $nb.ipynb >> "$D2/resume.log" 2>&1
done
touch "$D2/COMPLETE"
log "DONE; removing cron entry"
crontab -l 2>/dev/null | grep -v resume_deep2.sh | crontab -
