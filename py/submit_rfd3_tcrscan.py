#!/usr/bin/env python3
"""TCR-window conformational scan, ~300 designs, project IF-pMHC.

Biology (per user): N-term strongly anchors pMHC, C-term restrains into the pocket, the
MIDDLE/bulge is free and points to the TCR. So MHC hotspots = BRACKET ONLY
(N-term 159,66,70 + C-term 146); NO mid-floor (don't force the bulge). The TCR windows
drive the central conformation.

Scan (per structure): TCRa(D) anchor 30 always; slide the two CDR loops in +2 steps.
  TCRb(E) window x3 : {95,96} {97,98} {99,100}     (covers both numbering reads 95..100)
  TCRa(D) window x2 : CDR1 {28,30} | CDR3 {30,91,93}
  -> 6 combos x 2 structures = 12 jobs x 21 designs = 252
Cross-conditioning x2 (chase the OTHER peptide; same DMF5 TCR, residues transfer):
  6AMU MHC + 6AM5 TCR pattern (Da{30,31,50}+Eb{96,97,98})
  6AM5 MHC + 6AMU TCR pattern (Da{28,30,91,93}+Eb{99,100,101})
  -> 2 jobs x 24 designs = 48        TOTAL = 300
"""
import json, os, time, requests
API_KEY="REDACTED_TAMARIND_KEY"; HEADERS={"x-api-key":API_KEY}
BASE="https://app.tamarind.bio/api/"; ROOT="/home/ubuntu/if-mhc"
MHC="159 66 70 146"          # bracket: N-term anchor + C-term pocket restraint (both structures)
EB={"lo":"95 96","mid":"97 98","hi":"99 100"}
DA={"cdr1":"28 30","cdr3":"30 91 93"}

def upload(pid):
    remote=f"{pid}_scan.pdb"
    requests.put(BASE+f"upload/{remote}",headers=HEADERS,data=open(f"{ROOT}/inputs/focus_6am/{pid}.pdb","rb").read(),timeout=120)
    return remote
def submit(jn,remote,hot,n):
    settings={"task":"protein-binder-design","pdbFile":remote,"targetChains":["A","B","D","E"],
              "hotspots":hot,"binderLength":"10,10","ligands":[],"numDesigns":n}
    r=requests.post(BASE+"submit-job",headers=HEADERS,
                    json={"jobName":jn,"type":"rfdiffusion3","settings":settings,"project":"IF-pMHC"},timeout=120)
    return r.status_code,r.text[:120]

man=[]
for pid in ["6AMU","6AM5"]:
    remote=upload(pid)
    for dk,dv in DA.items():
        for ek,ev in EB.items():
            hot={"A":MHC,"D":dv,"E":ev}
            jn=f"ifmhc_rfd3_scan_{pid}_{dk}_{ek}"
            sc,txt=submit(jn,remote,hot,21)
            print(f"[{'OK' if sc==200 else 'ERR'}] {jn} D[{dv}] E[{ev}] -> {sc}")
            man.append({"jobName":jn,"pid":pid,"kind":"scan","hotspots":hot,"numDesigns":21,"ok":sc==200,"resp":txt})
            time.sleep(1)
# cross-conditioning
CROSS=[("ifmhc_rfd3_xcond_6AMU_as6AM5","6AMU",{"A":MHC,"D":"30 31 50","E":"96 97 98"}),
       ("ifmhc_rfd3_xcond_6AM5_as6AMU","6AM5",{"A":MHC,"D":"28 30 91 93","E":"99 100 101"})]
for jn,pid,hot in CROSS:
    remote=upload(pid)
    sc,txt=submit(jn,remote,hot,24)
    print(f"[{'OK' if sc==200 else 'ERR'}] {jn} (cross) D[{hot['D']}] E[{hot['E']}] -> {sc}")
    man.append({"jobName":jn,"pid":pid,"kind":"cross","hotspots":hot,"numDesigns":24,"ok":sc==200,"resp":txt})
    time.sleep(1)
json.dump(man,open(f"{ROOT}/outputs/tamarind/rfd3_tcrscan_manifest.json","w"),indent=2)
ok=sum(m["ok"] for m in man)
print(f"\n{ok}/{len(man)} jobs submitted | total designs ~{sum(m['numDesigns'] for m in man if m['ok'])} | manifest -> outputs/tamarind/rfd3_tcrscan_manifest.json")
