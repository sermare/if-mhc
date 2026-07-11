#!/usr/bin/env python3
"""Pilot: submit OpenMM MD of the two native pMHC-TCR complexes (6AMU/DRG, 6AM5/GIG)
to Tamarind at 300 K and 370 K (4 jobs, 10 ns production each). Tests whether MD samples
the local basin and drifts toward the OTHER peptide conformation (DRG<->GIG gap = 2.4 A).
Upload: PUT /api/upload/<name> (binary) -> reference pdbFile by name in submit-job."""
import json, os, time, requests
API_KEY="REDACTED_TAMARIND_KEY"
HEADERS={"x-api-key":API_KEY}
BASE="https://app.tamarind.bio/api/"
OUT="outputs/tamarind"; os.makedirs(OUT,exist_ok=True)

def upload(path):
    name=os.path.basename(path)
    with open(path,"rb") as f:
        r=requests.put(BASE+f"upload/{name}",headers={**HEADERS,"Content-Type":"application/octet-stream"},
                       data=f.read(),timeout=180)
    return name, r.status_code, (r.text[:200] if r.text else "")

def submit(job_name, settings):
    params={"jobName":job_name,"type":"openmm","settings":settings}
    r=requests.post(BASE+"submit-job",headers=HEADERS,json=params,timeout=120)
    try: body=r.json()
    except ValueError: body=r.text
    return {"status_code":r.status_code,"ok":r.ok,"response":body,"request":params}

PDBS={"6AMU":"inputs/focus_6am/6AMU.pdb",   # DRG native complex
      "6AM5":"inputs/focus_6am/6AM5.pdb"}   # GIG native complex
TEMPS=["300","370"]   # 300K local basin; 370K to push toward the 2.4A gap
manifest=[]
for pid,path in PDBS.items():
    name,uc,utext=upload(path)
    print(f"[upload] {name}: HTTP {uc} {utext[:80]}")
    for T in TEMPS:
        jn=f"ifmhc_{pid}_md_{T}K"
        settings={"systemType":"protein","pdbFile":name,
                  "minimizationSteps":"10000","equilibrationTime":0.2,"productionTime":10,
                  "temperature":T,"removeWaters":"yes"}
        res=submit(jn,settings)
        tag="OK " if res["ok"] else "ERR"
        print(f"[{tag}] {jn} (HTTP {res['status_code']}) {str(res['response'])[:160]}")
        manifest.append({"jobName":jn,"pid":pid,"tempK":T,"upload_http":uc,**{k:res[k] for k in ("status_code","ok","response")}})
        json.dump(res,open(f"{OUT}/{jn}.json","w"),indent=2)
        time.sleep(1)
json.dump(manifest,open(f"{OUT}/md_pilot_manifest.json","w"),indent=2)
print(f"\n{sum(1 for m in manifest if m['ok'])}/{len(manifest)} jobs submitted OK -> {OUT}/md_pilot_manifest.json")
