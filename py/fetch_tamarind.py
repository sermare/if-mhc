#!/usr/bin/env python3
"""Poll our Tamarind jobs (MD pilot + 10 relax candidates) and download completed results."""
import json, os, io, zipfile, sys, requests
H={"x-api-key":"REDACTED_TAMARIND_KEY"}; B="https://app.tamarind.bio/api/"
OUT="outputs/tamarind"; RES=f"{OUT}/results"; os.makedirs(RES,exist_ok=True)
TERMINAL={"Complete","Stopped","Failed","Error"}

def jobs():
    js=[]
    for mf in ["rfd3_native_manifest.json"]:
        p=f"{OUT}/{mf}"
        if os.path.exists(p):
            for m in json.load(open(p)):
                if m.get("ok"): js.append(m["jobName"])
    return sorted(set(js))

def status(jn):
    try:
        r=requests.get(B+"jobs",headers=H,params={"jobName":jn},timeout=60)
        d=r.json()
        if isinstance(d,list) and d: d=d[0]
        return d.get("JobStatus") or d.get("status") or str(d)[:40]
    except Exception as e: return f"ERR {e}"

def fetch(jn):
    d=f"{RES}/{jn}"
    if os.path.exists(f"{d}/.done"): return "cached"
    os.makedirs(d,exist_ok=True)
    try:
        r=requests.post(B+"result",headers=H,json={"jobName":jn},timeout=120); url=r.json()
        url=url.get("url") if isinstance(url,dict) else url
        z=requests.get(url,timeout=300); zipfile.ZipFile(io.BytesIO(z.content)).extractall(d)
        open(f"{d}/.done","w").close(); return "downloaded"
    except Exception as e: return f"fetch-err {e}"

if __name__=="__main__":
    js=jobs(); done=0
    for jn in js:
        st=status(jn)
        if st in ("Complete","Succeeded"):
            print(f"  {jn}: {st} -> {fetch(jn)}"); done+=1
        else:
            print(f"  {jn}: {st}")
    print(f"\n{done}/{len(js)} complete & fetched")
