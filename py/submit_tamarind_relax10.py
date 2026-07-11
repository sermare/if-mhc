#!/usr/bin/env python3
"""Stage 1 of the 10-candidate plan: thread each candidate's top MPNN sequence onto its peptide
(chain A), build a full PDB, upload, and submit to Tamarind openmm-relax (fast triage).
Survivors (low-energy / conformation held) go to full openmm MD 10ns in stage 2."""
import json, os, re, glob, time, warnings, requests
warnings.filterwarnings("ignore")
from Bio.PDB import PDBParser, PDBIO
API_KEY=os.environ.get("TAMARIND_API_KEY",""); H={"x-api-key":API_KEY}; B="https://app.tamarind.bio/api/"
OUT="outputs/tamarind"; FULL=f"{OUT}/cand_full"; os.makedirs(FULL,exist_ok=True)
AA1to3={'A':'ALA','R':'ARG','N':'ASN','D':'ASP','C':'CYS','Q':'GLN','E':'GLU','G':'GLY','H':'HIS','I':'ILE','L':'LEU','K':'LYS','M':'MET','F':'PHE','P':'PRO','S':'SER','T':'THR','W':'TRP','Y':'TYR','V':'VAL'}
_p=PDBParser(QUIET=True)

def top_seq(fa):
    ls=open(fa).read().splitlines(); best=(9e9,None)
    for i in range(0,len(ls)-1,2):
        if "sample=" in ls[i]:
            m=re.search(r"score=([-\d.]+)",ls[i]); s=float(m.group(1)) if m else 9e9
            if s<best[0]: best=(s,ls[i+1].strip())
    return best[1]

def thread(pdb, seq, out):
    st=_p.get_structure("x",pdb)[0]
    chA=[r for r in st["A"] if r.id[0]==' ']
    for r,aa in zip(chA,seq): r.resname=AA1to3[aa]
    io=PDBIO(); io.set_structure(st); io.save(out)

def upload(path):
    name=os.path.basename(path)
    r=requests.put(B+f"upload/{name}",headers={**H,"Content-Type":"application/octet-stream"},data=open(path,"rb").read(),timeout=180)
    return name, r.status_code

def submit(job, settings, typ="openmm-relax"):
    r=requests.post(B+"submit-job",headers=H,json={"jobName":job,"type":typ,"settings":settings},timeout=120)
    try: body=r.json()
    except ValueError: body=r.text
    return {"status_code":r.status_code,"ok":r.ok,"response":body}

cands=json.load(open(f"{OUT}/candidates10.json"))
manifest=[]
for pdb in cands:
    base=os.path.splitext(os.path.basename(pdb))[0]
    fa=f"{OUT}/seqs/{base}.fa"
    if not os.path.exists(fa): print(f"  no seq for {base}, skip"); continue
    seq=top_seq(fa)
    full=f"{FULL}/{base}_threaded.pdb"; thread(pdb,seq,full)
    name,uc=upload(full)
    jn=f"ifmhc_relax_{base}"[:60]
    res=submit(jn,{"pdbFile":name})
    tag="OK " if res["ok"] else "ERR"
    print(f"[{tag}] {jn} seq={seq} (up {uc}, HTTP {res['status_code']}) {str(res['response'])[:120]}")
    manifest.append({"jobName":jn,"backbone":base,"seq":seq,"upload_http":uc,**{k:res[k] for k in ('status_code','ok','response')}})
    time.sleep(1)
json.dump(manifest,open(f"{OUT}/relax10_manifest.json","w"),indent=2)
print(f"\n{sum(1 for m in manifest if m['ok'])}/{len(manifest)} openmm-relax jobs submitted -> {OUT}/relax10_manifest.json")
