#!/usr/bin/env python3
"""RFdiffusion (v1) Partial-Diffusion NOISE SWEEP on Tamarind, project IF-pMHC.

Reframed experiment (not a single setting): map where on the partial_T noise axis, if
anywhere, register transfer (source->other peptide geometry) appears -- and whether it
appears BEFORE the peptide ejects from the groove (the predicted "squeeze").

Two conditioning arms (250 each): L2 (Nterm+1TCR) vs L4 (Nterm+pocket+3TCR) hotspot
guidance during denoising. Both native seeds (6AMU/DRG, 6AM5/GIG) so both crossing
directions are observed. diffusedResidues = peptide only (C1-10); MHC+TCR held fixed.

Instruments (applied to results, NOT here): (1) in-groove burial vs partial_T, (2)
register SIGNATURE match to target crystal (anchor-pocket + bulge pattern), NOT mere
RMSD-to-other shrinkage. Hold prediction: low T = seated+locked; high T = crosses only
after/with ejection.
"""
import json, os, time, requests, re
KEY=re.search(r'API_KEY\s*=\s*"([^"]+)"',open("/home/ubuntu/if-mhc/py/submit_rfd3_groove.py").read()).group(1)
H={"x-api-key":KEY}; B="https://app.tamarind.bio/api/"; ROOT="/home/ubuntu/if-mhc"

# L2/L4 hotspot sets (RFdiffusion comma-string format), per crystal (from outputs/ladder/run.log)
HOT={
 "L2":{"6AMU":"A159,A66,A70,E100","6AM5":"A159,A66,A70,E97"},
 "L4":{"6AMU":"A159,A66,A70,A9,A77,A80,E100,D30,E99","6AM5":"A159,A66,A70,A9,A77,A80,E97,D30,E96"},
}
PARTIAL_T=[2,5,10,15,20,25,30,40]
SEEDS=["6AMU","6AM5"]
NDES=16   # 2 arms x 2 seeds x 8 T x 16 = 512 (~256 per arm)

def upload(pid):
    remote=f"{pid}_pd.pdb"
    requests.put(B+f"upload/{remote}",headers=H,data=open(f"{ROOT}/inputs/focus_6am/{pid}.pdb","rb").read(),timeout=120)
    return remote

def submit(jn,remote,hot,T,n):
    s={"task":"Partial Diffusion","pdbFile":remote,"designedChain":"C","diffusedResidues":"C1-10",
       "hotspots":hot,"partial_T":T,"numDesigns":n}
    r=requests.post(B+"submit-job",headers=H,json={"jobName":jn,"type":"rfdiffusion","settings":s,"project":"IF-pMHC"},timeout=120)
    return r.status_code,r.text[:120]

def main():
    remotes={pid:upload(pid) for pid in SEEDS}
    man=[]; ok=0
    for arm in ["L2","L4"]:
        for pid in SEEDS:
            for T in PARTIAL_T:
                jn=f"ifmhc_pd_{arm}_{pid}_T{T}"
                sc,txt=submit(jn,remotes[pid],HOT[arm][pid],T,NDES)
                ok+=sc==200
                print(f"[{'OK' if sc==200 else 'ERR'}] {jn} hot=[{HOT[arm][pid]}] -> {sc} {txt}")
                man.append({"jobName":jn,"arm":arm,"pid":pid,"partial_T":T,"hotspots":HOT[arm][pid],
                            "numDesigns":NDES,"ok":sc==200,"resp":txt})
                time.sleep(1)
    json.dump(man,open(f"{ROOT}/outputs/tamarind/pd_sweep_manifest.json","w"),indent=2)
    print(f"\n{ok}/{len(man)} jobs OK | ~{sum(m['numDesigns'] for m in man if m['ok'])} designs | manifest -> outputs/tamarind/pd_sweep_manifest.json")

if __name__=="__main__":
    main()
