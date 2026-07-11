#!/usr/bin/env python
"""Thread each 10-mer validated-binder into chain C of 6AMU/6AM5 and get ProteinMPNN NLL (score_only).
Writes outputs/panel_nll/panel_nll.csv : crystal, peptide, nll (mean per-residue NLL of designed chain)."""
import os, subprocess, glob, json, warnings, numpy as np
warnings.filterwarnings("ignore")
from Bio.PDB import PDBParser
ROOT="/home/ubuntu/if-mhc"; OUT=f"{ROOT}/outputs/panel_nll"; os.makedirs(OUT, exist_ok=True)
PY=f"/home/ubuntu/miniforge3/envs/esmcba/bin/python"
PANEL10=['ELAGIGILTV','SMLGIGIVPV','NMGGLGIMPV','ILEDRGFNQV','LMFDRGMSLL','MMWDRGLGMM','MMWDRGMGLL','SMAGIGIVDV','IMEDVGWLNV']
aa3={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
p=PDBParser(QUIET=True)
def chains(pid):
    m=p.get_structure(pid,f"{ROOT}/inputs/focus_6am/{pid}.pdb")[0]
    return {c.id:"".join(aa3.get(r.resname,"") for r in c if r.id[0]==' ' and r.resname in aa3) for c in m}

rows=[]
for pid in ["6AMU","6AM5"]:
    ch=chains(pid); order=sorted(ch)            # A,B,C,D,E
    od=f"{OUT}/{pid}"; os.makedirs(od, exist_ok=True)
    # parse the crystal once (chain_id: design C so the score reflects the peptide chain)
    subprocess.run([PY,f"{ROOT}/ProteinMPNN/helper_scripts/parse_multiple_chains.py",
                    f"--input_path={ROOT}/inputs/focus_6am",f"--output_path={od}/parsed.jsonl"],
                   stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    # keep only this pid in jsonl
    keep=[l for l in open(f"{od}/parsed.jsonl") if json.loads(l)["name"]==pid]
    open(f"{od}/parsed.jsonl","w").writelines(keep)
    cj={pid:[["C"],[c for c in order if c!="C"]]}; json.dump(cj,open(f"{od}/chain.jsonl","w"))
    for pep in PANEL10:
        fa=f"{od}/{pep}.fa"
        open(fa,"w").write(">x\n"+"/".join(pep if c=="C" else ch[c] for c in order)+"\n")
        subprocess.run([PY,f"{ROOT}/ProteinMPNN/protein_mpnn_run.py",
            "--jsonl_path",f"{od}/parsed.jsonl","--chain_id_jsonl",f"{od}/chain.jsonl",
            "--out_folder",od,"--score_only","1","--path_to_fasta",fa,
            "--model_name","v_48_020"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        # score_only writes .npz with 'score' (global) and per-fasta files
        npz=sorted(glob.glob(f"{od}/score_only/{pid}_fasta_*.npz"))
        if npz:
            d=np.load(npz[-1]); nll=float(np.mean(d["score"]))
            rows.append({"crystal":pid,"peptide":pep,"nll":round(nll,4)})
            os.rename(npz[-1], npz[-1]+f".{pep}")   # avoid clobber
            print(pid,pep,"NLL",round(nll,4),flush=True)
import csv
with open(f"{OUT}/panel_nll.csv","w",newline="") as o:
    csv.DictWriter(o,fieldnames=["crystal","peptide","nll"]).writeheader() if rows else None
    if rows: w=csv.DictWriter(o,fieldnames=["crystal","peptide","nll"]);w.writeheader();w.writerows(rows)
print("wrote",f"{OUT}/panel_nll.csv",len(rows),"rows")
