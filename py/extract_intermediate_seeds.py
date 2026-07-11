#!/usr/bin/env python3
"""Extract INTERMEDIATE (swap≈0, barrier-top) seed frames from the de-novo design-MD trajectories that
genuinely populate the mid-region (L4_8344724, L4_0161000, L5_5584195). Unlike native MD (firmly in one
register), these sit between the two — no home basin to relax into. Written as pMHC A/B/C seeds into
outputs/md_seeds/ (added to the marathon pool). Peptide-first layout (design MD)."""
import sys, os, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,"py")
import numpy as np, mdtraj as mdt
from Bio.PDB import PDBParser, PDBIO, Structure, Model, Chain
AA={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V','MSE':'M'}
FP=[77,80,84,116,123,143]; RES="outputs/tamarind/results"; OUT="outputs/md_seeds"; os.makedirs(OUT,exist_ok=True)
JOBS=["ifmhc_md_6AM5_L4_expanded_w1_8344724_0","ifmhc_md_6AM5_L4_expanded_w1_0161000_1","ifmhc_md_6AM5_L5_max_w1_5584195_1"]
NPICK=int(os.environ.get("NPICK","8")); io=PDBIO(); man=[]
for job in JOBS:
    d=f"{RES}/{job}"
    try: t=mdt.load(f"{d}/traj_prod_no_water_seg1.xtc",top=f"{d}/topology_no_water.pdb")
    except Exception as e: print("skip",job,e); continue
    res=list(t.topology.residues); seq="".join(AA.get(r.name,'x') for r in res)
    mhc=seq.find("SHSMRYFF"); b2m=seq.find("MIQRTP",mhc); ta=seq.find("EVEQNSGPL",b2m)
    lg=mhc>0 and seq[mhc-1]=="G"; mhc0=mhc-1 if lg else mhc; a0=1 if lg else 2
    ka=ta-1 if ta>0 and seq[ta-1]=="K" else ta
    pep0=0 if mhc0>=10 else ka-10
    r2ca={a.residue.index:a.index for a in t.topology.atoms if a.name=="CA"}
    fp=[r2ca[mhc0+(a-a0)] for a in FP]; p9=r2ca[pep0+8]; p10=r2ca[pep0+9]
    xyz=t.xyz*10.0; fc=xyz[:,fp,:].mean(1)
    swap=np.linalg.norm(xyz[:,p10,:]-fc,axis=1)-np.linalg.norm(xyz[:,p9,:]-fc,axis=1)
    picks=np.argsort(np.abs(swap))[:NPICK]                       # most-intermediate (|swap|→0)
    mhc_res=list(range(mhc0,mhc0+180)); b2m_res=list(range(b2m,b2m+100)); pep_res=list(range(pep0,pep0+10))
    keep=mhc_res+b2m_res+pep_res
    ai=t.topology.select(" or ".join(f"resid {r}" for r in keep)); sub=t.atom_slice(ai)
    for k in picks:
        tmp=f"{OUT}/_tmp_int.pdb"; sub[int(k)].save_pdb(tmp)
        m=PDBParser(QUIET=True).get_structure("x",tmp)[0]; allres=[r for c in m for r in c if r.id[0]==" "]
        st=Structure.Structure("s"); mdl=Model.Model(0); st.add(mdl)
        chs={c:Chain.Chain(c) for c in "ABC"}; [mdl.add(chs[c]) for c in "ABC"]
        for i,r in enumerate(allres):
            cid="A" if i<180 else ("B" if i<280 else "C"); rn=(i+1) if i<180 else ((i-179) if i<280 else (i-279))
            r2=r.copy(); r2.detach_parent(); r2.id=(" ",rn," "); chs[cid].add(r2)
        tag=job.split("ifmhc_md_")[1][:16]
        name=f"INT_{tag}_f{int(k)}_sw{swap[k]:+.1f}".replace("+","p").replace("-","m").replace(".","")
        io.set_structure(st); io.save(f"{OUT}/{name}.pdb"); man.append((name,round(float(swap[k]),2)))
        print(f"{name}: swap {swap[k]:+.2f}")
os.path.exists(f"{OUT}/_tmp_int.pdb") and os.remove(f"{OUT}/_tmp_int.pdb")
print(f"\nadded {len(man)} intermediate seeds to {OUT}/ (total now {len(os.listdir(OUT))} files)")
