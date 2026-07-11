#!/usr/bin/env python3
"""Extract RFdiffusion seed frames from the native MD (300/370K) trajectories. Per trajectory, pick the
frames MOST excursed toward the OTHER register (by F-pocket swap) — the best crossing head-start — plus a
couple median frames. Write each as a pMHC (MHC α1α2 + β2m + peptide; TCR dropped for speed) re-chained
A/B/C and renumbered A1-180 / B1-100 / C1-10 so RFD contigs are clean. → outputs/md_seeds/*.pdb + manifest."""
import sys, os, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,"py")
import numpy as np, mdtraj as mdt
from Bio.PDB import PDBParser, PDBIO, Select
AA={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V','MSE':'M'}
FP=[77,80,84,116,123,143]; RES="outputs/tamarind/results"; OUT="outputs/md_seeds"; os.makedirs(OUT,exist_ok=True)
NPICK=int(os.environ.get("NPICK","5"))
def locate(t):
    res=list(t.topology.residues); seq="".join(AA.get(r.name,'x') for r in res)
    mhc=seq.find("SHSMRYFF"); b2m=seq.find("MIQRTP",mhc); ta=seq.find("EVEQNSGPL",b2m)
    if min(mhc,b2m,ta)<0: return None
    lg=mhc>0 and seq[mhc-1]=="G"; mhc0=mhc-1 if lg else mhc; a0=1 if lg else 2
    ka=ta-1 if ta>0 and seq[ta-1]=="K" else ta
    pep0=0 if mhc0>=10 else ka-10
    return dict(mhc0=mhc0,b2m0=b2m,pep0=pep0,a0=a0,ridx=lambda auth: mhc0+(auth-a0))
io=PDBIO()
manifest=[]
for job in ["ifmhc_6AM5_md_300K","ifmhc_6AM5_md_370K","ifmhc_6AMU_md_300K","ifmhc_6AMU_md_370K"]:
    seed="6AM5" if "6AM5" in job else "6AMU"; T=job[-4:]
    d=f"{RES}/{job}"; t=mdt.load(f"{d}/traj_prod_no_water_seg1.xtc",top=f"{d}/topology_no_water.pdb")
    L=locate(t)
    if L is None: print("skip",job); continue
    r2ca={a.residue.index:a.index for a in t.topology.atoms if a.name=="CA"}
    fp=[r2ca[L["ridx"](a)] for a in FP]; p9=r2ca[L["pep0"]+8]; p10=r2ca[L["pep0"]+9]
    xyz=t.xyz*10.0; fc=xyz[:,fp,:].mean(1)
    swap=np.linalg.norm(xyz[:,p10,:]-fc,axis=1)-np.linalg.norm(xyz[:,p9,:]-fc,axis=1)
    # most excursed toward OTHER register: 6AM5(GIG,swap<0) -> max swap ; 6AMU(DRG,swap>0) -> min swap
    order=np.argsort(-swap) if seed=="6AM5" else np.argsort(swap)
    picks=list(order[:NPICK])+[int(np.argsort(np.abs(swap-np.median(swap)))[0])]   # excursed + 1 median
    # atom indices for pMHC: MHC α1α2 (180), β2m (100), peptide (10)
    mhc_res=list(range(L["mhc0"],L["mhc0"]+180)); b2m_res=list(range(L["b2m0"],L["b2m0"]+100)); pep_res=list(range(L["pep0"],L["pep0"]+10))
    keep=mhc_res+b2m_res+pep_res
    ai=t.topology.select(" or ".join(f"resid {r}" for r in keep))
    sub=t.atom_slice(ai)
    for k in picks:
        tmp=f"{OUT}/_tmp.pdb"; sub[k].save_pdb(tmp)
        m=PDBParser(QUIET=True).get_structure("x",tmp)[0]
        allres=[r for ch in m for r in ch if r.id[0]==" "]
        name=f"{seed}_{T}_f{int(k)}_sw{swap[k]:+.1f}".replace("+","p").replace("-","m").replace(".","")
        # build a clean 3-chain structure: A=MHC(1-180), B=β2m(1-100), C=peptide(1-10)
        from Bio.PDB import Structure, Model, Chain
        st=Structure.Structure("s"); mdl=Model.Model(0); st.add(mdl)
        chs={c:Chain.Chain(c) for c in "ABC"}; [mdl.add(chs[c]) for c in "ABC"]
        for i,r in enumerate(allres):
            cid="A" if i<180 else ("B" if i<280 else "C"); rn=(i+1) if i<180 else ((i-179) if i<280 else (i-279))
            r2=r.copy(); r2.detach_parent(); r2.id=(" ",rn," "); chs[cid].add(r2)
        io.set_structure(st); io.save(f"{OUT}/{name}.pdb")
        manifest.append((name,seed,round(float(swap[k]),2)))
        print(f"{name}: swap {swap[k]:+.2f}")
os.remove(f"{OUT}/_tmp.pdb") if os.path.exists(f"{OUT}/_tmp.pdb") else None
open(f"{OUT}/manifest.tsv","w").write("\n".join(f"{n}\t{s}\t{sw}" for n,s,sw in manifest)+"\n")
print(f"\nwrote {len(manifest)} MD seed frames to {OUT}/")
