#!/usr/bin/env python3
"""Extract FULL-COMPLEX (TCR kept) seed frames from MD, renumbered to uniform chains so RFD contigs are
fixed: A=MHC a1a2 (1-180), B=b2m (1-100), C=peptide (1-10), D=TCRa Vdomain (1-110), E=TCRb Vdomain (1-115).
Two seed classes: INT = design-MD barrier-top frames (swap≈0, no home basin); NAT = native-MD frames most
excursed toward the other register. Both carry the TCR interface. -> inputs/full_seeds/*.pdb
Uniform contig: 'A1-180/0 B1-100/0 C1-10/0 D1-110/0 E1-115'  provide_seq (A,B,D,E) '0-279,290-514'."""
import os, warnings, sys; warnings.filterwarnings("ignore"); sys.path.insert(0,"py")
import numpy as np, mdtraj as mdt
from Bio.PDB import PDBParser, PDBIO, Structure, Model, Chain
AA={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V','MSE':'M'}
FP=[77,80,84,116,123,143]; RES="outputs/tamarind/results"; OUT="inputs/full_seeds"; os.makedirs(OUT,exist_ok=True)
LA,LB,LC,LD,LE=180,100,10,110,115
INT=["ifmhc_md_6AM5_L4_expanded_w1_8344724_0","ifmhc_md_6AM5_L4_expanded_w1_0161000_1","ifmhc_md_6AM5_L5_max_w1_5584195_1"]
NAT=["ifmhc_6AM5_md_370K","ifmhc_6AMU_md_370K","ifmhc_6AM5_md_300K","ifmhc_6AMU_md_300K"]
io=PDBIO()
def seg(t):
    res=list(t.topology.residues); s="".join(AA.get(r.name,'x') for r in res)
    mhc=s.find("SHSMRYFF"); b2=s.find("MIQRTP",mhc); ta=s.find("EVEQNSGPL",b2); tb=s.find("IAGITQAP",ta)
    if min(mhc,b2,ta,tb)<0: return None
    lg=mhc>0 and s[mhc-1]=="G"; mhc0=mhc-1 if lg else mhc; a0=1 if lg else 2
    ka=ta-1 if ta>0 and s[ta-1]=="K" else ta
    pep0=0 if mhc0>=10 else ka-10
    return dict(res=res, mhc0=mhc0, b2=b2, pep0=pep0, ta=ka, tb=tb, ridx=lambda a: mhc0+(a-a0))
def write(t, fr, name):
    L=seg(t)
    if L is None: return None
    if max(L["mhc0"]+LA, L["b2"]+LB, L["pep0"]+LC, L["ta"]+LD, L["tb"]+LE) > t.n_residues: return None
    tmp=f"{OUT}/_tmp.pdb"; t[fr].save_pdb(tmp)                       # save FULL frame, index by original position
    allres=[r for c in PDBParser(QUIET=True).get_structure("x",tmp)[0] for r in c if r.id[0]==" "]
    segs=[("A",L["mhc0"],LA),("B",L["b2"],LB),("C",L["pep0"],LC),("D",L["ta"],LD),("E",L["tb"],LE)]
    st=Structure.Structure("s"); mdl=Model.Model(0); st.add(mdl); chs={c:Chain.Chain(c) for c in "ABCDE"}; [mdl.add(chs[c]) for c in "ABCDE"]
    for cid,start,n in segs:
        for j in range(n):
            r=allres[start+j]; r2=r.copy(); r2.detach_parent(); r2.id=(" ",j+1," "); chs[cid].add(r2)
    io.set_structure(st); io.save(f"{OUT}/{name}.pdb"); return True
def swaps(t,L):
    r2ca={a.residue.index:a.index for a in t.topology.atoms if a.name=="CA"}
    fp=[r2ca[L["ridx"](a)] for a in FP]; p9=r2ca[L["pep0"]+8]; p10=r2ca[L["pep0"]+9]
    xyz=t.xyz*10.0; fc=xyz[:,fp,:].mean(1)
    return np.linalg.norm(xyz[:,p10,:]-fc,axis=1)-np.linalg.norm(xyz[:,p9,:]-fc,axis=1)
n=0
for job in INT:
    t=mdt.load(f"{RES}/{job}/traj_prod_no_water_seg1.xtc",top=f"{RES}/{job}/topology_no_water.pdb"); L=seg(t)
    if L is None: continue
    sw=swaps(t,L); picks=np.argsort(np.abs(sw))[:6]
    for k in picks:
        if write(t,int(k),f"INT_{job.split('ifmhc_md_')[1][:14]}_f{int(k)}"): n+=1
for job in NAT:
    t=mdt.load(f"{RES}/{job}/traj_prod_no_water_seg1.xtc",top=f"{RES}/{job}/topology_no_water.pdb"); L=seg(t)
    if L is None: continue
    sw=swaps(t,L); seed="6AM5" if "6AM5" in job else "6AMU"
    order=np.argsort(-sw) if seed=="6AM5" else np.argsort(sw)  # most toward other register
    for k in order[:4]:
        if write(t,int(k),f"NAT_{seed}_{job[-4:]}_f{int(k)}"): n+=1
os.path.exists(f"{OUT}/_tmp.pdb") and os.remove(f"{OUT}/_tmp.pdb")
print(f"wrote {n} full-complex seeds to {OUT}/ ({len(os.listdir(OUT))} files)")
