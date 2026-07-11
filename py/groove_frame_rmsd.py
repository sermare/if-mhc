#!/usr/bin/env python
"""Groove-frame (register-preserving) peptide Cα-RMSD. The de-novo MHC is FIXED to its source crystal,
so each peptide already sits in that crystal's groove frame. We map every 6AM5-frame peptide into the
6AMU groove via the residue-id-matched 6AM5->6AMU MHC α1/α2 transform, then take DIRECT (no best-fit)
peptide RMSD to DRG and to GIG (GIG also mapped into the 6AMU frame). This keeps the register that the
best-fit RMSD discards. Writes outputs/struct_ood/groove_dual_rmsd.csv."""
import glob,os,csv,warnings,numpy as np
warnings.filterwarnings("ignore")
from Bio.PDB import PDBParser
ROOT="/home/ubuntu/if-mhc"; _p=PDBParser(QUIET=True)
TOK2COND={"mhc":"MHCpocket(7)","tcr2":"TCR2(2)","mhc_tcr2":"MHC+TCR(9)","L1_nterm":"Nterm(3)","L2_nterm_t1":"Nterm+TCR1(4)","L3_nterm_t2":"Nterm+TCR2(5)","L4_expanded":"Nterm+pkt+TCR3(9)","L5_max":"Nterm+fullpkt+TCR4(12)"}
def cond_label(name):
    b="_"+os.path.basename(name)
    for t in ["mhc_tcr2","L1_nterm","L2_nterm_t1","L3_nterm_t2","L4_expanded","L5_max","tcr2","mhc"]:
        if f"_{t}_" in b: return TOK2COND[t]
    return None
def cca(m,ch,lo=None,hi=None):
    if ch not in m: return None
    return np.array([r['CA'].coord for r in m[ch] if r.id[0]==' ' and 'CA' in r and (lo is None or lo<=r.id[1]<=hi)])
def RT(P,Q):
    Pc=P.mean(0);Qc=Q.mean(0);P0=P-Pc;Q0=Q-Qc;V,S,Wt=np.linalg.svd(P0.T@Q0);d=np.sign(np.linalg.det(V@Wt));R=V@np.diag([1,1,d])@Wt;return R,Qc-Pc@R
def rmsd(P,Q): return float(np.sqrt(((P-Q)**2).sum()/len(P)))

mU=_p.get_structure("U",f"{ROOT}/inputs/focus_6am/6AMU.pdb")[0]
m5=_p.get_structure("5",f"{ROOT}/inputs/focus_6am/6AM5.pdb")[0]
REF={"6AMU":cca(mU,"A",2,180),"6AM5":cca(m5,"A",2,180)}  # each crystal's MHC α1/α2 (res 2-180, id-matched)
DRG=cca(mU,"C")
R5,t5=RT(REF["6AM5"],REF["6AMU"])                        # 6AM5 crystal frame -> 6AMU groove frame
GIGref=cca(m5,"C")@R5+t5                                  # GIG peptide in the 6AMU groove frame
print(f"6AM5->6AMU groove align: {rmsd(REF['6AM5']@R5+t5,REF['6AMU']):.2f} Å | DRG-GIG groove RMSD: {rmsd(DRG,GIGref):.2f} Å")

def mhc_window(m,chain,pid):
    """MHC α1/α2 Cα for superposition. de-novo/candidate: positional chain B (6AMU trim A2-180 -> [0:179];
    6AM5 trim A1-180 -> [1:180]). relaxed/native: chain A by residue id 2-180."""
    if chain=="A_resid": return cca(m,"A",2,180)
    cab=cca(m,chain)
    if cab is None: return None
    return cab[0:179] if pid=="6AMU" else cab[1:180]
def add(pid,origin,stage,pep,mhc):
    if pep is None or mhc is None or len(pep)!=10 or len(mhc)<150: return
    n=min(len(mhc),len(REF[pid])); R,t=RT(mhc[:n],REF[pid][:n])   # struct MHC -> own crystal MHC frame
    pa=pep@R+t                                                    # peptide in own crystal frame
    if pid=="6AM5": pa=pa@R5+t5                                   # -> 6AMU groove frame
    rows.append({"crystal":pid,"origin":origin,"stage":stage,"to_DRG":round(rmsd(pa,DRG),2),"to_GIG":round(rmsd(pa,GIGref),2)})
rows=[]
# de-novo (peptide chain A, MHC chain B)
for d in ["outputs/grind/pdb","outputs/ladder/pdb","outputs/promising/pdb"]:
    for f in glob.glob(f"{ROOT}/{d}/*.pdb"):
        if "_split" in f: continue
        lab=cond_label(f)
        if not lab: continue
        m=_p.get_structure("x",f)[0]; pid="6AMU" if "/6AMU" in f else "6AM5"
        add(pid,lab,"de-novo",cca(m,"A"),mhc_window(m,"B",pid))
# natives (reference: 0 by construction for own native)
add("6AMU","crystal","crystal",DRG,REF["6AMU"]); add("6AM5","crystal","crystal",cca(m5,"C"),REF["6AM5"])
# relaxed snapshots (peptide chain C, MHC chain A)
for g in ["outputs/focus_relax/snapshots","outputs/_relax_gap_peek/snaps","outputs/relax_campaign/snapshots"]:
    for f in glob.glob(f"{ROOT}/{g}/6AM*_snap*.pdb"):
        m=_p.get_structure("x",f)[0]; pid="6AMU" if "6AMU" in f else "6AM5"
        add(pid,"relaxed","relaxed",cca(m,"C"),mhc_window(m,"A_resid",pid))
# openmm-relax + rosetta-relax candidates (peptide chain A, MHC chain B)
for f in glob.glob(f"{ROOT}/outputs/tamarind/results/ifmhc_relax_*/relaxed.pdb"):
    base=os.path.basename(os.path.dirname(f)).replace("ifmhc_relax_",""); pid="6AMU" if base.startswith("6AMU") else "6AM5"
    m=_p.get_structure("x",f)[0]; add(pid,cond_label(base),"openmm-relax",cca(m,"A"),mhc_window(m,"B",pid))
for d in glob.glob(f"{ROOT}/outputs/tamarind/results/ifmhc_rosrlx_*"):
    base=os.path.basename(d).replace("ifmhc_rosrlx_",""); pid="6AMU" if base.startswith("6AMU") else "6AM5"
    pdbs=glob.glob(f"{d}/*/input_relaxed_0001.pdb")
    if pdbs:
        m=_p.get_structure("x",pdbs[0])[0]; add(pid,cond_label(base),"rosetta-relax",cca(m,"A"),mhc_window(m,"B",pid))
out=f"{ROOT}/outputs/struct_ood/groove_dual_rmsd.csv"
with open(out,"w",newline="") as o:
    w=csv.DictWriter(o,fieldnames=["crystal","origin","stage","to_DRG","to_GIG"]);w.writeheader();w.writerows(rows)
print(f"wrote {out} ({len(rows)} structures)")
import pandas as pd; print(pd.DataFrame(rows).groupby("stage")[["to_DRG","to_GIG"]].agg(["mean","min"]).round(2))
