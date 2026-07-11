#!/usr/bin/env python3
"""Make-or-break check before seeding RFD v5 from MD-drifted DRG frames.
Take the 6AMU (DRG) 370K MD frames that drift GIG-ward by RMSD (low toGIG). For THOSE specific frames ask:
 (1) F-pocket OCCUPANCY  swap = d(p10,Fc)-d(p9,Fc):  GIG/unshift ~ -3.3, DRG/shift ~ +3.5.
     still strongly +  -> distorted-DRG (p9 still in F-pocket) => seeding RFD there just recovers DRG.
     ~0 or negative    -> intermediate / p10-leaning = genuinely proto-GIG => worth seeding.
 (2) Is the drift PEPTIDE-only or whole-complex? Is the TCR POSE also GIG-shifted in those frames
     (closer to the 6AM5 TCR pose than 6AMU)? If yes, MD moved the one variable component GIG-ward too."""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,"py")
import numpy as np, mdtraj as mdt
import core_load as CL
from plot_components_grid import struct_ca, segments, fit_mhc, matchd, RA, RB, REF
FP_POS=[76,79,83,115,122,142]                       # 0-based in MHC a1a2 = authors 77,80,84,116,123,143
AA3={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V','MSE':'M'}

# crystal TCR poses (MHC-frame), matched to 6AMU ref, common indices
def tcr_vec(seq,ca,comA,comB):
    s=segments(seq,ca); R,t=fit_mhc(*s["mhc"])
    da=matchd(RA,s["tcra"][0],s["tcra"][1]@R+t); db=matchd(RB,s["tcrb"][0],s["tcrb"][1]@R+t)
    return da,db
sq5,ca5=struct_ca("inputs/focus_6am/6AM5.pdb"); sqU,caU=struct_ca("inputs/focus_6am/6AMU.pdb")
d5=tcr_vec(sq5,ca5,None,None); dU=tcr_vec(sqU,caU,None,None)
comA=sorted(set(d5[0])&set(dU[0])); comB=sorted(set(d5[1])&set(dU[1]))
def tvec(da,db): return np.concatenate([np.array([da[i] for i in comA]).ravel(),np.array([db[i] for i in comB]).ravel()])
T5=tvec(*d5); TU=tvec(*dU)
tcr_sep=np.sqrt(((T5-TU).reshape(-1,3)**2).sum(1).mean())
print(f"crystal TCR pose 6AM5<->6AMU = {tcr_sep:.2f} A ; GIG native swap ~ -3.3, DRG ~ +3.5\n")

def frame_metrics(seq, ca):
    s=segments(seq,ca)
    if s is None or s["pep"][1].shape[0]!=10: return None
    R,t=fit_mhc(*s["mhc"])
    if R is None: return None
    pep=s["pep"][1]@R+t
    toG=CL._rmsd(pep,CL.GIG); toD=CL._rmsd(pep,CL.DRG)
    mca=s["mhc"][1]; fc=mca[[i for i in FP_POS if i<len(mca)]].mean(0)   # F-pocket centroid (raw frame, internal)
    p=s["pep"][1]; swap=float(np.linalg.norm(p[9]-fc)-np.linalg.norm(p[8]-fc))
    da=matchd(RA,s["tcra"][0],s["tcra"][1]@R+t); db=matchd(RB,s["tcrb"][0],s["tcrb"][1]@R+t)
    if all(i in da for i in comA) and all(i in db for i in comB):
        tv=tvec(da,db); t5=np.sqrt(((tv-T5).reshape(-1,3)**2).sum(1).mean()); tu=np.sqrt(((tv-TU).reshape(-1,3)**2).sum(1).mean())
    else: t5=tu=np.nan
    return toG,toD,swap,t5,tu

for job in ["ifmhc_6AMU_md_370K","ifmhc_6AMU_md_300K"]:
    d=f"outputs/tamarind/results/{job}"
    t=mdt.load(f"{d}/traj_prod_no_water_seg1.xtc",top=f"{d}/topology_no_water.pdb")
    sel=t.topology.select("name CA"); seq="".join(r.code if r.code else "x" for r in t.topology.residues if any(a.name=="CA" for a in r.atoms))
    M=[]
    for fr in range(t.n_frames):
        m=frame_metrics(seq, t.xyz[fr,sel,:]*10.0)
        if m: M.append(m)
    M=np.array(M); toG,toD,swap,t5,tu=M.T
    print(f"=== {job}  ({len(M)} frames) ===")
    print(f"  toGIG range {toG.min():.2f}-{toG.max():.2f} | swap range {swap.min():+.2f} to {swap.max():+.2f} (DRG native +3.5)")
    print(f"  corr(toGIG, swap) = {np.corrcoef(toG,swap)[0,1]:+.2f}  (negative => GIG-ward frames DO lose the p9 anchor)")
    k=max(10,len(M)//10); idx=np.argsort(toG)[:k]                 # the GIG-WARD frames (lowest toGIG)
    gw=M[idx]
    print(f"  -- {k} most GIG-ward frames (lowest toGIG) --")
    print(f"     toGIG {gw[:,0].mean():.2f}  toDRG {gw[:,1].mean():.2f}  swap {gw[:,2].mean():+.2f} (min {gw[:,2].min():+.2f})")
    proto=(gw[:,2]<1.5).mean()*100; distort=(gw[:,2]>2.5).mean()*100
    print(f"     F-pocket verdict: {proto:.0f}% proto-GIG (swap<1.5) | {distort:.0f}% distorted-DRG (swap>2.5)")
    if np.isfinite(gw[:,3]).any():
        gigpose=(gw[:,3]<gw[:,4]).mean()*100
        print(f"     TCR pose: {gigpose:.0f}% of GIG-ward frames are pose-closer to 6AM5(GIG) than 6AMU(DRG) | mean t5={np.nanmean(gw[:,3]):.2f} tU={np.nanmean(gw[:,4]):.2f}")
    print()
