#!/usr/bin/env python3
"""Score B3 anchor-conditioned RFD outputs: coherent backbone? which register?
Each output = merged chain [MHC(a1a2) | b2m | peptide(10)]. Superpose MHC floor (core_load.to_common),
peptide Cα-RMSD toGIG/toDRG (basin map). Coherence = consecutive peptide CA-CA distance (~3.8 Å ok;
>4.5 or <3.0 = broken). Reports, per target, %coherent and the toGIG/toDRG distribution + basin calls."""
import sys, glob, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,"py")
import numpy as np; from Bio.PDB import PDBParser; import core_load as CL
p=PDBParser(QUIET=True)
def score(f):
    m=p.get_structure("x",f)[0]; res=[r for c in m for r in c if r.id[0]==" " and "CA" in r]
    if len(res)<200: return None
    pep=res[:10]; mhc=res[10:-100]                      # chain A = peptide(10); chain B = MHC α1α2 (drop trailing β2m 100)
    ca=np.array([r["CA"].coord for r in pep])
    dd=np.linalg.norm(np.diff(ca,axis=0),axis=1)        # consecutive CA-CA
    coherent = bool((dd>3.0).all() and (dd<4.5).all())
    pc,fit=CL.to_common(mhc,pep)
    if pc is None: return dict(coherent=coherent, g=np.nan, d=np.nan, fit=np.nan, capap=float(dd.max()))
    return dict(coherent=coherent, g=CL._rmsd(pc,CL.GIG), d=CL._rmsd(pc,CL.DRG), fit=fit, capap=float(dd.max()))
for tgt in ["p10_GIGanchor","p9_DRGanchor"]:
    fs=sorted(glob.glob(f"outputs/rfdiff_b3/{tgt}_*.pdb"))
    R=[score(f) for f in fs]; R=[r for r in R if r]
    if not R: print(f"{tgt}: no designs yet"); continue
    coh=[r for r in R if r["coherent"]]; g=np.array([r["g"] for r in coh]); d=np.array([r["d"] for r in coh])
    exp="DRG" if "DRG" in tgt else "GIG"
    print(f"\n{tgt} (target register {exp}): n={len(R)}, coherent={len(coh)} ({100*len(coh)/len(R):.0f}%)")
    if len(coh):
        near = d if exp=="DRG" else g
        print(f"  toGIG {np.nanmedian(g):.2f} (min {np.nanmin(g):.2f}) | toDRG {np.nanmedian(d):.2f} (min {np.nanmin(d):.2f})")
        print(f"  → nearest INTENDED basin ({exp}) median {np.nanmedian(near):.2f}, best {np.nanmin(near):.2f}  (<1.45 = in basin)")
        seated=int((np.minimum(g,d)<1.45).sum()); print(f"  seated in SOME basin (<1.45): {seated}/{len(coh)}")
