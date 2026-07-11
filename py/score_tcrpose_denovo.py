#!/usr/bin/env python3
"""Score the register of de-novo RFD peptides generated within each crystal TCR pose.
Output layout (RFdiffusion binder mode): chain A = 10-mer de-novo peptide; chain B = merged receptor
(MHC a1a2 first 180 res, then b2m, TCR). Locate the F-pocket by the MHC sequence anchor 'SHSMRYFF'
(HLA author 2), so it is robust to RFD renumbering and works for both poses. Register readout is the
frame-independent F-pocket swap = d(p10,Fc) - d(p9,Fc): <0 unshifted/GIG-like (p10-in), >0 shifted/
DRG-like (p9-in). 'seated' = peptide actually in the groove (both p9,p10 within 11 A of Fc)."""
import sys, glob, warnings; warnings.filterwarnings("ignore")
import numpy as np
from Bio.PDB import PDBParser
AA3={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
     'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V','MSE':'M'}
FP_AUTHOR=[77,80,84,116,123,143]   # HLA F-pocket; author 2 = first residue of 'SHSMRYFF'
p=PDBParser(QUIET=True)
def score(fn):
    m=p.get_structure('x',fn)[0]
    pep=[r for r in m['A'] if r.id[0]==' ' and 'CA' in r]
    rec=[r for r in m['B'] if r.id[0]==' ' and 'CA' in r]
    seq="".join(AA3.get(r.resname,'x') for r in rec)
    i0=seq.find("SHSMRYFF")
    if i0<0 or len(pep)<10: return None
    # HLA author 2 -> rec index i0 ; author n -> i0 + (n-2)
    fp=[rec[i0+(n-2)] for n in FP_AUTHOR if 0<=i0+(n-2)<len(rec)]
    fc=np.mean([r['CA'].coord for r in fp],axis=0)
    p9,p10=pep[-2]['CA'].coord, pep[-1]['CA'].coord
    d9,d10=np.linalg.norm(p9-fc),np.linalg.norm(p10-fc)
    swap=float(d10-d9)
    seated = (d9<11 and d10<11)
    return dict(swap=round(swap,2), d9=round(float(d9),1), d10=round(float(d10),1), seated=seated, nfp=len(fp))

for pose in ["GIGpose","DRGpose"]:
    rows=[score(f) for f in sorted(glob.glob(f"outputs/tcrpose_denovo/{pose}_*.pdb"))]
    rows=[r for r in rows if r]
    if not rows: print(f"{pose}: no designs yet"); continue
    seat=[r for r in rows if r["seated"]]
    sw=np.array([r["swap"] for r in seat]) if seat else np.array([])
    print(f"\n=== {pose}  (n={len(rows)}, seated_in_groove={len(seat)}) ===")
    for r in rows: print(f"   swap={r['swap']:+.2f}  d(p9)={r['d9']:.1f} d(p10)={r['d10']:.1f}  {'SEATED' if r['seated'] else 'floating'}")
    if len(sw):
        unshift=int((sw<-1).sum()); shift=int((sw>1).sum()); amb=len(sw)-unshift-shift
        print(f"   SEATED register: unshifted(GIG-like,swap<-1)={unshift}  shifted(DRG-like,swap>+1)={shift}  ambiguous={amb}")
        print(f"   seated mean swap = {sw.mean():+.2f} +/- {sw.std():.2f}")
print("\n(reference: GIG native swap -3.25, DRG native +3.54)")
