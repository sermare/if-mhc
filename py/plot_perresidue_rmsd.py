#!/usr/bin/env python3
"""Per-residue 6AM5(GIG) vs 6AMU(DRG) deviation, superposed on the invariant beta-sheet floor
(q30_analysis.FLOOR; floor RMSD ~0.21 A). THREE panels:
  (1) MHC BACKBONE-only (N,Cα,C,O) — strips rotamer/refinement noise; shows if any real backbone
      rearrangement localizes to the F-pocket.
  (2) MHC ALL-ATOM (heavy) — includes side chains (the anchor swap is a packing event).
  (3) Peptide per-position (Cα + backbone; side chains differ between epitopes).
Each notable MHC spike is LABELED with its residue name and its distance to the peptide C-terminal
anchor (p9/p10) so we can tell F-pocket-adjacent rearrangement from unrelated surface rotamer noise."""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,"py")
import numpy as np, matplotlib.pyplot as plt
from Bio.PDB import PDBParser
import core_load as CL
from q30_analysis import FLOOR, FPOCKET
plt.rcParams.update({"figure.dpi":130,"font.size":9})
p=PDBParser(QUIET=True)
m5=p.get_structure("5","inputs/focus_6am/6AM5.pdb")[0]; mU=p.get_structure("U","inputs/focus_6am/6AMU.pdb")[0]
rm=lambda ch:{r.id[1]:r for r in ch if r.id[0]==" "}
A5,AU=rm(m5["A"]),rm(mU["A"]); C5,CU=rm(m5["C"]),rm(mU["C"])
cf=[a for a in FLOOR if a in A5 and a in AU and "CA" in A5[a] and "CA" in AU[a]]
P=np.array([A5[a]["CA"].coord for a in cf]); Q=np.array([AU[a]["CA"].coord for a in cf])
R,t=CL._RT(P,Q); xf=lambda v:v@R+t
floor_rmsd=np.sqrt(((P@R+t-Q)**2).sum(1).mean()); print(f"floor RMSD {floor_rmsd:.2f} A")

# peptide anchor atoms (6AMU frame) = p9+p10 heavy atoms; and all peptide heavy atoms for contact
pep_heavy=np.array([a.coord for q in CU for a in CU[q] if a.element!="H"])
anchor=np.array([a.coord for q in (9,10) if q in CU for a in CU[q] if a.element!="H"])
def dev(r5,rU,names):
    return np.sqrt(np.mean([np.sum((xf(r5[n].coord)-rU[n].coord)**2) for n in names])) if names else np.nan
def mindist(rU, target):
    hv=np.array([a.coord for a in rU if a.element!="H"]);
    return float(np.min(np.linalg.norm(hv[:,None,:]-target[None,:,:],axis=2)))

auth=[a for a in sorted(set(A5)&set(AU)) if a<=181]
rows=[]
for a in auth:
    r5,rU=A5[a],AU[a]
    heavy=[x.name for x in r5 if x.name in rU and x.element!="H"]
    bb=[n for n in ("N","CA","C","O") if n in heavy]; sc=[n for n in heavy if n not in ("N","CA","C","O")]
    rows.append(dict(a=a,name=rU.resname, ca=dev(r5,rU,["CA"]) if ("CA" in r5 and "CA" in rU) else np.nan,
                     bb=dev(r5,rU,bb), allv=dev(r5,rU,heavy), sc=dev(r5,rU,sc) if sc else np.nan,
                     d_pep=mindist(rU,pep_heavy), d_anc=mindist(rU,anchor)))
import pandas as pd; D=pd.DataFrame(rows)
FP_SET=sorted(set(FPOCKET)|{77,80,81,84,116,123,143,146,147}); ARM=list(range(145,153))
print("\nTop-12 all-atom spikes — labeled (F-pocket-adjacent if near p9/p10 anchor):")
print(f"{'res':>10} {'bb':>5} {'allatom':>7} {'sidech':>6} {'d→pep':>6} {'d→p9p10':>7}  location")
for _,r in D.sort_values("allv",ascending=False).head(12).iterrows():
    loc = "F-POCKET" if r.a in FP_SET else ("α2-arm" if r.a in ARM else ("pocket-adjacent" if r.d_anc<8 else "distal (not pocket)"))
    print(f"{int(r.a):>4}{r['name']:>6} {r.bb:5.2f} {r.allv:7.2f} {r.sc:6.2f} {r.d_pep:6.1f} {r.d_anc:7.1f}  {loc}")
# backbone: does it localize to pocket?
bb_pocket=D[D.a.isin(FP_SET+ARM)].bb.mean(); bb_else=D[~D.a.isin(FP_SET+ARM)].bb.mean()
print(f"\nBACKBONE (N,Cα,C,O): F-pocket/arm mean {bb_pocket:.2f} vs elsewhere {bb_else:.2f} A (ratio {bb_pocket/bb_else:.1f}x)")

def bands(ax):
    for a0 in ARM: ax.axvspan(a0-.5,a0+.5,color="#eda100",alpha=.12,zorder=0)
    for a0 in FP_SET: ax.axvspan(a0-.5,a0+.5,color="#e34948",alpha=.16,zorder=0)
def annot(ax,col,thr):
    for _,r in D.iterrows():
        if r[col]>thr:
            adj = r.a in FP_SET or r.a in ARM or r.d_anc<8
            ax.annotate(f"{int(r.a)}{r['name'][0]}",(r.a,r[col]),fontsize=6.3,ha="center",va="bottom",
                        color=("#e34948" if adj else "#52514e"),fontweight=("bold" if adj else "normal"))

fig,ax=plt.subplots(3,1,figsize=(15,10),gridspec_kw={"height_ratios":[3,3,1.4]})
# Panel 1 — backbone only
bands(ax[0]); ax[0].plot(D.a,D.bb,color="#2a78d6",lw=1.4); ax[0].axhline(bb_else,color="#898781",ls=":",lw=.8,label=f"non-pocket baseline {bb_else:.2f} Å")
annot(ax[0],"bb",0.8)
ax[0].set_ylabel("backbone dev (Å)"); ax[0].set_xlim(min(auth),181); ax[0].legend(fontsize=8)
ax[0].set_title(f"MHC BACKBONE (N,Cα,C,O) per residue — floor-superposed (floor {floor_rmsd:.2f} Å). "
                f"Flat everywhere (F-pocket/arm {bb_pocket:.2f} ≈ baseline {bb_else:.2f}) ⇒ NO backbone rearrangement, pocket or elsewhere\n"
                f"red band=F-pocket · gold=α2 arm · red bold labels=pocket-adjacent, gray=distal")
# Panel 2 — all atom
bands(ax[1]); ax[1].plot(D.a,D.allv,color="#e34948",lw=1.4); ax[1].plot(D.a,D.ca,color="#2a78d6",lw=.9,alpha=.7,label="Cα")
ax[1].axhline(D[~D.a.isin(FP_SET+ARM)].allv.mean(),color="#898781",ls=":",lw=.8)
annot(ax[1],"allv",1.6)
ax[1].set_ylabel("all-atom dev (Å)"); ax[1].set_xlim(min(auth),181); ax[1].legend(fontsize=8)
ax[1].set_title("MHC ALL-ATOM (heavy) — side-chain-inclusive. Spikes labeled; distal (gray) spikes = surface rotamer/refinement (their backbone is flat above)")
ax[1].set_xlabel("MHC residue (HLA-A*02 author numbering, α1α2 groove)")
# Panel 3 — peptide
pos=sorted(set(C5)&set(CU)); BB=["N","CA","C","O"]; w=0.38
pca=[dev(C5[q],CU[q],["CA"]) for q in pos]; pbb=[dev(C5[q],CU[q],[n for n in BB if n in C5[q] and n in CU[q]]) for q in pos]
ax[2].bar(np.array(pos)-w/2,pca,w,color="#2a78d6",label="Cα"); ax[2].bar(np.array(pos)+w/2,pbb,w,color="#e34948",label="backbone")
ax[2].axvspan(8.5,10.5,color="#e34948",alpha=.12); ax[2].set_xticks(pos); ax[2].legend(fontsize=8)
ax[2].set_xlabel("peptide position p1→p10"); ax[2].set_ylabel("dev (Å)")
ax[2].set_title("Peptide per-position (side chains differ → backbone only) — flat p1–p4, all difference p5–p10, anchor shaded")
plt.tight_layout(); plt.savefig("/home/ubuntu/if-mhc/perresidue_gig_drg.png",dpi=150)
print("saved perresidue_gig_drg.png")
