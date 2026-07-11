#!/usr/bin/env python3
"""Are the two F-pocket rotamers (Tyr116, Lys146) register-COUPLED, or free refinement noise?
Across the native MD ensembles (GIG=6AM5, DRG=6AMU; 300+370K) plot side-chain chi1 of Tyr116 & Lys146
vs the F-pocket occupancy coordinate  swap = d(p10,Fc)-d(p9,Fc)  (GIG/unshift ~ -3.3, DRG/shift ~ +3.5).
Negative control = a DISTAL refinement-noise residue (Arg111, 20.7 A from the anchor). If 116/146 chi
separate by register / track swap while 111 scatters identically in both -> the two rotamers are
mechanistically anchor-coupled (two-region statement is MEASURED). If 116/146 also scatter freely ->
they are refinement noise too and the register is PURELY the peptide C-terminus (even cleaner)."""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,"py")
import numpy as np, matplotlib.pyplot as plt, mdtraj as mdt
plt.rcParams.update({"figure.dpi":130,"font.size":9})
AA3={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V','MSE':'M'}
FPA=[77,80,84,116,123,143]
RES="outputs/tamarind/results"

def analyze(job):
    d=f"{RES}/{job}"; t=mdt.load(f"{d}/traj_prod_no_water_seg1.xtc",top=f"{d}/topology_no_water.pdb")
    res=list(t.topology.residues); seq="".join(AA3.get(r.name,"x") for r in res)
    mhc=seq.find("SHSMRYFF"); b2m=seq.find("MIQRTP",mhc); ta=seq.find("EVEQNSGPL",b2m)
    leadingG = mhc>0 and seq[mhc-1]=="G"; mhc0=mhc-1 if leadingG else mhc; a0=1 if leadingG else 2
    ka=ta-1 if ta>0 and seq[ta-1]=="K" else ta                 # TCRa true start (6AM5 leading K)
    pep0=ka-10                                                  # peptide = 10 before TCRa
    def R(auth): return res[mhc0+(auth-a0)]                     # MHC residue by author number
    def aidx(r): return {a.name:a.index for a in r.atoms}
    fp_ca=[aidx(R(a))["CA"] for a in FPA]
    p9=[a.index for a in res[pep0+8].atoms if a.name=="CA"][0]; p10=[a.index for a in res[pep0+9].atoms if a.name=="CA"][0]
    xyz=t.xyz*10.0
    fc=xyz[:,fp_ca,:].mean(1); swap=np.linalg.norm(xyz[:,p10,:]-fc,axis=1)-np.linalg.norm(xyz[:,p9,:]-fc,axis=1)
    def chi1(auth):
        A=aidx(R(auth)); idx=np.array([[A["N"],A["CA"],A["CB"],A["CG"]]])
        return np.degrees(mdt.compute_dihedrals(t,idx)[:,0])
    return dict(swap=swap, y116=chi1(116), y146=chi1(146), y111=chi1(111),
                name116=R(116).name, name146=R(146).name, name111=R(111).name)

GIG=[analyze("ifmhc_6AM5_md_300K"),analyze("ifmhc_6AM5_md_370K")]
DRG=[analyze("ifmhc_6AMU_md_300K"),analyze("ifmhc_6AMU_md_370K")]
def cat(L,k): return np.concatenate([x[k] for x in L])
print(f"names: 116={GIG[0]['name116']} 146={GIG[0]['name146']} 111={GIG[0]['name111']} (control)")
print(f"swap: GIG {cat(GIG,'swap').mean():+.2f}±{cat(GIG,'swap').std():.2f} | DRG {cat(DRG,'swap').mean():+.2f}±{cat(DRG,'swap').std():.2f}")

def circ_summary(nm,key):
    g=cat(GIG,key); d=cat(DRG,key)
    # fraction in each rotamer well (g+ ~+60, trans ~180, g- ~-60)
    def wells(a): return [ (np.abs(((a-c+180)%360)-180)<60).mean()*100 for c in (60,180,-60)]
    gw=wells(g); dw=wells(d)
    print(f"  {nm}: GIG χ1 wells g+/t/g- = {gw[0]:.0f}/{gw[1]:.0f}/{gw[2]:.0f}%  |  DRG = {dw[0]:.0f}/{dw[1]:.0f}/{dw[2]:.0f}%  "
          f"=> {'REGISTER-COUPLED (distributions differ)' if max(abs(gw[i]-dw[i]) for i in range(3))>25 else 'independent (same in both)'}")
print("χ1 rotamer occupancy by register:")
for nm,k in [("Tyr116",'y116'),("Lys146",'y146'),("Arg111 CONTROL",'y111')]: circ_summary(nm,k)

fig,ax=plt.subplots(1,3,figsize=(16,5))
for j,(nm,k) in enumerate([("Tyr116","y116"),("Lys146","y146"),("Arg111 (distal control)","y111")]):
    ax[j].scatter(cat(GIG,'swap'),cat(GIG,k),s=6,c="#2a78d6",alpha=.35,label="GIG (6AM5)")
    ax[j].scatter(cat(DRG,'swap'),cat(DRG,k),s=6,c="#e34948",alpha=.35,label="DRG (6AMU)")
    ax[j].axvline(0,color="#898781",ls=":",lw=.7)
    ax[j].set_xlabel("F-pocket swap  d(p10,Fc)−d(p9,Fc)  (Å)   ← GIG   DRG →")
    ax[j].set_ylabel(f"{nm} χ1 (°)"); ax[j].set_ylim(-180,180); ax[j].set_yticks([-180,-120,-60,0,60,120,180])
    ax[j].set_title(nm); ax[j].legend(fontsize=8,loc="upper right",markerscale=2)
fig.suptitle("F-pocket rotamer coupling: side-chain χ1 vs anchor occupancy across GIG & DRG MD  (Tyr116/Lys146 vs distal control Arg111)",fontsize=12)
plt.tight_layout(rect=[0,0,1,0.96]); plt.savefig("/home/ubuntu/if-mhc/rotamer_coupling.png",dpi=150)
print("saved rotamer_coupling.png")
