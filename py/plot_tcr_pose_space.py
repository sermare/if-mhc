#!/usr/bin/env python3
"""Project the TCR POSE (relative to the MHC groove) into a shared low-dim space, across:
crystals (6AM5 GIG / 6AMU DRG), MD of the crystals (native 300/370K), design MD (L4/L5/L2 designs),
and OpenMM relaxations. Every structure is superposed on the MHC alpha1alpha2 floor (register-agnostic
frame, core_load.REF_CA), then the sequence-matched TCR Va+Vb Calpha (the register-relevant variable
domains + CDR loops, the common set across full-crystal and trimmed-design TCRs) are collected and PCA'd.
The plot answers: are the two crystal TCR poses distinguishable, and is there a continuous pose axis
between them that MD/designs populate? (== does a 'TCR pose axis' carrying the register even exist.)
Also prints the raw TCR-CA RMSD between the two crystal poses in the MHC frame (is the axis long enough).
"""
import sys, glob, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,"/home/ubuntu/if-mhc/py")
import numpy as np, matplotlib.pyplot as plt
import mdtraj as mdt
from Bio.PDB import PDBParser
import core_load as CL
plt.rcParams.update({"figure.dpi":130,"font.size":9})
RES="outputs/tamarind/results"
AA3={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
     'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V','MSE':'M'}

def split_sigs(seq, ca):
    """Return dict of (subseq, subca) for mhc/tcra/tcrb located by sequence signatures."""
    out={}
    m=seq.find("SHSMRYFF");  a=seq.find("EVEQNSGPL");  b=seq.find("IAGITQAP")
    if m<0 or a<0 or b<0: return None
    out["mhc"] =(seq[m:m+180], ca[m:m+180])          # alpha1alpha2 floor
    out["tcra"]=(seq[a:b],     ca[a:b])              # Valpha .. up to Vbeta start
    out["tcrb"]=(seq[b:b+130], ca[b:b+130])          # Vbeta variable domain
    return out

def load_struct_ca(path):
    """Ordered (seq, Nx3 CA) for a PDB, all residues in file order."""
    m=PDBParser(QUIET=True).get_structure("x",path)[0]
    seq=[]; ca=[]
    for ch in m:
        for r in ch:
            if r.id[0]==" " and "CA" in r: seq.append(AA3.get(r.resname,"x")); ca.append(r["CA"].coord)
    return "".join(seq), np.array(ca,float)

def fit_mhc(mhc_seq, mhc_ca):
    """Robust superposition of an MHC alpha1alpha2 onto core_load REF_CA; return R,t (source->ref)."""
    k=CL._off(mhc_seq, CL._REFseq)
    idx=[i for i in range(len(mhc_seq)) if 0<=i+k<len(CL._REFseq) and mhc_seq[i]==CL._REFseq[i+k]]
    if len(idx)<60: return None,None,0
    R,t,_=CL._robust(mhc_ca[idx], CL.REF_CA[[i+k for i in idx]])
    return R,t,len(idx)

def match(ref_seq, sub_seq, sub_ca_ref):
    """map sub TCR CA (already in ref frame) onto reference-residue indices by sequence offset."""
    k=CL._off(sub_seq, ref_seq)
    return {i+k: sub_ca_ref[i] for i in range(len(sub_seq)) if 0<=i+k<len(ref_seq) and sub_seq[i]==ref_seq[i+k]}

def frame_feature(seq, ca):
    """-> ({refA_idx:coord}, {refB_idx:coord}) in the MHC-superposed ref frame, or None."""
    s=split_sigs(seq,ca)
    if s is None: return None
    R,t,n=fit_mhc(*s["mhc"])
    if R is None: return None
    da=match(REF_A, s["tcra"][0], s["tcra"][1]@R+t)
    db=match(REF_B, s["tcrb"][0], s["tcrb"][1]@R+t)
    return da,db

# ---- reference TCR (6AMU) in the ref MHC frame ----
seqU,caU=load_struct_ca("inputs/focus_6am/6AMU.pdb"); sU=split_sigs(seqU,caU)
Ru,tu,_=fit_mhc(*sU["mhc"]); REF_A=sU["tcra"][0]; REF_B=sU["tcrb"][0]
refA_coords=sU["tcra"][1]@Ru+tu; refB_coords=sU["tcrb"][1]@Ru+tu

# ---- gather all sources as (label, seq, ca) frames ----
PTS=[]  # (source, seed, da, db)
def add_pdb(path, source, seed):
    try:
        seq,ca=load_struct_ca(path); f=frame_feature(seq,ca)
        if f: PTS.append((source,seed,f[0],f[1]))
    except Exception as e: print("skip",path,e)
def add_traj(job, source, seed, stride):
    d=f"{RES}/{job}"; xtc=f"{d}/traj_prod_no_water_seg1.xtc"; top=f"{d}/topology_no_water.pdb"
    try:
        t=mdt.load(xtc,top=top)
    except Exception as e: print("skip",job,e); return
    ca_sel=t.topology.select("name CA"); res=[r for r in t.topology.residues]
    # residue code per CA (align CA atoms to residues that HAVE a CA, in order)
    resc=[];
    for r in res:
        if any(a.name=="CA" for a in r.atoms): resc.append(r.code if r.code else "x")
    seq="".join(resc)
    for fr in range(0, t.n_frames, stride):
        ca=t.xyz[fr, ca_sel, :]*10.0
        f=frame_feature(seq, ca)
        if f: PTS.append((source,seed,f[0],f[1]))

# crystals
add_pdb("inputs/focus_6am/6AM5.pdb","crystal","6AM5")
add_pdb("inputs/focus_6am/6AMU.pdb","crystal","6AMU")
# MD of crystals (native)
for job in ["ifmhc_6AM5_md_300K","ifmhc_6AM5_md_370K","ifmhc_6AMU_md_300K","ifmhc_6AMU_md_370K"]:
    seed="6AM5" if "6AM5" in job else "6AMU"; add_traj(job,f"native MD {job[-4:]}",seed,8)
# design MD
for job in glob.glob("ifmhc_md_6AM*", root_dir=RES) if hasattr(glob,'glob') else []:
    pass
import os
for job in sorted(os.listdir(RES)):
    if job.startswith("ifmhc_md_6AM"):
        seed="6AM5" if "6AM5" in job else "6AMU"; add_traj(job,"design MD",seed,10)
    if job.startswith("ifmhc_relax_6AM"):
        seed="6AM5" if "6AM5" in job else "6AMU"; add_pdb(f"{RES}/{job}/relaxed.pdb","openmm relax",seed)

print(f"collected {len(PTS)} TCR-pose frames")
# ---- common reference indices present in ALL frames ----
comA=set(range(len(REF_A))); comB=set(range(len(REF_B)))
for _,_,da,db in PTS: comA&=set(da); comB&=set(db)
comA=sorted(comA); comB=sorted(comB)
print(f"common matched TCR CA: Valpha {len(comA)} + Vbeta {len(comB)} = {len(comA)+len(comB)} residues")
def vec(da,db): return np.concatenate([np.array([da[i] for i in comA]).ravel(), np.array([db[i] for i in comB]).ravel()])
X=np.array([vec(da,db) for _,_,da,db in PTS])
# raw crystal-crystal TCR pose separation (Angstrom, MHC frame)
ci={s:k for k,(src,s,_,_) in enumerate(PTS) if src=="crystal"}
allca=lambda k: X[k].reshape(-1,3)
sep=np.sqrt(((allca(ci["6AM5"])-allca(ci["6AMU"]))**2).sum(1).mean())
print(f"RAW crystal 6AM5<->6AMU TCR-Valpha/Vbeta CA RMSD (MHC-superposed) = {sep:.2f} A")

# ---- separability stats in raw MHC-frame TCR-CA space (Angstrom) ----
def rmsd_rows(A,b): return np.sqrt((((A-b).reshape(len(A),-1,3))**2).sum(2).mean(1))  # per-frame CA-RMSD to vector b
src_arr=np.array([p[0] for p in PTS]); seed_arr=np.array([p[1] for p in PTS])
def grp(pred): return X[np.array([pred(s,e) for s,e in zip(src_arr,seed_arr)])]
natA=grp(lambda s,e: s.startswith("native MD") and e=="6AM5")
natU=grp(lambda s,e: s.startswith("native MD") and e=="6AMU")
def spread(G): return float(rmsd_rows(G, G.mean(0)).mean()) if len(G) else float("nan")
cen=lambda G: G.mean(0)
d_nat=float(np.sqrt(((cen(natA)-cen(natU)).reshape(-1,3)**2).sum(1).mean()))
print(f"\n-- TCR pose separability (raw MHC-frame CA-RMSD, A) --")
print(f"crystal 6AM5<->6AMU              = {sep:.2f}")
print(f"native-MD 6AM5<->6AMU centroids  = {d_nat:.2f}   (within-6AM5 spread {spread(natA):.2f}, within-6AMU {spread(natU):.2f})")
print(f"=> crystal & native-MD register separation is {'SWAMPED by' if d_nat<spread(natA) else 'comparable to'} thermal spread")
desg=grp(lambda s,e: s=="design MD");
print(f"design-MD centroid distance from native-MD centroid = "
      f"{float(np.sqrt(((cen(desg)-cen(np.vstack([natA,natU]))).reshape(-1,3)**2).sum(1).mean())):.2f}  (off-manifold check)")

# ---- CA-RMSD of every frame to each crystal TCR pose (MHC-superposed) ----
rmsd_to_5=rmsd_rows(X, X[ci["6AM5"]]); rmsd_to_U=rmsd_rows(X, X[ci["6AMU"]])
# ---- PCA #1: all sources ----
mu=X.mean(0); U,S,Vt=np.linalg.svd(X-mu, full_matrices=False); PC=(X-mu)@Vt[:2].T; ev=(S**2)/(S**2).sum()
# ---- PCA #2: physiological only (crystal+native MD+relax) — design MD is off-manifold & hijacks PC1 ----
mphys=np.array([s=="crystal" or s.startswith("native MD") or s=="openmm relax" for s in src_arr])
Xp=X[mphys]; mup=Xp.mean(0); Up,Sp,Vtp=np.linalg.svd(Xp-mup, full_matrices=False); evp=(Sp**2)/(Sp**2).sum()
PCp=np.full((len(X),2),np.nan); PCp[mphys]=(Xp-mup)@Vtp[:2].T
print(f"PCA all: PC1 {ev[0]*100:.1f}% PC2 {ev[1]*100:.1f}% | PCA physio: PC1 {evp[0]*100:.1f}% PC2 {evp[1]*100:.1f}%")

# ---- 10-class scheme: 8 validated categorical hues for MD/relax series + 2 neutral crystal stars;
#      SHAPE encodes seed (o=6AM5, ^=6AMU) so 300K vs 370K are now distinct COLORS, not shades. ----
t10=plt.cm.tab10.colors   # 10 distinct hues; shape (o/^) carries seed, so color is per source×seed
CMAP={("crystal","6AM5"):"#0b0b0b",        ("crystal","6AMU"):"#6f6f6f",           # black / gray (references)
      ("native MD 300K","6AM5"):t10[0],    ("native MD 300K","6AMU"):t10[1],       # blue / orange
      ("native MD 370K","6AM5"):t10[2],    ("native MD 370K","6AMU"):t10[3],       # green / red
      ("openmm relax","6AM5"):t10[4],      ("openmm relax","6AMU"):t10[6],         # purple / pink
      ("design MD","6AM5"):t10[8],         ("design MD","6AMU"):t10[9]}            # olive / cyan
ORDER=[("crystal","6AM5"),("crystal","6AMU"),
       ("native MD 300K","6AM5"),("native MD 300K","6AMU"),("native MD 370K","6AM5"),("native MD 370K","6AMU"),
       ("openmm relax","6AM5"),("openmm relax","6AMU"),("design MD","6AM5"),("design MD","6AMU")]
def style(src,seed):
    mk="*" if src=="crystal" else ("o" if seed=="6AM5" else "^")
    sz=300 if src=="crystal" else (60 if src=="openmm relax" else (30 if src.startswith("native") else 12))
    return mk,sz
def draw(ax,xv,yv):
    for src,seed in ORDER:
        ks=[k for k,p in enumerate(PTS) if p[0]==src and p[1]==seed and np.isfinite(xv[k]) and np.isfinite(yv[k])]
        if not ks: continue
        mk,sz=style(src,seed)
        ax.scatter(xv[ks],yv[ks],marker=mk,s=sz,c=CMAP[(src,seed)],
                   alpha=(1.0 if src=="crystal" else 0.6),
                   edgecolor=("k" if src in("crystal","openmm relax") else "none"),
                   linewidth=.5,zorder=(7 if src=="crystal" else 3),
                   label=f"{src}·{'GIG' if seed=='6AM5' else 'DRG'}")

fig,axs=plt.subplots(1,3,figsize=(19,6.4))
# Panel 1 — CA RMSD to the two crystal poses
draw(axs[0], rmsd_to_5, rmsd_to_U)
lim=max(rmsd_to_5.max(),rmsd_to_U.max())*1.02
axs[0].plot([0,lim],[0,lim],"--",c="#898781",lw=.8)
axs[0].set_xlabel("TCR Cα-RMSD to 6AM5/GIG crystal pose (Å)")
axs[0].set_ylabel("TCR Cα-RMSD to 6AMU/DRG crystal pose (Å)")
axs[0].set_title(f"TCR pose — CA-RMSD to each crystal\n(crystals differ by only {sep:.2f} Å)")
# Panel 2 — PCA all sources
draw(axs[1], PC[:,0], PC[:,1])
axs[1].set_xlabel(f"PC1 ({ev[0]*100:.0f}%)  — native↔design axis"); axs[1].set_ylabel(f"PC2 ({ev[1]*100:.0f}%)")
axs[1].set_title("PCA — all sources\n(design MD sits off the physiological manifold)")
# Panel 3 — PCA physiological only
draw(axs[2], PCp[:,0], PCp[:,1])
axs[2].set_xlabel(f"PC1 ({evp[0]*100:.0f}%)"); axs[2].set_ylabel(f"PC2 ({evp[1]*100:.0f}%)")
axs[2].set_title("PCA — physiological only (crystal+native+relax)\ndoes GIG vs DRG separate without design MD?")
h,l=axs[0].get_legend_handles_labels()
fig.legend(h,l,loc="lower center",ncol=10,fontsize=7.5,frameon=False,bbox_to_anchor=(0.5,-0.02))
fig.suptitle("TCR pose relative to the MHC groove — CA-RMSD + two PCAs  (color = source, ○=GIG ▲=DRG)",
             fontsize=12,y=1.0)
plt.tight_layout(rect=[0,0.05,1,1]); plt.savefig("/home/ubuntu/if-mhc/native_tcr_pose_space.png",dpi=150,bbox_inches="tight")
print("saved native_tcr_pose_space.png")

# ================== CDR3 LOOP CONFORMATION (pose-free: framework-superposed) ==================
# The pose plot above superposes on MHC -> dominated by rigid-body position, ~blind to loop-internal
# rearrangement. Here each V-domain is superposed on ITS OWN framework (V-domain CA minus CDR3), then
# only the CDR3 loop CA is measured -> isolates CDR3 conformation, the remaining register candidate.
import re
def find_cdr3(seq):
    ms=list(re.finditer("[FW]G.G", seq))                 # J-region motif ends CDR3
    if not ms: return None
    end=ms[-1].start(); cys=seq.rfind("C",0,end)         # conserved 2nd Cys begins CDR3
    return list(range(cys+1,end)) if 0<=cys<end else None
c3a=find_cdr3(REF_A); c3b=find_cdr3(REF_B)
print(f"\nCDR3a = '{''.join(REF_A[i] for i in c3a)}' ({len(c3a)} res) | CDR3b = '{''.join(REF_B[i] for i in c3b)}' ({len(c3b)} res)")
fwA=[i for i in range(len(REF_A)) if i not in set(c3a)]; fwB=[i for i in range(len(REF_B)) if i not in set(c3b)]
def loop_fw(d, fw, loop, refc):
    have=[i for i in fw if i in d]
    if len(have)<25: return None
    R,t=CL._RT(np.array([d[i] for i in have]), refc[have])   # align this frame's framework to ref framework
    return {i:(d[i]@R+t) for i in loop if i in d}            # CDR3 CA in that framework frame
feats=[(loop_fw(da,fwA,c3a,refA_coords), loop_fw(db,fwB,c3b,refB_coords)) for _,_,da,db in PTS]
comA=set(c3a); comB=set(c3b)
for la,lb in feats:
    comA &= set(la) if la else set(); comB &= set(lb) if lb else set()
comA=sorted(comA); comB=sorted(comB)
print(f"CDR3 loop CA in common: alpha {len(comA)} + beta {len(comB)}")
def cvec(la,lb):
    if la is None or lb is None or not all(i in la for i in comA) or not all(i in lb for i in comB): return None
    return np.concatenate([np.array([la[i] for i in comA]).ravel(), np.array([lb[i] for i in comB]).ravel()])
Xc=np.array([cvec(*f) if cvec(*f) is not None else np.full((len(comA)+len(comB))*3,np.nan) for f in feats])
ok=np.isfinite(Xc).all(1)
# CA-RMSD to each crystal CDR3 conformation + separability
r5=rmsd_rows(Xc, Xc[ci["6AM5"]]); rU=rmsd_rows(Xc, Xc[ci["6AMU"]])
csep=float(np.sqrt(((Xc[ci["6AM5"]]-Xc[ci["6AMU"]]).reshape(-1,3)**2).sum(1).mean()))
def grpc(pred):
    m=np.array([pred(s,e) for s,e in zip(src_arr,seed_arr)])&ok; return Xc[m]
nA=grpc(lambda s,e: s.startswith("native MD") and e=="6AM5"); nU=grpc(lambda s,e: s.startswith("native MD") and e=="6AMU")
spr=lambda G: float(rmsd_rows(G,G.mean(0)).mean()) if len(G) else float("nan")
dc=float(np.sqrt(((nA.mean(0)-nU.mean(0)).reshape(-1,3)**2).sum(1).mean()))
print(f"-- CDR3 loop conformation (framework-superposed CA-RMSD, A) --")
print(f"crystal GIG<->DRG CDR3            = {csep:.2f}")
print(f"native-MD GIG<->DRG CDR3 centroids= {dc:.2f}  (within-GIG {spr(nA):.2f}, within-DRG {spr(nU):.2f})")
print(f"=> CDR3 register separation is {'RESOLVED above' if dc>1.3*max(spr(nA),spr(nU)) else 'STILL SWAMPED by'} thermal spread")
# PCA of CDR3 conformation — (a) all sources, (b) NO design MD (physiological loop shape only)
Xf=Xc[ok]; muc=Xf.mean(0); Uc,Sc,Vtc=np.linalg.svd(Xf-muc,full_matrices=False); evc=(Sc**2)/(Sc**2).sum()
PCc=np.full((len(Xc),2),np.nan); PCc[ok]=(Xf-muc)@Vtc[:2].T
nd=ok & (src_arr!="design MD")                                  # exclude designs
Xn=Xc[nd]; mun=Xn.mean(0); Un,Sn,Vtn=np.linalg.svd(Xn-mun,full_matrices=False); evn=(Sn**2)/(Sn**2).sum()
PCn=np.full((len(Xc),2),np.nan); PCn[nd]=(Xn-mun)@Vtn[:2].T
no=ok & (src_arr!="design MD") & (src_arr!="openmm relax")      # native MD + crystals only
Xo=Xc[no]; muo=Xo.mean(0); Uo,So,Vto=np.linalg.svd(Xo-muo,full_matrices=False); evo=(So**2)/(So**2).sum()
PCo=np.full((len(Xc),2),np.nan); PCo[no]=(Xo-muo)@Vto[:2].T
print(f"CDR3 PCA all: PC1 {evc[0]*100:.1f}% | no-design: PC1 {evn[0]*100:.1f}% | native-only: PC1 {evo[0]*100:.1f}% PC2 {evo[1]*100:.1f}%")

fig2,ax2=plt.subplots(1,4,figsize=(25,6.4))
draw(ax2[0], r5, rU); lim2=max(np.nanmax(r5),np.nanmax(rU))*1.02
ax2[0].plot([0,lim2],[0,lim2],"--",c="#898781",lw=.8)
ax2[0].set_xlabel("CDR3 Cα-RMSD to 6AM5/GIG (Å)"); ax2[0].set_ylabel("CDR3 Cα-RMSD to 6AMU/DRG (Å)")
ax2[0].set_title(f"CDR3 loop conformation — CA-RMSD to each crystal\n(crystal GIG↔DRG CDR3 = {csep:.2f} Å; framework-superposed)")
draw(ax2[1], PCc[:,0], PCc[:,1])
ax2[1].set_xlabel(f"PC1 ({evc[0]*100:.0f}%)"); ax2[1].set_ylabel(f"PC2 ({evc[1]*100:.0f}%)")
ax2[1].set_title("CDR3 conformation — PCA (all sources)")
draw(ax2[2], PCn[:,0], PCn[:,1])
ax2[2].set_xlabel(f"PC1 ({evn[0]*100:.0f}%)"); ax2[2].set_ylabel(f"PC2 ({evn[1]*100:.0f}%)")
ax2[2].set_title("CDR3 conformation — PCA, NO design MD\ndoes GIG vs DRG separate in loop shape?")
draw(ax2[3], PCo[:,0], PCo[:,1])
ax2[3].set_xlabel(f"PC1 ({evo[0]*100:.0f}%)"); ax2[3].set_ylabel(f"PC2 ({evo[1]*100:.0f}%)")
ax2[3].set_title("CDR3 conformation — PCA, NO design & NO openmm\n(native MD + crystals only)")
h2,l2=ax2[0].get_legend_handles_labels()
fig2.legend(h2,l2,loc="lower center",ncol=10,fontsize=7.5,frameon=False,bbox_to_anchor=(0.5,-0.02))
fig2.suptitle(f"TCR CDR3α+CDR3β loop conformation, framework-superposed (pose removed)  ○=GIG ▲=DRG  [tab10]",fontsize=12,y=1.0)
plt.tight_layout(rect=[0,0.06,1,1]); plt.savefig("/home/ubuntu/if-mhc/native_tcr_cdr3_conformation.png",dpi=150,bbox_inches="tight")
print("saved native_tcr_cdr3_conformation.png")
