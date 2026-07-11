#!/usr/bin/env python3
"""One grid, a ROW per structural component (peptide, MHC alpha1alpha2, beta2m, TCR Va/Vb, CDR3 loops),
COLUMNS = the same four views used for the TCR: CA-RMSD to the two crystals, PCA all sources, PCA no
design MD, PCA no design & no openmm. Everything superposed on the MHC floor (register-agnostic frame;
core_load REF_CA) EXCEPT CDR3 which is framework-superposed (pose removed). Sources: crystals, native MD
300/370K, design MD, openmm relax. Peptide is scored positionally (p1..p10) so its RMSD-to-crystal map
IS the register (toGIG/toDRG); the other rows should stay register-blind. tab10 palette, o=GIG ^=DRG."""
import sys, os, glob, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,"/home/ubuntu/if-mhc/py")
import numpy as np, matplotlib.pyplot as plt, mdtraj as mdt, re
from Bio.PDB import PDBParser
import core_load as CL
plt.rcParams.update({"figure.dpi":120,"font.size":8})
RES="outputs/tamarind/results"
AA3={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V','MSE':'M'}

def struct_ca(path):
    m=PDBParser(QUIET=True).get_structure("x",path)[0]; s=[];c=[]
    for ch in m:
        for r in ch:
            if r.id[0]==" " and "CA" in r: s.append(AA3.get(r.resname,"x")); c.append(r["CA"].coord)
    return "".join(s), np.array(c,float)

def segments(seq, ca):
    mhc=seq.find("SHSMRYFF")
    if mhc<0: return None
    b2m=seq.find("MIQRTP",mhc); ta=seq.find("EVEQNSGPL",max(b2m,0)); tb=seq.find("IAGITQAP",max(ta,0))
    if min(b2m,ta,tb)<0: return None
    mhc0=mhc-1 if mhc>0 and seq[mhc-1]=="G" else mhc
    ka=ta-1 if ta>0 and seq[ta-1]=="K" else ta                       # TCRa true start (6AM5 has leading Lys)
    if mhc0>=10: pep=slice(0,mhc0); b2s=slice(b2m,ka)                 # design: peptide first
    else:        pep=slice(ka-10,ka); b2s=slice(b2m,ka-10)           # crystal/native: pep between b2m & TCRa
    sl=dict(pep=pep, mhc=slice(mhc0,b2m), b2m=b2s, tcra=slice(ka,tb), tcrb=slice(tb,len(seq)))
    return {k:(seq[s],ca[s]) for k,s in sl.items()}

def fit_mhc(mseq,mca):
    k=CL._off(mseq,CL._REFseq); idx=[i for i in range(len(mseq)) if 0<=i+k<len(CL._REFseq) and mseq[i]==CL._REFseq[i+k]]
    if len(idx)<60: return None,None
    R,t,_=CL._robust(mca[idx],CL.REF_CA[[i+k for i in idx]]); return R,t
def matchd(refseq,sseq,sca):
    k=CL._off(sseq,refseq); return {i+k:sca[i] for i in range(len(sseq)) if 0<=i+k<len(refseq) and sseq[i]==refseq[i+k]}

# reference component sequences (6AMU), superposed to the MHC floor
sqU,caU=struct_ca("inputs/focus_6am/6AMU.pdb"); segU=segments(sqU,caU)
Ru,tu=fit_mhc(*segU["mhc"]); REF={k:(segU[k][0], segU[k][1]@Ru+tu) for k in segU}
RB2M,RA,RB=REF["b2m"][0],REF["tcra"][0],REF["tcrb"][0]
def find_cdr3(seq):
    ms=list(re.finditer("[FW]G.G",seq));
    if not ms: return []
    end=ms[-1].start(); cys=seq.rfind("C",0,end); return list(range(cys+1,end)) if 0<=cys<end else []
C3A,C3B=find_cdr3(RA),find_cdr3(RB)

PTS=[]   # (src, seed, comp-dict)  comp = pep:(10,3) ; mhc/b2m/tcra/tcrb: {refidx:xyz}  — all in MHC frame
def add_frame(seq,ca,src,seed):
    s=segments(seq,ca)
    if s is None or s["pep"][1].shape[0]!=10: return
    R,t=fit_mhc(*s["mhc"])
    if R is None: return
    comp=dict(pep=s["pep"][1]@R+t,
              mhc=matchd(CL._REFseq,s["mhc"][0],s["mhc"][1]@R+t),
              b2m=matchd(RB2M,s["b2m"][0],s["b2m"][1]@R+t),
              tcra=matchd(RA,s["tcra"][0],s["tcra"][1]@R+t),
              tcrb=matchd(RB,s["tcrb"][0],s["tcrb"][1]@R+t))
    PTS.append((src,seed,comp))
def add_pdb(p,src,seed):
    try: seq,ca=struct_ca(p); add_frame(seq,ca,src,seed)
    except Exception as e: pass
def add_traj(job,src,seed,stride):
    d=f"{RES}/{job}"
    try: t=mdt.load(f"{d}/traj_prod_no_water_seg1.xtc",top=f"{d}/topology_no_water.pdb")
    except Exception: return
    sel=t.topology.select("name CA"); seq="".join(r.code if r.code else "x" for r in t.topology.residues if any(a.name=="CA" for a in r.atoms))
    for fr in range(0,t.n_frames,stride): add_frame(seq, t.xyz[fr,sel,:]*10.0, src, seed)

add_pdb("inputs/focus_6am/6AM5.pdb","crystal","6AM5"); add_pdb("inputs/focus_6am/6AMU.pdb","crystal","6AMU")
for j in ["ifmhc_6AM5_md_300K","ifmhc_6AM5_md_370K","ifmhc_6AMU_md_300K","ifmhc_6AMU_md_370K"]:
    add_traj(j,f"native MD {j[-4:]}", "6AM5" if "6AM5" in j else "6AMU", 8)
for j in sorted(os.listdir(RES)):
    if j.startswith("ifmhc_md_6AM"): add_traj(j,"design MD","6AM5" if "6AM5" in j else "6AMU",10)
    if j.startswith("ifmhc_relax_6AM"): add_pdb(f"{RES}/{j}/relaxed.pdb","openmm relax","6AM5" if "6AM5" in j else "6AMU")
print(f"collected {len(PTS)} frames")
src_arr=np.array([p[0] for p in PTS]); seed_arr=np.array([p[1] for p in PTS])
ci={s:k for k,(sr,s,_) in enumerate(PTS) if sr=="crystal"}

def feat_matched(key, idxset=None):
    com=None
    for _,_,c in PTS:
        d=c[key]
        if idxset is not None: d={i:d[i] for i in d if i in idxset}
        com=set(d) if com is None else com&set(d)
    com=sorted(com);
    X=np.full((len(PTS),len(com)*3),np.nan)
    for r,(_,_,c) in enumerate(PTS):
        if all(i in c[key] for i in com): X[r]=np.array([c[key][i] for i in com]).ravel()
    return X
def feat_pep():
    return np.array([c["pep"].ravel() for _,_,c in PTS])
def feat_tcr():
    A=feat_matched("tcra"); B=feat_matched("tcrb"); return np.hstack([A,B])
def feat_cdr3():
    # framework-superpose each V-domain (domain CA minus CDR3), then take CDR3 CA
    fwA=[i for i in range(len(RA)) if i not in set(C3A)]; fwB=[i for i in range(len(RB)) if i not in set(C3B)]
    refA={i:REF["tcra"][1][i] for i in range(len(RA))}; refB={i:REF["tcrb"][1][i] for i in range(len(RB))}
    rows=[]
    for _,_,c in PTS:
        out=[]
        for dd,fw,loop,ref in [(c["tcra"],fwA,C3A,refA),(c["tcrb"],fwB,C3B,refB)]:
            have=[i for i in fw if i in dd]
            if len(have)<20 or not all(i in dd for i in loop): out=None; break
            R,t=CL._RT(np.array([dd[i] for i in have]), np.array([ref[i] for i in have]))
            out.extend([(dd[i]@R+t) for i in loop])
        rows.append(np.array(out).ravel() if out is not None else None)
    d=max(len(r) for r in rows if r is not None)
    return np.array([r if (r is not None and len(r)==d) else np.full(d,np.nan) for r in rows])

def rmsd_rows(A,b): return np.sqrt((((A-b).reshape(len(A),-1,3))**2).sum(2).mean(1))
def pca(X,mask):
    m=mask&np.isfinite(X).all(1); Xs=X[m]; mu=Xs.mean(0)
    U,S,Vt=np.linalg.svd(Xs-mu,full_matrices=False); ev=(S**2)/(S**2).sum()
    P=np.full((len(X),2),np.nan); P[m]=(Xs-mu)@Vt[:2].T; return P,ev

t10=plt.cm.tab10.colors
CMAP={("crystal","6AM5"):"#0b0b0b",("crystal","6AMU"):"#6f6f6f",
      ("native MD 300K","6AM5"):t10[0],("native MD 300K","6AMU"):t10[1],
      ("native MD 370K","6AM5"):t10[2],("native MD 370K","6AMU"):t10[3],
      ("openmm relax","6AM5"):t10[4],("openmm relax","6AMU"):t10[6],
      ("design MD","6AM5"):t10[8],("design MD","6AMU"):t10[9]}
ORDER=[("crystal","6AM5"),("crystal","6AMU"),("native MD 300K","6AM5"),("native MD 300K","6AMU"),
       ("native MD 370K","6AM5"),("native MD 370K","6AMU"),("openmm relax","6AM5"),("openmm relax","6AMU"),
       ("design MD","6AM5"),("design MD","6AMU")]
def style(src,seed):
    mk="*" if src=="crystal" else ("o" if seed=="6AM5" else "^")
    sz=210 if src=="crystal" else (46 if src=="openmm relax" else (22 if src.startswith("native") else 9))
    return mk,sz
def draw(ax,xv,yv):
    for src,seed in ORDER:
        ks=[k for k,p in enumerate(PTS) if p[0]==src and p[1]==seed and np.isfinite(xv[k]) and np.isfinite(yv[k])]
        if not ks: continue
        mk,sz=style(src,seed)
        ax.scatter(xv[ks],yv[ks],marker=mk,s=sz,c=[CMAP[(src,seed)]],alpha=(1 if src=="crystal" else .6),
                   edgecolor=("k" if src in("crystal","openmm relax") else "none"),linewidth=.4,
                   zorder=(7 if src=="crystal" else 3),label=f"{src}·{'GIG' if seed=='6AM5' else 'DRG'}")

COMPS=[("peptide (register)",feat_pep),("MHC α1α2 floor",lambda:feat_matched("mhc")),
       ("β2-microglobulin",lambda:feat_matched("b2m")),("TCR Vα+Vβ (pose)",feat_tcr),
       ("CDR3α+β loops (framework-fit)",feat_cdr3)]
nd=(src_arr!="design MD"); no=nd&(src_arr!="openmm relax")
fig,axs=plt.subplots(len(COMPS),4,figsize=(21,5.0*len(COMPS)))
for ri,(name,fn) in enumerate(COMPS):
    X=fn(); r5=rmsd_rows(X,X[ci["6AM5"]]); rU=rmsd_rows(X,X[ci["6AMU"]])
    sep=float(np.sqrt(((X[ci["6AM5"]]-X[ci["6AMU"]]).reshape(-1,3)**2).sum(1).mean()))
    Pa,eva=pca(X,np.ones(len(X),bool)); Pn,evn=pca(X,nd); Po,evo=pca(X,no)
    draw(axs[ri,0],r5,rU); lim=np.nanmax([np.nanmax(r5),np.nanmax(rU)])*1.02
    axs[ri,0].plot([0,lim],[0,lim],"--",c="#898781",lw=.7)
    axs[ri,0].set_ylabel(f"{name}\n\nCα-RMSD to 6AMU/DRG (Å)",fontsize=8)
    axs[ri,0].set_xlabel("Cα-RMSD to 6AM5/GIG (Å)")
    axs[ri,0].set_title(f"{name} — CA-RMSD to crystals (GIG↔DRG={sep:.2f} Å)",fontsize=8.5)
    for cj,(P,ev,lab) in enumerate([(Pa,eva,"all sources"),(Pn,evn,"no design MD"),(Po,evo,"native+crystal only")]):
        draw(axs[ri,cj+1],P[:,0],P[:,1])
        axs[ri,cj+1].set_xlabel(f"PC1 {ev[0]*100:.0f}%"); axs[ri,cj+1].set_ylabel(f"PC2 {ev[1]*100:.0f}%")
        axs[ri,cj+1].set_title(f"{name} — PCA, {lab}",fontsize=8.5)
    print(f"{name:32s} crystal GIG↔DRG separation = {sep:.2f} Å")
h,l=axs[0,0].get_legend_handles_labels()
fig.legend(h,l,loc="lower center",ncol=10,fontsize=8,frameon=False,bbox_to_anchor=(0.5,-0.012))
fig.suptitle("Per-component register analysis — CA-RMSD + PCAs by structural component (MHC-floor frame; CDR3 framework-fit)  ○=GIG ▲=DRG [tab10]",fontsize=12,y=1.0)
plt.tight_layout(rect=[0,0.03,1,0.995]); plt.savefig("/home/ubuntu/if-mhc/native_components_grid.png",dpi=140,bbox_inches="tight")
print("saved native_components_grid.png")
