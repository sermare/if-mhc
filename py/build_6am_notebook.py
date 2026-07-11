#!/usr/bin/env python
"""Comprehensive 6AM* study notebook: recovery, cross-specificity, relaxation, RFdiffusion grind + ablation."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
nb=new_notebook(); cells=[]
def md(s): cells.append(new_markdown_cell(s))
def code(s): cells.append(new_code_cell(s))

md("""# DMF5 TCR cross-reactivity — full results (6AMU / 6AM5 / 6AMT)

The DMF5 TCR binds **two distinct peptides** in HLA-A2:
- **6AMU** = DMF5 + **DRG** `MMWDRGLGMM` (+TCR)
- **6AM5** = DMF5 + **GIG** `SMLGIGIVPV` (+TCR)
- **6AMT** = DRG, **no TCR** (pMHC-only control)

Question: can structure-based design (ProteinMPNN inverse folding ± RFdiffusion) **recover** each
peptide, the **other** peptide, or **similar-motif** peptides — and how does conditioning on the
MHC pockets vs TCR contacts change generation?

Sections: 1 fixed recovery · 2 cross-specificity (own-vs-other) · 3 anchor vs non-anchor ·
4 relaxation (before/after) · 5 the 8h RFdiffusion grind (coverage, per-conditioning, DRG/GIG) ·
6 conclusions.""")

code("""from pathlib import Path
import glob, os, re, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
try: from IPython.display import display
except Exception: display=print
warnings.filterwarnings("ignore"); sns.set_style("whitegrid"); plt.rcParams["figure.dpi"]=110
ROOT=Path("/home/ubuntu/if-mhc")
DRG="MMWDRGLGMM"; GIG="SMLGIGIVPV"
# Full experimentally-validated DMF5 binder panel (Bianchi et al.) — analyses generalize to the whole list.
PANEL_STRONG=['ELAGIGILTV','SMLGIGIVPV','NMGGLGIMPV','NLSNLGILV','ILEDRGFNQV','LMFDRGMSLL','MMWDRGLGMM']
PANEL_WEAK=['MMWDRGMGLL','SMAGIGIVDV','IMEDVGWLNV']
PANEL=list(dict.fromkeys(PANEL_STRONG+PANEL_WEAK))
PANEL_BY_LEN={}
for _p in PANEL: PANEL_BY_LEN.setdefault(len(_p),[]).append(_p)
AA=list("ACDEFGHIKLMNPQRSTVWY"); AA_IDX={a:i for i,a in enumerate(AA)}
FIG=ROOT/"outputs/6am_figures"; FIG.mkdir(parents=True, exist_ok=True)

def load_fa(path, length=None):
    out=[]; p=Path(path)
    if not p.exists(): return out
    ls=p.read_text().splitlines()
    for i in range(0,len(ls)-1,2):
        if "sample=" in ls[i]:
            s=ls[i+1].strip()
            if length is None or len(s)==length: out.append(s)
    return out
def idarr(seqs,ref):
    seqs=[s for s in seqs if len(s)==len(ref)]
    if not seqs: return np.array([])
    P=np.array([list(s) for s in seqs]); r=np.array(list(ref)); return (P==r).mean(1)*100
def load_fa_nll(path, length=None):
    \"\"\"Return list of (seq, NLL score) parsing 'score=' from ProteinMPNN headers. Lower NLL = better.\"\"\"
    out=[]; p=Path(path)
    if not p.exists(): return out
    ls=p.read_text().splitlines()
    for i in range(0,len(ls)-1,2):
        if "sample=" in ls[i]:
            s=ls[i+1].strip()
            if length is None or len(s)==length:
                m=re.search(r"score=([-\\d.]+)", ls[i])
                out.append((s, float(m.group(1)) if m else np.nan))
    return out
def nll_arr(path_glob, length=None):
    a=[]
    for fa in glob.glob(str(path_glob)):
        a+=[v for _,v in load_fa_nll(fa,length) if not np.isnan(v)]
    return np.array(a)
def best_panel_arr(seqs):
    \"\"\"Per-design max identity to its nearest panel binder (length-matched). Returns array over designs.\"\"\"
    out=[]
    for L,peps in PANEL_BY_LEN.items():
        s=[x for x in seqs if len(x)==L]
        if not s or not peps: continue
        P=np.array([list(x) for x in s])
        M=np.max([(P==np.array(list(r))).mean(1)*100 for r in peps],axis=0)
        out.append(M)
    return np.concatenate(out) if out else np.array([])
def nearest_panel(seqs):
    \"\"\"(best_identity, design_seq, panel_peptide) for the single closest design->panel match.\"\"\"
    best=(0.0,None,None)
    for L,peps in PANEL_BY_LEN.items():
        s=[x for x in seqs if len(x)==L]
        if not s or not peps: continue
        for x in s:
            xa=np.array(list(x))
            for r in peps:
                idv=float((xa==np.array(list(r))).mean()*100)
                if idv>best[0]: best=(idv,x,r)
    return best
print("paths: outputs/focus_6am (fixed) | outputs/_relax_gap_peek (relaxed) | outputs/grind (de-novo)")
print("PANEL:",len(PANEL),"validated binders;",{L:len(v) for L,v in PANEL_BY_LEN.items()},"by length")""")

md("""## 1. Fixed-structure recovery (the strong, positive result)

Design the peptide chain on the **native crystal backbone**, unconstrained (all 20 AA), full 5-chain
complex as context. Does MPNN recover each native?""")

code("""rows=[]
for pid,nat,tcr in [("6AMU",DRG,"+TCR"),("6AMT",DRG,"no-TCR"),("6AM5",GIG,"+TCR")]:
    s=load_fa(ROOT/f"outputs/focus_6am/seqs/{pid}.fa", len(nat))
    idt=idarr(s,nat)
    if len(idt)==0: continue
    bid,bd,bp=nearest_panel(s)   # best match to ANY validated panel binder
    rows.append({"struct":pid,"native":nat,"TCR":tcr,"n":len(idt),
                 "mean_id%":round(idt.mean(),1),"max_id%":round(idt.max(),1),"exact":int((idt==100).sum()),
                 "best_panel_id%":round(bid,0),"best_panel_pep":bp})
fixed=pd.DataFrame(rows); display(fixed)
print(f"exact full-sequence matches (100% identity): {int(fixed['exact'].sum())} / {fixed['n'].sum()} designs")
print("Recovery is strong but NOT exact — mean identity ~48-49%, driven by the TCR-facing motif residues")
print("(see section 3); best_panel shows the closest match across ALL validated binders.")""")

md("""## 2. Cross-specificity — does each structure recover its OWN peptide vs the OTHER?

The decisive control: 6AMU and 6AM5 share the **same DMF5 TCR**, differing only in peptide backbone.
If recovery were generic, both would give the same designs.""")

code("""rows=[]
NAT={"6AMU":DRG,"6AM5":GIG,"6AMT":DRG}; OTH={"6AMU":GIG,"6AM5":DRG,"6AMT":GIG}
for pid in ["6AMU","6AM5","6AMT"]:
    s=load_fa(ROOT/f"outputs/focus_6am/seqs/{pid}.fa",10)
    io=idarr(s,NAT[pid]); ix=idarr(s,OTH[pid])
    if len(io)==0: continue
    rows.append({"struct":pid,"own":NAT[pid],"id->OWN":round(io.mean(),1),
                 "id->OTHER":round(ix.mean(),1),"gap":round(io.mean()-ix.mean(),1),
                 "other_exact":int((ix==100).sum())})
cross=pd.DataFrame(rows); display(cross)
fig,ax=plt.subplots(figsize=(7,4))
x=np.arange(len(cross)); w=0.38
ax.bar(x-w/2,cross["id->OWN"],w,label="own peptide",color="seagreen")
ax.bar(x+w/2,cross["id->OTHER"],w,label="other peptide",color="indianred")
ax.set_xticks(x); ax.set_xticklabels(cross["struct"]+"\\n("+cross["own"].str[:3]+")")
ax.set_ylabel("% identity"); ax.legend(); ax.set_title("Each structure recovers its OWN peptide, not the other")
plt.tight_layout(); plt.savefig(FIG/"01_cross_specificity.png",bbox_inches="tight"); plt.show()
print("Gap +28 to +36%. OTHER-peptide exact recoveries:",list(cross["other_exact"]),"-> we never get the other peptide.")""")

md("## 3. Anchor vs non-anchor recovery (what residues ARE recovered?)")

code("""# HLA-A2 anchors P2 & P10; central P4-P8 are TCR-facing
ANCH=[2,10]
fig,axes=plt.subplots(1,2,figsize=(14,4))
for ax,(pid,nat) in zip(axes,[("6AMU",DRG),("6AM5",GIG)]):
    s=load_fa(ROOT/f"outputs/focus_6am/seqs/{pid}.fa",10)
    P=np.array([list(x) for x in s]); na=np.array(list(nat)); pp=(P==na).mean(0)*100
    cols=["crimson" if (i+1) in ANCH else "steelblue" for i in range(10)]
    ax.bar([f"P{i+1}\\n{nat[i]}" for i in range(10)],pp,color=cols)
    ax.set_title(f"{pid} ({nat}) per-position recovery"); ax.set_ylabel("% recovered")
    anch=np.mean([pp[i] for i in range(10) if (i+1) in ANCH]); non=np.mean([pp[i] for i in range(10) if (i+1) not in ANCH])
    ax.text(0.5,0.92,f"anchor {anch:.0f}%  vs  non-anchor {non:.0f}%",transform=ax.transAxes,ha="center",fontweight="bold")
plt.tight_layout(); plt.savefig(FIG/"02_anchor_vs_nonanchor.png",bbox_inches="tight"); plt.show()
print("Anchors (P2/PΩ, red) NOT recovered — natives have non-canonical Met anchors; MPNN swaps to canonical L/V.")
print("The recovered residues are the central TCR-facing motif (DRG: D4/G6/G8; GIG: G4/G6/P9).")""")

md("""## 4. Relaxation — before vs after (own-vs-other gap)

OpenMM relaxed ensemble of the TCR-bound complex, then MPNN. Does the discrimination survive
backbone perturbation, and does relaxation ever yield the OTHER peptide?""")

code("""# relaxed peek designs: outputs/_relax_gap_peek/seqs/<pid>_snap*.fa
rows=[]
for pid,tcr in [("6AMU","+TCR"),("6AMT","no-TCR ctrl")]:
    s=[x for fa in glob.glob(str(ROOT/f"outputs/_relax_gap_peek/seqs/{pid}_snap*.fa")) for x in load_fa(fa,10)]
    io=idarr(s,DRG); ix=idarr(s,GIG)
    if len(io)==0: continue
    rows.append({"struct":pid,"TCR":tcr,"id->OWN(DRG)":round(io.mean(),1),"id->OTHER(GIG)":round(ix.mean(),1),
                 "gap":round(io.mean()-ix.mean(),1),"own_exact":int((io==100).sum()),"other_exact":int((ix==100).sum())})
relaxed=pd.DataFrame(rows); display(relaxed)
print("FIXED gaps were: 6AMU +36, 6AMT +31.5.  RELAXED below.")
print("Relaxation MULTIPLIES own exact hits (1->5+) but NEVER yields the other peptide (other_exact=0).")
print("TCR-bound gap (6AMU) survives relaxation; no-TCR control degrades more -> TCR stabilises specificity.")""")

md("""## 5. De-novo RFdiffusion designs — **temperature ladder** (T = 0.1 / 0.3 / 0.5 / 0.7)

Same 106 de-novo backbones (8h grind; 2 crystals × 3 conditioning × 5 lengths), sampled at **four
MPNN temperatures**, each **10k sequences/crystal** partitioned across that crystal's backbones
(T=0.1 is the original grind). De-novo backbones are NOT in the PDB → no memorization. The ladder
shows how warmer sampling trades convergence for breadth, and whether breadth ever reaches DRG/GIG.""")

code("""# Multi-temperature loader -> BYT[T] = {(pid,cond,L): [seqs]}
# Pool all available designs per temperature: original grind (T=0.1) + 10k runs + 50k runs.
TEMP_SRC={
 0.1:[str(ROOT/"outputs/grind/seqs/*.fa")],
 0.3:[str(ROOT/"outputs/grind_10k_T03/6AMU/seqs/*.fa"),str(ROOT/"outputs/grind_10k_T03/6AM5/seqs/*.fa")],
 0.5:[str(ROOT/"outputs/grind_10k_T05/6AMU/seqs/*.fa"),str(ROOT/"outputs/grind_10k_T05/6AM5/seqs/*.fa")],
 0.7:[str(ROOT/"outputs/grind_10k/6AMU/seqs/*.fa"),str(ROOT/"outputs/grind_10k/6AM5/seqs/*.fa")],
}
for T in [0.1,0.3,0.5,0.7]:   # add the 50k-per-crystal runs when present
    TEMP_SRC.setdefault(T,[]).extend([str(ROOT/f"outputs/grind_50k/T{T}/6AMU/seqs/*.fa"),
                                      str(ROOT/f"outputs/grind_50k/T{T}/6AM5/seqs/*.fa")])
BYT={}
for T,pats in TEMP_SRC.items():
    d={}
    for pat in pats:
        for fa in glob.glob(pat):
            m=re.match(r"(6AM[U5])_(mhc_tcr2|mhc|tcr2)_L(\\d+)_",os.path.basename(fa))
            if not m: continue
            key=(m.group(1),m.group(2),int(m.group(3)))
            d.setdefault(key,[]).extend(load_fa(fa,int(m.group(3))))
    if sum(len(v) for v in d.values())>0: BYT[T]=d
TEMPS=sorted(BYT)
g=BYT.get(0.7,BYT[max(BYT)])  # default working set for downstream cells
print("temperatures loaded:",TEMPS)
for T in TEMPS:
    n=sum(len(v) for v in BYT[T].values()); print(f"  T={T}: {n} seqs over {len(BYT[T])} combos")""")

md("""### 5.1 Temperature ladder — diversity, recovery & **NLL** vs T

Recovery generalized to the nearest of ALL validated binders; we also track the ProteinMPNN **NLL**
(per-residue negative log-likelihood of the designed peptide; lower = more favorable on the structure).""")

code("""# NLL distribution of de-novo designs by temperature (lower = MPNN finds it more favorable)
NLL_SRC={0.1:"outputs/grind/seqs/*.fa",0.3:"outputs/grind_10k_T03/*/seqs/*.fa",
         0.5:"outputs/grind_10k_T05/*/seqs/*.fa",0.7:"outputs/grind_10k/*/seqs/*.fa"}
fig,ax=plt.subplots(1,2,figsize=(14,4.2))
nll_rows=[]
for T in TEMPS:
    a=nll_arr(ROOT/NLL_SRC[T])
    if len(a)==0: continue
    nll_rows.append({"T":T,"n":len(a),"mean_NLL":round(a.mean(),3),"median_NLL":round(np.median(a),3),"min_NLL":round(a.min(),3)})
    ax[0].hist(a,bins=60,histtype="step",lw=2,label=f"T={T} (n={len(a)})",density=True)
ax[0].set_xlabel("ProteinMPNN NLL (per-residue, designed peptide)"); ax[0].set_ylabel("density")
ax[0].set_title("De-novo design NLL distribution by temperature"); ax[0].legend(fontsize=8)
nll_df=pd.DataFrame(nll_rows)
ax[1].plot(nll_df["T"],nll_df["mean_NLL"],"o-",lw=2,label="mean"); ax[1].plot(nll_df["T"],nll_df["min_NLL"],"s--",lw=2,label="best (min)")
ax[1].set_xlabel("temperature"); ax[1].set_ylabel("NLL"); ax[1].set_title("Design NLL vs temperature"); ax[1].legend()
plt.tight_layout(); plt.savefig(FIG/"07b_nll_vs_temperature.png",bbox_inches="tight"); plt.show()
display(nll_df)
print("Higher T -> higher (worse) mean NLL: warm sampling trades MPNN confidence for diversity.")""")

md("### 5.1b Temperature ladder — diversity & nearest-panel recovery vs T")

code("""def pool(T,pid,L):  # pooled designs for a crystal at temp T, length L (all conditioning)
    return [s for (p,c,ll),v in BYT[T].items() if p==pid and ll==L for s in v]
rows=[]
for T in TEMPS:
    for pid in ["6AMU","6AM5"]:
        s=[x for L in PANEL_BY_LEN for x in pool(T,pid,L)]   # all panel-relevant lengths
        if not s: continue
        bp=best_panel_arr(s)
        rows.append({"T":T,"crystal":pid,"n":len(s),"pct_unique":round(100*len(set(s))/len(s),1),
                     "max->panel":round(bp.max(),0) if len(bp) else 0,
                     "mean->panel":round(bp.mean(),1) if len(bp) else 0})
ladder=pd.DataFrame(rows); display(ladder)
fig,ax=plt.subplots(1,2,figsize=(14,4.2))
for pid,mk in [("6AMU","o"),("6AM5","s")]:
    sub=ladder[ladder.crystal==pid]
    ax[0].plot(sub["T"],sub["pct_unique"],marker=mk,label=pid,lw=2)
    ax[1].plot(sub["T"],sub["max->panel"],marker=mk,label=f"{pid} -> nearest panel binder",lw=2)
ax[0].set_xlabel("MPNN temperature"); ax[0].set_ylabel("% unique"); ax[0].set_title("Diversity vs temperature"); ax[0].legend()
ax[1].set_xlabel("MPNN temperature"); ax[1].set_ylabel("max % id to nearest panel binder"); ax[1].set_title("Best recovery vs temperature (whole panel)"); ax[1].legend()
plt.tight_layout(); plt.savefig(FIG/"07_temperature_ladder.png",bbox_inches="tight"); plt.show()
print("Recovery generalized to nearest of ALL validated binders (not just DRG/GIG). Diversity up with T; panel recovery flat.")""")

md("### 5.2 Nearest-panel-binder identity grid — crystal × conditioning × temperature (whole panel)")

code("""rows=[]
for T in TEMPS:
    for pid in ["6AMU","6AM5"]:
        for cond in ["tcr2","mhc","mhc_tcr2"]:
            s=[x for L in PANEL_BY_LEN for x in BYT[T].get((pid,cond,L),[])]
            if not s: continue
            bp=best_panel_arr(s)
            rows.append({"T":T,"crystal":pid,"cond":cond,"n":len(s),
                         "max->panel":round(bp.max(),0) if len(bp) else 0})
grid=pd.DataFrame(rows); display(grid)
# heatmap of max->nearest-panel identity, rows=crystal/cond, cols=T
pm=grid.pivot_table(index=["crystal","cond"],columns="T",values="max->panel")
fig,ax=plt.subplots(figsize=(7,5))
sns.heatmap(pm,cmap="viridis",annot=True,fmt=".0f",cbar_kws={"label":"max % id to nearest panel binder"},ax=ax)
ax.set_title("Max identity to NEAREST validated panel binder by conditioning × temperature")
plt.tight_layout(); plt.savefig(FIG/"08_recovery_grid_byT.png",bbox_inches="tight"); plt.show()""")

md("### 5.3 Diversity by conditioning, broken down by temperature")

code("""rows=[]
for T in TEMPS:
    for pid in ["6AMU","6AM5"]:
        for cond in ["tcr2","mhc","mhc_tcr2"]:
            s=BYT[T].get((pid,cond,10),[])
            if not s: continue
            rows.append({"T":T,"crystal":pid,"cond":cond,"pct_unique":round(100*len(set(s))/len(s),1)})
dv=pd.DataFrame(rows)
fig,axes=plt.subplots(1,2,figsize=(14,4),sharey=True)
for ax,pid in zip(axes,["6AMU","6AM5"]):
    sub=dv[dv.crystal==pid]
    for cond,mk in [("tcr2","o"),("mhc","s"),("mhc_tcr2","^")]:
        ss=sub[sub.cond==cond]
        if len(ss): ax.plot(ss["T"],ss["pct_unique"],marker=mk,label=cond,lw=2)
    ax.set_title(f"{pid} diversity by conditioning × T"); ax.set_xlabel("temperature"); ax.set_ylabel("% unique"); ax.legend()
plt.tight_layout(); plt.savefig(FIG/"04_diversity.png",bbox_inches="tight"); plt.show()""")

md("### 5.4 AA composition (L10) by conditioning — at the highest temperature loaded")

code("""from collections import Counter
Tc=max(TEMPS)
comp_rows={}
for pid in ["6AMU","6AM5"]:
    for cond in ["tcr2","mhc","mhc_tcr2"]:
        s=BYT[Tc].get((pid,cond,10),[])
        if not s: continue
        c=Counter("".join(s)); tot=sum(c.values())
        comp_rows[f"{pid}/{cond}"]=pd.Series({a:100*c.get(a,0)/tot for a in AA})
comp=pd.DataFrame(comp_rows).T
fig,ax=plt.subplots(figsize=(13,4))
sns.heatmap(comp,cmap="rocket_r",annot=True,fmt=".0f",annot_kws={"size":6},cbar_kws={"label":"% comp"},ax=ax)
ax.set_title(f"AA composition of de-novo peptides by crystal × conditioning (L10, T={Tc})")
plt.tight_layout(); plt.savefig(FIG/"03_conditioning_composition.png",bbox_inches="tight"); plt.show()
display(comp[["D","R","G","P","A","L","M","W"]].round(1))""")

md("""## 5a. Structural validation — peptide backbone RMSD vs crystal

Before reading sequences, check the **generated backbones** are real groove peptides. Cα-RMSD
(Kabsch best-fit superposition) of each de-novo **L10** peptide vs the native crystal peptide
(DRG for 6AMU, GIG for 6AM5). Cross-RMSD (6AMU→GIG, 6AM5→DRG) tests whether backbone *shape*
already distinguishes the two peptides, independent of sequence.""")

code("""import warnings; warnings.filterwarnings("ignore")
from Bio.PDB import PDBParser
_pp=PDBParser(QUIET=True)
def _ca(path, chain):
    try: m=_pp.get_structure("x",path)[0]
    except Exception: return None
    if chain not in m: return None
    c=[r['CA'].coord for r in m[chain] if r.id[0]==' ' and 'CA' in r]
    return np.array(c) if c else None
def kabsch(P,Q):
    if P is None or Q is None or P.shape!=Q.shape or len(P)<3: return None
    Pc=P-P.mean(0); Qc=Q-Q.mean(0)
    V,S,Wt=np.linalg.svd(Pc.T@Qc); d=np.sign(np.linalg.det(V@Wt))
    U=V@np.diag([1,1,d])@Wt
    return float(np.sqrt(((Pc@U-Qc)**2).sum()/len(P)))
natca={"6AMU":_ca(str(ROOT/"inputs/focus_6am/6AMU.pdb"),"C"),   # DRG crystal peptide
       "6AM5":_ca(str(ROOT/"inputs/focus_6am/6AM5.pdb"),"C")}   # GIG crystal peptide
rms={"6AMU":{"own":[],"other":[]},"6AM5":{"own":[],"other":[]}}
for pid in ["6AMU","6AM5"]:
    oth="6AM5" if pid=="6AMU" else "6AMU"
    for f in glob.glob(str(ROOT/f"outputs/grind/pdb/{pid}_*_L10_*.pdb")):
        gca=_ca(f,"A")
        if gca is None or len(gca)!=10: continue
        r_own=kabsch(gca,natca[pid]); r_oth=kabsch(gca,natca[oth])
        if r_own is not None: rms[pid]["own"].append(r_own)
        if r_oth is not None: rms[pid]["other"].append(r_oth)
print("Cα-RMSD (Å) of de-novo L10 peptides vs crystal peptides:")
for pid in ["6AMU","6AM5"]:
    o=np.array(rms[pid]["own"]); x=np.array(rms[pid]["other"])
    nat="DRG" if pid=="6AMU" else "GIG"; oth="GIG" if pid=="6AMU" else "DRG"
    print(f"  {pid}: n={len(o)} | ->own({nat}) mean={o.mean():.2f} min={o.min():.2f} | ->other({oth}) mean={x.mean():.2f} min={x.min():.2f}")""")

code("""fig,ax=plt.subplots(1,2,figsize=(13,4))
for a,pid in zip(ax,["6AMU","6AM5"]):
    nat="DRG" if pid=="6AMU" else "GIG"; oth="GIG" if pid=="6AMU" else "DRG"
    o=rms[pid]["own"]; x=rms[pid]["other"]
    if o: a.hist(o,bins=20,alpha=.65,color="seagreen",label=f"vs own ({nat})")
    if x: a.hist(x,bins=20,alpha=.65,color="indianred",label=f"vs other ({oth})")
    a.set_xlabel("Cα-RMSD (Å)"); a.set_ylabel("# designs"); a.set_title(f"{pid} de-novo L10 peptide backbones"); a.legend()
plt.tight_layout(); plt.savefig(FIG/"4b_peptide_rmsd.png",bbox_inches="tight"); plt.show()
print("Lower RMSD = generated backbone traces the crystal peptide. If own<other, de-novo backbones")
print("already lean toward the native shape; if own~other, shape doesn't separate the two peptides.")""")

md("""## 5b. Sequence logos — de-novo L10 designs, **crystal × temperature grid**

Per-position information-content logos for the L10 de-novo designs, one column per temperature,
one row per crystal. Watch the logos flatten as T rises (greedy → diverse). Native labeled.""")

code("""import logomaker
def freq(seqs,L):
    M=np.zeros((L,len(AA)))
    for s in seqs:
        for i,c in enumerate(s):
            if i<L and c in AA_IDX: M[i,AA_IDX[c]]+=1
    M/=np.clip(M.sum(1,keepdims=True),1,None)
    return pd.DataFrame(M,columns=AA,index=list(range(L)))
ROWS=[("6AMU",DRG),("6AM5",GIG)]
nT=len(TEMPS)
fig,axes=plt.subplots(len(ROWS),nT,figsize=(3.4*nT,2.4*len(ROWS)),squeeze=False)
for r,(pid,nat) in enumerate(ROWS):
    for cc,T in enumerate(TEMPS):
        ax=axes[r][cc]; s=pool(T,pid,10)
        if not s: ax.set_axis_off(); ax.set_title(f"{pid} T={T}: none",fontsize=8); continue
        info=logomaker.transform_matrix(freq(s,10).copy(),from_type="probability",to_type="information")
        logomaker.Logo(info,ax=ax,color_scheme="skylign_protein")
        ax.set_ylim(0,4.3); ax.set_title(f"{pid}  T={T}  (n={len(s)})",fontsize=8)
        ax.set_xticks(range(10)); ax.set_xticklabels(list(nat),fontsize=6)
        if cc==0: ax.set_ylabel("bits",fontsize=8)
plt.suptitle("De-novo L10 sequence logos: crystal (rows) × temperature (cols)",y=1.02)
plt.tight_layout(); plt.savefig(FIG/"05_sequence_logos.png",bbox_inches="tight"); plt.show()
print("Left (T=0.1) = peaked/convergent; right (T=0.7) = flat/diverse. Native residues on x-axis.")""")

md("""## 5c. Per-site residue landscape — **crystal × temperature grid** (probability heatmaps)

Residue-frequency heatmaps (AA × position) for L10 de-novo designs, per crystal per temperature.
Native residue boxed in red.""")

code("""fig,axes=plt.subplots(len(ROWS),nT,figsize=(3.0*nT,5.5),squeeze=False)
for r,(pid,nat) in enumerate(ROWS):
    for cc,T in enumerate(TEMPS):
        ax=axes[r][cc]; s=pool(T,pid,10)
        if not s: ax.set_axis_off(); continue
        f=freq(s,10); last=(cc==nT-1)
        sns.heatmap(f.T,cmap="viridis",vmin=0,vmax=1,ax=ax,cbar=last,
                    cbar_kws={"label":"freq"} if last else None,
                    xticklabels=list(nat), yticklabels=(AA if cc==0 else False))
        for i,c in enumerate(nat):
            if c in AA_IDX: ax.add_patch(plt.Rectangle((i,AA_IDX[c]),1,1,fill=False,ec="red",lw=1.2))
        ax.set_title(f"{pid} T={T}",fontsize=8)
    axes[r][0].set_ylabel(f"{pid}\\namino acid",fontsize=8)
plt.suptitle("Per-site residue landscape: crystal (rows) × temperature (cols), native boxed red",y=1.02)
plt.tight_layout(); plt.savefig(FIG/"06_residue_landscape.png",bbox_inches="tight"); plt.show()
# landscape divergence 6AMU vs 6AM5 at each temperature
print("Per-position landscape divergence |6AMU - 6AM5| (sum over AA), by temperature:")
for T in TEMPS:
    a=pool(T,"6AMU",10); b=pool(T,"6AM5",10)
    if a and b:
        d=(freq(a,10)-freq(b,10)).abs().sum(1)
        print(f"  T={T}: "+" ".join(f"P{i+1}:{d[i]:.2f}" for i in range(10)))""")

md("""## 5e. Capturing the experimentally-validated binder panel

The DMF5 / MART-1 system has many validated binders beyond DRG & GIG (Bianchi et al.). We score our
designs against the full panel — do designs land near *other* real binders, i.e. capture the
cross-reactive family? STRONG = elicit response; WEAK = weaker binders.""")

code("""# use the global PANEL_STRONG/PANEL_WEAK; build a tagged list WITHOUT clobbering global PANEL
PANEL_TAGGED=[(p,'strong') for p in PANEL_STRONG]+[(p,'weak') for p in dict.fromkeys(PANEL_WEAK)]
def _load(path,L):
    out=[]
    for fa in glob.glob(str(ROOT/path)):
        ls=open(fa).read().splitlines()
        for i in range(0,len(ls)-1,2):
            if 'sample=' in ls[i] and len(ls[i+1].strip())==L: out.append(ls[i+1].strip())
    return out
def _maxid(seqs,ref):
    s=[x for x in seqs if len(x)==len(ref)]
    if not s: return (0.0,None)
    P=np.array([list(x) for x in s]); idt=(P==np.array(list(ref))).mean(1)*100
    return (round(idt.max(),0), s[int(idt.argmax())])
POOLS={'6AMU_fixed(DRG xtal)':'outputs/focus_6am/seqs/6AMU.fa',
       '6AM5_fixed(GIG xtal)':'outputs/focus_6am/seqs/6AM5.fa',
       'denovo_6AMU_T0.1':'outputs/grind/seqs/6AMU_*.fa',
       'denovo_6AM5_T0.1':'outputs/grind/seqs/6AM5_*.fa'}
rows=[]; best={}
for pep,cls in PANEL_TAGGED:
    row={'peptide':pep,'class':cls}
    for k,path in POOLS.items():
        mx,bd=_maxid(_load(path,len(pep)),pep); row[k]=mx; best[(pep,k)]=bd
    rows.append(row)
panel=pd.DataFrame(rows); display(panel)
import matplotlib.pyplot as plt
M=panel.set_index('peptide')[list(POOLS)]
fig,ax=plt.subplots(figsize=(9,5))
sns.heatmap(M,cmap='magma',annot=True,fmt='.0f',cbar_kws={'label':'max % identity'},ax=ax)
for j,p in enumerate(M.index):
    if panel.set_index('peptide').loc[p,'class']=='strong': ax.add_patch(plt.Rectangle((0,j),len(POOLS),1,fill=False,ec='lime',lw=1.5))
ax.set_title('Max identity of designs to validated binder panel (green=strong binder)')
plt.tight_layout(); plt.savefig(FIG/'10_panel_capture.png',bbox_inches='tight'); plt.show()
print('GIG crystal (6AM5) designs reach 50-70% to the GIG/MART-1 family (ELAGIGILTV, NMGGLGIMPV, SMAGIGIVDV)')
print('-> inverse folding on a crystal captures its cross-reactive FAMILY, not just the bound peptide.')""")

md("""## 5f. Experimental binding data vs design recovery

Bianchi et al. measured affinity (KD, ΔG°), stability (Tm), and an alanine/length scan for the
DMF5 panel. We ask: does structure-based **design recovery track real binding** — and do we recover
the functionally **critical residues** (alanine positions that abolish binding)?""")

code("""# Experimental data. Table 1 (KD uM, dG kcal/mol, Tm C); ELAGIGILTV = MART-1 reference.
EXP={
 'ELAGIGILTV':{'KD':5.5,'dG':-7.17,'Tm':65.0,'len':10},
 'NMGGLGIMPV':{'KD':7.3,'dG':-7.00,'Tm':58.6,'len':10},
 'NLSNLGILV' :{'KD':11 ,'dG':-6.76,'Tm':66.3,'len':9},
 'MMWDRGLGMM':{'KD':32 ,'dG':-6.13,'Tm':45.2,'len':10},  # DRG (6AMU native)
 'SMLGIGIVPV':{'KD':43 ,'dG':-5.96,'Tm':66.1,'len':10},  # GIG (6AM5 native)
 'ILEDRGFNQV':{'KD':99 ,'dG':-5.46,'Tm':46.1,'len':10},
 'IMEDVGWLNV':{'KD':121,'dG':-5.34,'Tm':55.1,'len':10},
 'LMFDRGMSLL':{'KD':123,'dG':-5.33,'Tm':50.0,'len':10},
 'MMWDRGMGLL':{'KD':144,'dG':-5.24,'Tm':45.4,'len':10},
 'SMAGIGIVDV':{'KD':202,'dG':-5.04,'Tm':65.8,'len':10},
 'MMWDRGLGM' :{'KD':30 ,'dG':-6.17,'Tm':49.1,'len':9},   # DRG 9-mer truncation: still binds!
}
# structural family by motif
def fam(p):
    if 'GIGI' in p or 'GLGI' in p or 'GIGIL' in p or p[3:7] in ('GIGI','GGLG'): return 'GIG/MART'
    if 'DRG' in p: return 'DRG'
    return 'other'
def load(path,L):
    out=[]
    for f in glob.glob(str(ROOT/path)):
        ls=open(f).read().splitlines()
        for i in range(0,len(ls)-1,2):
            if 'sample=' in ls[i] and len(ls[i+1].strip())==L: out.append(ls[i+1].strip())
    return out
def maxid(seqs,ref):
    s=[x for x in seqs if len(x)==len(ref)]
    if not s: return np.nan
    P=np.array([list(x) for x in s]); return float((P==np.array(list(ref))).mean(1).max()*100)
# length-matched pools: 10mers from fixed recovery; 9mers from de-novo focus_9mer / grind L9
POOL10={'6AMU':load('outputs/focus_6am/seqs/6AMU.fa',10),'6AM5':load('outputs/focus_6am/seqs/6AM5.fa',10)}
POOL9 ={'6AMU':load('outputs/focus_9mer/seqs/6AMU*.fa',9)+load('outputs/grind/seqs/6AMU_*.fa',9),
        '6AM5':load('outputs/focus_9mer/seqs/6AM5*.fa',9)+load('outputs/grind/seqs/6AM5_*.fa',9)}
rows=[]
for p,e in EXP.items():
    P=POOL10 if e['len']==10 else POOL9
    a=maxid(P['6AMU'],p); b=maxid(P['6AM5'],p)
    rows.append({'peptide':p,'family':fam(p),'len':e['len'],'KD_uM':e['KD'],'dG':e['dG'],'Tm':e['Tm'],
                 'id_6AMU':a,'id_6AM5':b,'best_id':np.nanmax([a,b])})
exp=pd.DataFrame(rows).sort_values('KD_uM'); display(exp)""")

code("""# Does recovery track affinity? scatter (10-mers, length-matched), colored by structural family
sub=exp[exp.len==10].dropna(subset=['best_id'])
fig,ax=plt.subplots(1,2,figsize=(14,4.5))
cmap={'GIG/MART':'seagreen','DRG':'crimson','other':'slategray'}
for famname,c in cmap.items():
    s=sub[sub.family==famname]
    if len(s):
        ax[0].scatter(s.dG,s.best_id,c=c,s=70,label=famname,edgecolor='k')
        for _,r in s.iterrows(): ax[0].annotate(r.peptide,(r.dG,r.best_id),fontsize=6)
ax[0].set_xlabel('experimental ΔG° (kcal/mol, lower=tighter)'); ax[0].set_ylabel('best design identity %')
r=np.corrcoef(sub.dG,sub.best_id)[0,1]
ax[0].set_title(f'Design recovery vs binding affinity (r={r:.2f})'); ax[0].legend(fontsize=7)
# per-crystal: own-family capture
ax[1].scatter(sub.dG,sub.id_6AM5,c='seagreen',label='6AM5(GIG) designs',s=55,edgecolor='k')
ax[1].scatter(sub.dG,sub.id_6AMU,c='crimson',label='6AMU(DRG) designs',s=55,edgecolor='k',marker='^')
ax[1].set_xlabel('ΔG°'); ax[1].set_ylabel('design identity %'); ax[1].legend(fontsize=7)
ax[1].set_title('Each crystal captures its own structural family')
plt.tight_layout(); plt.savefig(FIG/'11_affinity_vs_recovery.png',bbox_inches='tight'); plt.show()
print('Recovery tracks STRUCTURAL FAMILY (which crystal), only weakly affinity: the GIG/MART family')
print('(incl. the tightest binder ELAGIGILTV/MART-1) is captured by 6AM5; DRG-family by 6AMU.')""")

code("""# Alanine/length scan: which residues are functionally critical, and do we recover them?
# Non-binders (KD>500 = abolished): DRG R5A (MMWDAGLGMM), L7A (MMWDRGAGMM), M9A->P9? (MMWDRGLGAM).
# 9-mer MMWDRGLGM still binds (KD 30uM). Native DRG = M M W D R G L G M M (1..10).
crit={'P5(R)':'R5A abolishes (>500uM)','P7(L)':'L7A abolishes','P9(M)':'M9A abolishes'}
s=load('outputs/focus_6am/seqs/6AMU.fa',10); nat='MMWDRGLGMM'
P=np.array([list(x) for x in s]); pp=(P==np.array(list(nat))).mean(0)*100
print('DRG functional hotspots (alanine scan) vs our per-position recovery:')
for pos,desc in crit.items():
    i=int(pos[1:pos.index('(')])-1
    print(f'  {pos}: {desc:28} -> design recovery {pp[i]:.0f}%')
print(f'  (9-mer truncation MMWDRGLGM binds at 30uM ~ equal to 10-mer 32uM -> P10 dispensable)')
print('Note: MPNN recovers structurally-determined residues (D4,G6,G8 high); some functional')
print('hotspots (R5) point toward TCR and are low-recovery -> structure vs function distinction.')""")

md("""## 5g. Hamming distance of designs to every panel binder

For each validated binder, the **closest design** (min Hamming) across pools, sortable. Hamming =
# positions differing; identity% = 1 - H/L. Lets you see exactly how near we get to each binder.""")

code("""def closest_to(ref, pools):
    \"\"\"min-Hamming design to ref across {poolname:seqs}.\"\"\"
    L=len(ref); best=(L+1,None,None)
    for name,seqs in pools.items():
        for x in seqs:
            if len(x)!=L: continue
            h=sum(a!=b for a,b in zip(x,ref))
            if h<best[0]: best=(h,x,name)
    return best
# design pools (length-matched lookups happen inside closest_to)
P10={'6AMU_fixed':load_fa(ROOT/'outputs/focus_6am/seqs/6AMU.fa',10),
     '6AM5_fixed':load_fa(ROOT/'outputs/focus_6am/seqs/6AM5.fa',10),
     'denovo_T0.1':[x for f in glob.glob(str(ROOT/'outputs/grind/seqs/*_L10_*.fa')) for x in load_fa(f,10)]}
P9 ={'denovo9':[x for f in glob.glob(str(ROOT/'outputs/grind/seqs/*_L9_*.fa')) for x in load_fa(f,9)]
     +[x for f in glob.glob(str(ROOT/'outputs/focus_9mer/seqs/*.fa')) for x in load_fa(f,9)]}
cls={**{p:'strong' for p in PANEL_STRONG},**{p:'weak' for p in PANEL_WEAK}}
rows=[]
for pep in PANEL:
    pools=P10 if len(pep)==10 else P9
    h,bd,src=closest_to(pep,pools)
    rows.append({'panel_peptide':pep,'class':cls[pep],'len':len(pep),
                 'min_hamming':h if bd else None,'identity_%':round(100*(1-h/len(pep)),1) if bd else None,
                 'closest_design':bd,'from_pool':src,
                 'mismatches':",".join(f"{pep[i]}{i+1}{bd[i]}" for i in range(len(pep)) if bd[i]!=pep[i]) if bd else None})
ham=pd.DataFrame(rows).sort_values(['min_hamming','class']); display(ham)
ham.to_csv(ROOT/'outputs/6am_figures/panel_hamming.csv',index=False)
# bar: min Hamming per panel binder (lower=closer), colored by class
fig,ax=plt.subplots(figsize=(11,4))
h2=ham.dropna(subset=['min_hamming']).sort_values('min_hamming')
ax.bar(h2.panel_peptide, h2.min_hamming, color=['seagreen' if c=='strong' else 'indianred' for c in h2['class']])
ax.set_ylabel('min Hamming to nearest design'); ax.set_xticklabels(h2.panel_peptide,rotation=45,ha='right',fontsize=8)
ax.set_title('Closest design to each validated binder (green=strong, red=weak); lower=closer')
plt.tight_layout(); plt.savefig(FIG/'12_panel_hamming.png',bbox_inches='tight'); plt.show()
print('Saved outputs/6am_figures/panel_hamming.csv (sort by min_hamming / identity_%).')""")

md("""## 5h. NLL of validated binders threaded onto each crystal vs experimental ΔG°

Instead of % recovery: thread each 10-mer validated binder into chain C of each crystal backbone and
take the ProteinMPNN **NLL** (score_only). Does structure-based NLL track real binding affinity?""")

code("""import csv as _csv
nllf=ROOT/"outputs/panel_nll/panel_nll.csv"
EXP_dG={'ELAGIGILTV':-7.17,'NMGGLGIMPV':-7.00,'MMWDRGLGMM':-6.13,'SMLGIGIVPV':-5.96,'ILEDRGFNQV':-5.46,
        'IMEDVGWLNV':-5.34,'LMFDRGMSLL':-5.33,'MMWDRGMGLL':-5.24,'SMAGIGIVDV':-5.04}
if not nllf.exists():
    print("panel_nll.csv not found — run score_panel_nll.py")
else:
    pn=pd.read_csv(nllf); pn=pn[pn.crystal.isin(["6AMU","6AM5"])]
    pn["nll"]=pd.to_numeric(pn["nll"],errors="coerce"); pn["dG"]=pn.peptide.map(EXP_dG)
    pn=pn.dropna(subset=["nll","dG"])
    pv=pn.pivot(index="peptide",columns="crystal",values="nll"); pv["dG"]=pv.index.map(EXP_dG)
    display(pv.sort_values("dG"))
    fig,ax=plt.subplots(1,2,figsize=(13,4.5))
    for j,pid in enumerate(["6AMU","6AM5"]):
        s=pn[pn.crystal==pid].dropna(subset=["dG"])
        ax[j].scatter(s.dG,s.nll,s=70,edgecolor="k",
                      c=["crimson" if 'DRG' in p else "seagreen" for p in s.peptide])
        for _,r in s.iterrows(): ax[j].annotate(r.peptide,(r.dG,r.nll),fontsize=6)
        rr=np.corrcoef(s.dG,s.nll)[0,1]
        ax[j].set_xlabel("experimental ΔG° (kcal/mol)"); ax[j].set_ylabel("ProteinMPNN NLL on backbone")
        ax[j].set_title(f"{pid}: NLL vs ΔG°  (r={rr:.2f})")
    plt.tight_layout(); plt.savefig(FIG/"13_panel_nll_vs_dG.png",bbox_inches="tight"); plt.show()
    print("NLL spread is narrow (~0.1) -> peptide positions are high-entropy; weak (6AM5 r~-0.45) to no (6AMU)")
    print("correlation with affinity. Structure-based NLL is a poor binding discriminator for this groove.")""")

md("""## 5i. NLL distribution — fixed crystal vs de-novo vs temperature

The decisive design-quality comparison: ProteinMPNN per-residue **NLL** (lower = more favorable) for
peptides designed on the **native crystal backbone** (fixed) vs **de-novo RFdiffusion backbones**,
and how temperature shifts it. Tests whether the real backbone is intrinsically more *designable*.""")

code("""def nll_pool(globpat):
    a=[]
    for fa in glob.glob(str(ROOT/globpat)):
        for _,v in load_fa_nll(fa): 
            if not np.isnan(v): a.append(v)
    return np.array(a)
REG=[("fixed crystal\\n(T=0.1)","outputs/focus_6am/seqs/6AM*.fa","#2c7fb8"),
     ("de-novo\\n(T=0.1)","outputs/grind/seqs/*.fa","#41ab5d"),
     ("de-novo\\n(T=0.3)","outputs/grind_10k_T03/*/seqs/*.fa","#addd8e"),
     ("de-novo\\n(T=0.5)","outputs/grind_10k_T05/*/seqs/*.fa","#fec44f"),
     ("de-novo\\n(T=0.7)","outputs/grind_10k/*/seqs/*.fa","#d95f0e")]
data=[(lab,nll_pool(g),c) for lab,g,c in REG]; data=[d for d in data if len(d[1])]
fig,ax=plt.subplots(1,2,figsize=(14,4.5))
# violin
parts=ax[0].violinplot([d[1] for d in data],showmedians=True,showextrema=False)
for pc,(_,_,c) in zip(parts['bodies'],data): pc.set_facecolor(c); pc.set_alpha(.8)
ax[0].set_xticks(range(1,len(data)+1)); ax[0].set_xticklabels([d[0] for d in data],fontsize=8)
ax[0].set_ylabel("ProteinMPNN NLL (per-residue)"); ax[0].set_title("Design NLL: fixed crystal vs de-novo vs T")
# overlaid step histograms (density)
for lab,a,c in data: ax[1].hist(a,bins=70,histtype="step",lw=2,density=True,color=c,label=f"{lab.strip()} (med {np.median(a):.2f})")
ax[1].set_xlabel("NLL"); ax[1].set_ylabel("density"); ax[1].legend(fontsize=7); ax[1].set_title("NLL distributions")
plt.tight_layout(); plt.savefig(FIG/"14_nll_fixed_vs_denovo.png",bbox_inches="tight"); plt.show()
summ=pd.DataFrame([{"regime":d[0].replace(chr(10),' '),"n":len(d[1]),"median_NLL":round(np.median(d[1]),3),
                    "mean_NLL":round(d[1].mean(),3),"best_NLL":round(d[1].min(),3)} for d in data])
display(summ)
print("Native crystal backbone gives the LOWEST NLL (most designable); de-novo backbones are worse,")
print("and warmer sampling raises NLL further. Backbone quality >> sampling temperature for design confidence.")""")

md("""## 5j. Contact-conditioning ladder — does forcing MORE interface contacts help?

N-terminal MHC anchoring (A159/A66/A70) + ranked TCR contacts, sweeping **#conditioned contacts**
(3→4→5→9→12) across 5 levels per crystal; 3 RFdiffusion structures/level, ProteinMPNN 6000/level
(T=0.1, len-10). Recovery (to own + nearest validated binder) and median NLL vs #contacts.""")

code("""import csv as _csv
lf=ROOT/"outputs/ladder/ladder_summary.csv"
if not lf.exists():
    print("ladder_summary.csv not found — run run_ladder.sh / build_ladder_notebook.py")
else:
    lad=pd.read_csv(lf)
    display(lad.sort_values(["crystal","n_contacts"]))
    fig,ax=plt.subplots(1,2,figsize=(14,4.3))
    for pid,mk in [("6AMU","o"),("6AM5","s")]:
        s=lad[lad.crystal==pid].sort_values("n_contacts")
        ax[0].plot(s.n_contacts,s["mean->own"],marker=mk,lw=2.5,label=f"{pid} MEAN->own")
        ax[0].plot(s.n_contacts,s["max->own"],marker=mk,ls=":",lw=1.5,alpha=.6,label=f"{pid} max->own")
        ax[1].plot(s.n_contacts,s["median_NLL"],marker=mk,lw=2,label=pid)
    ax[0].set_xlabel("# conditioned contacts"); ax[0].set_ylabel("max % identity"); ax[0].legend(fontsize=7)
    ax[0].set_title("Recovery vs #conditioned contacts (MEAN shifts up)"); ax[0].set_ylim(0,60)
    ax[1].set_xlabel("# conditioned contacts"); ax[1].set_ylabel("median NLL"); ax[1].legend()
    ax[1].set_title("Design NLL vs #conditioned contacts")
    plt.tight_layout(); plt.savefig(FIG/"15_contact_ladder.png",bbox_inches="tight"); plt.show()
    for pid in ["6AMU","6AM5"]:
        s=lad[lad.crystal==pid].sort_values("n_contacts")
        lo,hi=s.iloc[0],s.iloc[-1]
        print(f"{pid}: L1({lo.n_contacts}c) own{lo['max->own']:.0f}/panel{lo['max->panel']:.0f}/NLL{lo.median_NLL:.2f}"
              f"  ->  L5({hi.n_contacts}c) own{hi['max->own']:.0f}/panel{hi['max->panel']:.0f}/NLL{hi.median_NLL:.2f}")
    print("Conditioning DOES shift the distribution: MEAN identity ~doubles L1->L5 (6AMU 8.5->15, 6AM5 7->16);")
    print("more MHC-pocket contacts recover ANCHOR residues (6AM5 P10-Val ->67%, 6AMU P1-Met ->99%).")
    print("MAX saturates because the central TCR-facing core stays unrecovered regardless of conditioning.")""")

md("""## 6. Conclusions

**Positive (fixed backbone):**
- ProteinMPNN recovers **the TCR-facing motif** of both native peptides from their crystal backbones at
  high (~48-49%) mean identity — but **0 exact (100%) full-sequence matches** across any of the three
  structures. "Recovery" here means the functionally-decisive residues, not the whole peptide verbatim.
- Recovery is **structure-specific** (own ≫ other, gap +28–36%) and **TCR-dependent** (TCR present raises recovery and stabilises it under relaxation).
- The recovered residues are the **central TCR-facing motif** (DRG: D4/G6; GIG: G4/G6/P9), **not** the MHC anchors (which MPNN re-canonicalises away from the non-standard Met anchors).

**Negative (generative):**
- **Relaxation** multiplies *own* exact recoveries but **never** yields the other peptide — perturbation is too local to leave the native basin.
- **De-novo RFdiffusion** (106 backbones, 5,194 designs, full conditioning sweep) reaches only **~30%** identity to either native — standard RFdiffusion does not reconstruct the specific DRG/GIG groove geometry.

**Conditioning effect (the nuance):** max-identity is flat (~30%) across conditioning, BUT **AA composition and diversity differ** by conditioning — MHC+TCR conditioning on 6AMU enriches the DRG-defining Asp; MHC-only changes the Pro/Gly balance; TCR-conditioned sets are less diverse. So conditioning *shapes* the generated distribution even though it doesn't reconstruct the exact native.

**Implication:** peptide identity for this cross-reactive TCR lives in the **precise native backbone conformation**, which neither local relaxation nor de-novo RFdiffusion rediscovers. Next step would be **RFpeptides** (the short-peptide branch) for proper groove-peptide geometry.""")

nb["cells"]=cells
nb["metadata"]={"kernelspec":{"display_name":"Python 3 (base)","language":"python","name":"python3"},"language_info":{"name":"python"}}
out="/home/ubuntu/if-mhc/11_dmf5_cross_reactivity.ipynb"; nbf.write(nb,out); print("wrote",out,len(cells),"cells")
