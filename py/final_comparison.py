#!/usr/bin/env python3
"""Final comparison across NATIVE crystals, RFdiffusion-v1 backbones, and RFD3 designs.

Two metrics, because contact-counting alone is confounded (v1 peptides are poly-G, so
side-chain reach is missing and contacts undercount):
  PLACEMENT (primary, unconfounded): superpose each structure's MHC onto native MHC
    (iterative outlier-rejected Kabsch), then peptide backbone vs NATIVE peptide envelope:
       frac<4A = fraction of peptide CA within 4 A of a native peptide CA  (1.0 = fully seated)
       maxOut  = worst protruding CA (A)                                   (high = sticking out)
  PANEL CONTACTS (what was asked): of a fixed 29-residue native interface panel
    (MHC + TCRa + TCRb), how many the peptide BACKBONE contacts (<=4.5 A to residue heavy).
    NOTE: undercounts poly-G v1 backbones; read alongside placement.
"""
import glob, os, warnings, csv, numpy as np
warnings.filterwarnings("ignore")
from Bio.PDB import PDBParser
ROOT="/home/ubuntu/if-mhc"; P=PDBParser(QUIET=True)
AA={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
    'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V','MSE':'M'}
BB={'N','CA','C','O'}; CUT=4.5
def rl(ch): return [r for r in ch if r.id[0]==' ' and 'CA' in r]
def sq(rs): return ''.join(AA.get(r.resname,'x') for r in rs)
def ca(rs): return np.array([r['CA'].coord for r in rs])
def heavy(r): return np.array([a.coord for a in r if a.element!='H'])
def bb(rs): return np.array([a.coord for r in rs for a in r if a.get_name() in BB])
def ah(rs): return np.vstack([heavy(r) for r in rs])   # peptide ALL-heavy (side chains incl.)
def boff(so,sn):
    best=(-1,0)
    for k in range(-4,12):
        mt=sum(1 for i,c in enumerate(so) if 0<=i+k<len(sn) and c==sn[i+k])
        if mt>best[0]: best=(mt,k)
    return best[1]
def kab(Pp,Qq):
    Pc=Pp.mean(0);Qc=Qq.mean(0);H=(Pp-Pc).T@(Qq-Qc);V,S,Wt=np.linalg.svd(H);d=np.sign(np.linalg.det(V@Wt));R=V@np.diag([1,1,d])@Wt;return R,Qc-Pc@R
def robust_fit(Po,Pn):
    R,t=kab(Po,Pn)
    for _ in range(3):
        res=np.linalg.norm(Po@R+t-Pn,axis=1); keep=res<3.0
        if keep.sum()<30: break
        R,t=kab(Po[keep],Pn[keep])
    res=np.linalg.norm(Po@R+t-Pn,axis=1); keep=res<3.0
    return R,t,float(np.sqrt((res[keep]**2).mean()))

NAT={}
for pid in ['6AMU','6AM5']:
    m=P.get_structure('n',f"{ROOT}/inputs/focus_6am/{pid}.pdb")[0]
    NAT[pid]=dict(Aseq=sq(rl(m['A'])),Aca=ca(rl(m['A'])),
                  chains={c:rl(m[c]) for c in ['A','D','E']},
                  ids={c:[r.id[1] for r in rl(m[c])] for c in ['A','D','E']},
                  pepC=rl(m['C']),pepCA=ca(rl(m['C'])))

# ---- panel from native (peptide backbone -> target heavy <=4.5) ----
def native_targets(pid):
    out=[]
    for lab,c in [('A','A'),('D','D'),('E','E')]:
        ids=NAT[pid]['ids'][c]
        for i,r in enumerate(NAT[pid]['chains'][c]): out.append((lab,ids[i],r))
    return out
def contacts(pep_bb,targets):
    hit=set()
    for lab,nid,r in targets:
        hv=heavy(r)
        if hv.size and np.min(np.linalg.norm(hv[:,None]-pep_bb[None],axis=2))<CUT: hit.add((lab,nid))
    return hit
panel=set()
for pid in ['6AMU','6AM5']:
    panel|=contacts(ah(NAT[pid]["pepC"]),native_targets(pid))
panel=sorted(panel,key=lambda x:({'A':0,'D':1,'E':2}[x[0]],x[1]))
def fmt(l,i): return f"{l}{i}"
print(f"=== 30-RESIDUE INTERFACE PANEL ({len(panel)} native contacts) ===")
for lab,name in [('A','MHC (chain A)'),('D','TCRa (chain D)'),('E','TCRb (chain E)')]:
    rr=[fmt(l,i) for l,i in panel if l==lab]
    print(f"  {name}: {' '.join(rr)}")
panel_set=set(panel)

def map_to_native(rs,pid,c,lab):
    ref=NAT[pid]['ids'][c]; refseq=sq(NAT[pid]['chains'][c]); k=boff(sq(rs),refseq)
    return [(lab,ref[i+k],r) for i,r in enumerate(rs) if 0<=i+k<len(ref)]

def placement(pid,mhc_rs,pep_rs):
    so=sq(mhc_rs); sA=NAT[pid]['Aseq']; k=boff(so,sA)
    pr=[(i,i+k) for i in range(len(so)) if 0<=i+k<len(sA) and so[i]==sA[i+k]]
    if len(pr)<50: return None
    Po=np.array([ca(mhc_rs)[i] for i,_ in pr]); Pn=np.array([NAT[pid]['Aca'][j] for _,j in pr])
    R,t,fit=robust_fit(Po,Pn)
    b=ca(pep_rs)@R+t; nd=np.min(np.linalg.norm(b[:,None]-NAT[pid]['pepCA'][None],axis=2),axis=1)
    return fit,float(nd.max()),float(nd.mean()),float(np.mean(nd<4.0))

rows=[]
def add(group,pid,name,mhc_rs,pep_rs,targets):
    pl=placement(pid,mhc_rs,pep_rs)
    if pl is None: return
    fit,mx,mn,fr=pl
    hit=contacts(ah(pep_rs),targets)&panel_set
    nA=sum(l=='A' for l,_ in hit); nD=sum(l=='D' for l,_ in hit); nE=sum(l=='E' for l,_ in hit)
    rows.append([group,pid,name,round(fr,2),round(mx,1),len(hit),nA,nD,nE,round(fit,2)])

# natives
for pid in ['6AMU','6AM5']:
    add('NATIVE',pid,f'{pid}_crystal',NAT[pid]['chains']['A'],NAT[pid]['pepC'],native_targets(pid))
# v1
v1=[]
for src in ['ladder','grind','promising']:
    v1+=[f for f in sorted(glob.glob(f'{ROOT}/outputs/{src}/pdb/*.pdb')) if '_split' not in f]
for f in v1[::max(1,len(v1)//14)][:14]:
    pid='6AMU' if os.path.basename(f).startswith('6AMU') else '6AM5'
    m=P.get_structure('x',f)[0]; chs={c.id:rl(c) for c in m}
    pep=[c for c,rs in chs.items() if 0<len(rs)<=12]; big=[c for c,rs in chs.items() if len(rs)>150]
    if not pep or not big: continue
    B=chs[big[0]]; sB=sq(B)
    tg=[]
    for lab,mot,c in [('A','SHSMRYFF','A'),('D','KEVEQNSGPL' if pid=='6AM5' else 'EVEQNSGPL','D'),('E','IAGITQAPT','E')]:
        st=sB.find(mot[:8])
        if st>=0: tg+=map_to_native(B[st:st+len(NAT[pid]['ids'][c])+8],pid,c,lab)
    add('RFD-v1',pid,os.path.basename(f)[:-4],B,chs[pep[0]],tg)
# RFD3
for job in sorted(glob.glob(f"{ROOT}/outputs/tamarind/results/ifmhc_rfd3_*")):
    jn=os.path.basename(job); pid='6AMU' if '6AMU' in jn else '6AM5'
    strat=jn.replace('ifmhc_rfd3_','').replace(f'_{pid}','')
    for f in sorted(glob.glob(f"{job}/{jn}-*/config_*model_0.pdb")):
        m=P.get_structure('x',f)[0]; chs={c.id:rl(c) for c in m}
        pep=[c for c,rs in chs.items() if 0<len(rs)<=12]
        mhc=[c for c,rs in chs.items() if len(rs)>150 and 'SHSMRYFF' in sq(rs)[:12]]
        if not pep or not mhc: continue
        tg=[]
        for lab,mot,c in [('A','SHSMRYFF','A'),('D','EVEQNSGPL' if pid=='6AMU' else 'KEVEQNSGPL','D'),('E','IAGITQAPT','E')]:
            cid=[cc for cc,rs in chs.items() if mot[:8] in sq(rs)[:14]]
            if cid: tg+=map_to_native(chs[cid[0]],pid,c,lab)
        nm=os.path.basename(f).split('config_')[1].split('_0_model')[0]
        add(f'RFD3-{strat}',pid,nm,chs[mhc[0]],chs[pep[0]],tg)

rows.sort(key=lambda r:(0 if r[0]=='NATIVE' else 1 if r[0]=='RFD-v1' else 2, r[2]))
print(f"\n{'group':13s} {'sys':5s} {'structure':34s} {'seated%':>7} {'maxOut':>6} {'panel':>5} {'MHC':>4}{'TCRa':>5}{'TCRb':>5} {'fit':>5}")
print("-"*100)
for g,pid,nm,fr,mx,np_,nA,nD,nE,fit in rows:
    flag=' <OUT' if (fr<0.7 or mx>10) else ''
    print(f"{g:13s} {pid:5s} {nm:34s} {fr*100:6.0f}% {mx:6.1f} {np_:5d} {nA:4d}{nD:5d}{nE:5d} {fit:5.2f}{flag}")
with open(f"{ROOT}/outputs/struct_ood/final_comparison.csv","w",newline="") as o:
    w=csv.writer(o); w.writerow(['group','sys','structure','seated_frac','maxOut_A','panel_contacts','MHC','TCRa','TCRb','mhc_fit_A'])
    w.writerows(rows)
print(f"\nwrote outputs/struct_ood/final_comparison.csv")
