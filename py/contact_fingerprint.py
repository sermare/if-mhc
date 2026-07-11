#!/usr/bin/env python3
"""Unified contact fingerprint: how many groove/TCR residues does each peptide BACKBONE
contact? Compares native crystals, RFdiffusion-v1 backbones (first experiments, poly-G),
and RFD3 designs on ONE metric:
  peptide backbone atom (N,CA,C,O)  <=4.5 A  target residue (any heavy atom).
Contacts are computed in each structure's OWN frame (no superposition); target residues
are mapped to native numbering by sequence offset so they are comparable. The 30-residue
PANEL is the union of native (6AMU+6AM5) contacts across MHC(A)+TCRa(D)+TCRb(E)."""
import glob, os, warnings, numpy as np
warnings.filterwarnings("ignore")
from Bio.PDB import PDBParser
ROOT="/home/ubuntu/if-mhc"; P=PDBParser(QUIET=True)
AA={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
    'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V','MSE':'M'}
BB={'N','CA','C','O'}
CUT=4.5

def res_list(ch):
    return [r for r in ch if r.id[0]==' ' and 'CA' in r]
def seq_of(rs): return ''.join(AA.get(r.resname,'x') for r in rs)
def heavy(r): return np.array([a.coord for a in r if a.element!='H'])
def bb_atoms(rs):
    pts=[]
    for r in rs:
        for a in r:
            if a.get_name() in BB: pts.append(a.coord)
    return np.array(pts)
def offset(so,sn):
    best=(-1,0)
    for k in range(-4,12):
        mt=sum(1 for i,c in enumerate(so) if 0<=i+k<len(sn) and c==sn[i+k])
        if mt>best[0]: best=(mt,k)
    return best[1]

# --- native reference chains (seq + resids) ---
NAT={}
for pid in ['6AMU','6AM5']:
    m=P.get_structure('n',f"{ROOT}/inputs/focus_6am/{pid}.pdb")[0]
    NAT[pid]={c:(seq_of(res_list(m[c])),[r.id[1] for r in res_list(m[c])],res_list(m[c])) for c in ['A','D','E'] if c in m}
    NAT[pid]['C']=res_list(m['C'])

def contacts(pep_bb, target_residues_with_natid):
    """return set of (chainlabel,natid) contacted by peptide backbone."""
    hit=set()
    for lab,natid,r in target_residues_with_natid:
        hv=heavy(r)
        if hv.size and np.min(np.linalg.norm(hv[:,None]-pep_bb[None],axis=2))<CUT:
            hit.add((lab,natid))
    return hit

def map_chain(rs, ref_seq, ref_ids, label):
    """map a chain's residues to native numbering via offset; return [(label,natid,res)]."""
    s=seq_of(rs); k=offset(s,ref_seq); out=[]
    for i,r in enumerate(rs):
        j=i+k
        if 0<=j<len(ref_ids): out.append((label,ref_ids[j],r))
    return out

def native_target(pid):
    out=[]
    for lab,nc in [('A','A'),('Da','D'),('Eb','E')]:
        if nc in NAT[pid]:
            seq,ids,rs=NAT[pid][nc]
            out+=[(lab,ids[i],rs[i]) for i in range(len(rs))]
    return out

# ---- build PANEL from native crystals ----
nat_hits={}
for pid in ['6AMU','6AM5']:
    pep_bb=bb_atoms(NAT[pid]['C'])
    nat_hits[pid]=contacts(pep_bb, native_target(pid))
panel=sorted(nat_hits['6AMU']|nat_hits['6AM5'], key=lambda x:(x[0],x[1]))
# label nicely
def fmt(lab,natid): return {'A':'A','Da':'D','Eb':'E'}[lab]+str(natid)
print(f"PANEL = {len(panel)} native contact residues (MHC A + TCRa D + TCRb E):")
for lab in ['A','Da','Eb']:
    rr=[fmt(l,i) for l,i in panel if l==lab]
    print(f"  {{'A':'MHC','Da':'TCRa','Eb':'TCRb'}}[{lab!r}] ({len(rr)}): {' '.join(rr)}")

panel_set=set(panel)

def score(pep_bb, target):
    h=contacts(pep_bb,target)
    p=h & panel_set
    nA=len([1 for l,i in p if l=='A']); nD=len([1 for l,i in p if l=='Da']); nE=len([1 for l,i in p if l=='Eb'])
    return len(h), len(p), nA, nD, nE, p

rows=[]
# natives
for pid in ['6AMU','6AM5']:
    tot,np_,nA,nD,nE,_=score(bb_atoms(NAT[pid]['C']), native_target(pid))
    rows.append(('NATIVE',pid,f'{pid}_crystal',tot,np_,nA,nD,nE))

# RFdiffusion v1 backbones (chain A = peptide; chain B = concatenated target)
def v1_target(pid, chB_rs):
    s=seq_of(chB_rs); out=[]
    segs=[('A','SHSMRYFF'),('Da','KEVEQNSGPL' if pid=='6AM5' else 'EVEQNSGPL'),('Eb','IAGITQAPT')]
    for lab,motif in segs:
        st=s.find(motif[:8])
        if st<0: continue
        # extend segment until next motif or end; map against native chain
        ncmap={'A':'A','Da':'D','Eb':'E'}; refseq,refids,_=NAT[pid][ncmap[lab]]
        sub=chB_rs[st:st+len(refseq)+5]
        out+=map_chain(sub, refseq, refids, lab)
    return out
v1=[]
for src,tag in [('ladder','RFD-v1'),('grind','RFD-v1'),('promising','RFD-v1')]:
    for f in sorted(glob.glob(f"{ROOT}/outputs/{src}/pdb/*.pdb")):
        if '_split' in f: continue
        v1.append((tag,f))
import random
# representative sample spanning files (deterministic: every Nth)
sample=v1[::max(1,len(v1)//12)][:12]
for tag,f in sample:
    pid='6AMU' if '/6AMU' in f or os.path.basename(f).startswith('6AMU') else '6AM5'
    m=P.get_structure('x',f)[0]
    chs={c.id:res_list(c) for c in m}
    pep=[c for c,rs in chs.items() if len(rs)<=12]
    big=[c for c,rs in chs.items() if len(rs)>150]
    if not pep or not big: continue
    pep_bb=bb_atoms(chs[pep[0]])
    tot,np_,nA,nD,nE,_=score(pep_bb, v1_target(pid, chs[big[0]]))
    rows.append((tag,pid,os.path.basename(f)[:-4],tot,np_,nA,nD,nE))

# RFD3 designs (binder<=12; MHC=SHSMRYFF chain; TCRa/b by motif)
def rfd3_target(pid, m):
    chs={c.id:res_list(c) for c in m}
    out=[]
    for lab,motif,nc in [('A','SHSMRYFF','A'),('Da','EVEQNSGPL' if pid=='6AMU' else 'KEVEQNSGPL','D'),('Eb','IAGITQAPT','E')]:
        cid=[c for c,rs in chs.items() if motif[:8] in seq_of(rs)[:14]]
        if not cid: continue
        refseq,refids,_=NAT[pid][nc]
        out+=map_chain(chs[cid[0]], refseq, refids, lab)
    return out
for job in sorted(glob.glob(f"{ROOT}/outputs/tamarind/results/ifmhc_rfd3_*")):
    jn=os.path.basename(job); pid='6AMU' if '6AMU' in jn else '6AM5'
    strat=jn.replace('ifmhc_rfd3_','').replace(f'_{pid}','')
    for f in sorted(glob.glob(f"{job}/{jn}-*/config_*model_0.pdb")):
        m=P.get_structure('x',f)[0]; chs={c.id:res_list(c) for c in m}
        pep=[c for c,rs in chs.items() if 0<len(rs)<=12]
        if not pep: continue
        pep_bb=bb_atoms(chs[pep[0]])
        tot,np_,nA,nD,nE,_=score(pep_bb, rfd3_target(pid,m))
        nm=os.path.basename(f).split('config_')[1].split('_0_model')[0]
        rows.append((f'RFD3-{strat}',pid,nm,tot,np_,nA,nD,nE))

print(f"\n{'group':14s} {'sys':5s} {'structure':34s} {'totContacts':>11} {'panel/30':>8} {'MHC':>4} {'TCRa':>5} {'TCRb':>5}")
print("-"*95)
for g,pid,nm,tot,np_,nA,nD,nE in rows:
    print(f"{g:14s} {pid:5s} {nm:34s} {tot:11d} {np_:8d} {nA:4d} {nD:5d} {nE:5d}")

import csv
with open(f"{ROOT}/outputs/struct_ood/contact_fingerprint.csv","w",newline="") as o:
    w=csv.writer(o); w.writerow(['group','sys','structure','total_contacts','panel_of_30','MHC','TCRa','TCRb'])
    w.writerows(rows)
print(f"\nwrote outputs/struct_ood/contact_fingerprint.csv  (panel size {len(panel)})")
