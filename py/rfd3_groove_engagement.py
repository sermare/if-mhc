#!/usr/bin/env python3
"""For every RFD3 design: superpose its MHC onto the native MHC, move the designed binder
into the native frame, and quantify whether the peptide engages the groove THROUGHOUT
(N-pocket .. F-pocket) or stayed outside.

Per design we report:
  cen   : binder-centroid -> native-peptide-centroid distance (in native frame)  [<~8 = in groove]
  nNP   : # native N-pocket residues contacted (<4.5 A heavy)   (anchors P1-P3 end)
  nFP   : # native F-pocket residues contacted                   (anchors P8-P10 end)
  nTot  : # of the full groove-floor set contacted
  span  : 'YES' if both ends (N-pocket AND F-pocket) are contacted -> threaded through
  D / E : binder min-dist to TCRa (D) and TCRb (E)               [<~5 = engaged]
"""
import glob, os, warnings, numpy as np
warnings.filterwarnings("ignore")
from Bio.PDB import PDBParser
ROOT = "/home/ubuntu/if-mhc"; P = PDBParser(QUIET=True)
AA = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H',
      'ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W',
      'TYR':'Y','VAL':'V','MSE':'M'}
NPOCK = [5,7,59,63,66,70,159,163,167,171]
FPOCK = [77,80,84,95,116,123,143,146,147]
GROOVE = sorted(set(NPOCK+FPOCK+[9,97,99,114,152,155,156]))

def chain_ca(ch):
    rs=[r for r in ch if r.id[0]==' ' and 'CA' in r]
    return ''.join(AA.get(r.resname,'x') for r in rs), np.array([r['CA'].coord for r in rs])
def heavy(res): return np.array([a.coord for a in res if a.element!='H'])
def kabsch(P_,Q_):
    Pc=P_.mean(0);Qc=Q_.mean(0);H=(P_-Pc).T@(Q_-Qc)
    V,S,Wt=np.linalg.svd(H);d=np.sign(np.linalg.det(V@Wt));R=V@np.diag([1,1,d])@Wt
    return R,Qc-Pc@R
def best_offset(s_out,s_nat):
    """offset k: out position i maps to native position i+k. pick k maximizing matches."""
    best=(-1,0)
    for k in range(-3,8):
        mt=sum(1 for i,c in enumerate(s_out) if 0<=i+k<len(s_nat) and c==s_nat[i+k])
        if mt>best[0]: best=(mt,k)
    return best[1]

def native(pid):
    m=P.get_structure('n',f"{ROOT}/inputs/focus_6am/{pid}.pdb")[0]
    sA,caA=chain_ca(m['A'])
    Ares={r.id[1]:heavy(r) for r in m['A'] if r.id[0]==' '}
    pepH=np.vstack([heavy(r) for r in m['C'] if r.id[0]==' '])
    pepCen=np.vstack([r['CA'].coord for r in m['C'] if 'CA' in r]).mean(0)
    return sA,caA,Ares,pepCen

def analyze(pid, job):
    sA,caA,Ares,pepCen = native(pid)
    out=[]
    for f in sorted(glob.glob(f"{ROOT}/outputs/tamarind/results/{job}/{job}-*/config_*model_0.pdb")):
        m=P.get_structure('x',f)[0]
        chains={ch.id:ch for ch in m}
        seqs={cid:chain_ca(ch) for cid,ch in chains.items()}
        # binder = a <=12 residue chain
        bid=[cid for cid,(s,c) in seqs.items() if 0<len(s)<=12]
        # MHC = chain matching native MHC start
        mid=[cid for cid,(s,c) in seqs.items() if len(s)>150 and ('SHSMRYFF' in s[:12])]
        if not bid or not mid: continue
        bid=bid[0]; mid=mid[0]
        s_out,ca_out=seqs[mid]
        k=best_offset(s_out,sA)
        pairs=[(i,i+k) for i in range(len(s_out)) if 0<=i+k<len(sA) and s_out[i]==sA[i+k]]
        Po=np.array([ca_out[i] for i,_ in pairs]); Pn=np.array([caA[j] for _,j in pairs])
        R,t=kabsch(Po,Pn)
        bH=heavy_chain(chains[bid])@R+t
        bCA=seqs[bid][1]@R+t
        cen=float(np.linalg.norm(bCA.mean(0)-pepCen))
        def ncontact(idlist):
            n=0
            for rid in idlist:
                if rid in Ares and Ares[rid].size:
                    if np.min(np.linalg.norm(Ares[rid][:,None]-bH[None],axis=2))<4.5: n+=1
            return n
        nNP=ncontact(NPOCK); nFP=ncontact(FPOCK); nTot=ncontact(GROOVE)
        # TCR chains in native frame: identify output TCRa/b by sequence, transform, min dist
        def tcr_min(seqpref):
            cid=[c for c,(s,_) in seqs.items() if s[:14].startswith(seqpref[:14]) or seqpref[:14] in s[:20]]
            if not cid: return 99.9
            cc=seqs[cid[0]][1]@R+t
            return float(np.min(np.linalg.norm(cc[:,None]-bCA[None],axis=2)))
        dmin=tcr_min('EVEQNSGPLSVP') if pid=='6AMU' else tcr_min('KEVEQNSGPLSV')
        emin=tcr_min('IAGITQAPTSQ')
        out.append((os.path.basename(f).split('config_')[1].split('_0_model')[0],cen,nNP,nFP,nTot,dmin,emin))
    return out

def heavy_chain(ch):
    return np.vstack([heavy(r) for r in ch if r.id[0]==' '])

JOBS=[("6AMU","ifmhc_rfd3_native_6AMU"),("6AM5","ifmhc_rfd3_native_6AM5"),
      ("6AMU","ifmhc_rfd3_L4_6AMU"),("6AM5","ifmhc_rfd3_L4_6AM5"),
      ("6AMU","ifmhc_rfd3_groove_6AMU"),("6AM5","ifmhc_rfd3_groove_6AM5"),
      ("6AMU","ifmhc_rfd3_mdc_6AMU"),("6AM5","ifmhc_rfd3_mdc_6AM5"),
      ("6AMU","ifmhc_rfd3_mdc2_6AMU"),("6AM5","ifmhc_rfd3_mdc2_6AM5"),
      ("6AMU","ifmhc_rfd3_anchor_6AMU"),("6AM5","ifmhc_rfd3_anchor_6AM5")]
print(f"{'design':38s} {'cen':>5} {'nNP':>3} {'nFP':>3} {'nTot':>4} {'span':>4} {'Dmin':>5} {'Emin':>5}")
print("-"*78)
for pid,job in JOBS:
    if not os.path.isdir(f"{ROOT}/outputs/tamarind/results/{job}"): continue
    for name,cen,nNP,nFP,nTot,d,e in analyze(pid,job):
        span="YES" if (nNP>0 and nFP>0) else "no"
        print(f"{name:38s} {cen:5.1f} {nNP:3d} {nFP:3d} {nTot:4d} {span:>4} {d:5.1f} {e:5.1f}")
