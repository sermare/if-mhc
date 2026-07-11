#!/usr/bin/env python3
"""Single-load core data module for the pMHC peptide-conformation notebook.

Loads EVERY peptide backbone once (natives, RFdiffusion-v1, tool-relaxed, RFD3),
puts them all in ONE common groove frame (superpose each structure's MHC alpha1/alpha2
onto the 6AMU native MHC -- valid because both crystals share a near-identical HLA-A2,
alpha1alpha2 RMSD ~0.07 A), and returns:

  DF     : one row per 10-mer peptide  (pid, cond, nice, group, toDRG, toGIG, basin,
           seated, fit, Rg, e2e, maxbulge, detail, file)
  COORDS : (N,10,3) groove-frame peptide CA, row-aligned to DF.index
  GIGref : DRG and GIG native peptides in the common frame (for plotting refs)
  TRAJ   : MD trajectory dual-RMSD frames (from outputs/tamarind/md_traj.csv)

Register-preserving (not best-fit): the dual-RMSD reflects WHERE in the groove the
peptide sits, not just its shape. Cached to outputs/struct_ood/core_cache/.
"""
import glob, os, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from Bio.PDB import PDBParser

ROOT = "/home/ubuntu/if-mhc"
CACHE = f"{ROOT}/outputs/struct_ood/core_cache"
_pp = PDBParser(QUIET=True)
AA = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
      'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V','MSE':'M'}

# ── conditioning label machinery (canonical) ────────────────────────────────
COND_DESC = {
    "MHCpocket(7)":"grind: A9,A63,A66,A77,A80,A116,A143 (HLA-A2 B+F pocket)",
    "TCR2(2)":"grind: E100/E97,D30 (top-2 TCR contacts)",
    "MHC+TCR(9)":"grind: MHCpocket + TCR2",
    "Nterm(3)":"ladder L1: A159,A66,A70 (N-term MHC anchors)",
    "Nterm+TCR1(4)":"ladder L2: N-term + 1 TCR",
    "Nterm+TCR2(5)":"L3: N-term + 2 TCR",
    "Nterm+pkt+TCR3(9)":"L4: N-term + A9,A77,A80 + 3 TCR",
    "Nterm+fullpkt+TCR4(12)":"L5: N-term + full pocket + 4 TCR",
    "RFD3:native(13)":"rfd3: A159,66,70,9,77,80,116,143 + E100,99/E97,96 + D30,93/D30",
    "RFD3:groove(floor)":"rfd3: full beta-sheet floor A5..167, NO TCR",
    "RFD3:L4(N+pkt+3TCR)":"rfd3: A159,66,70,9,77,80 + 3 TCR",
    "RFD3:MDfloor+TCR":"rfd3 (mdc/mdc2): MD-derived 12-res floor + top-3 MD contacts each TCR",
    "RFD3:bracket+TCR":"rfd3 (anchor): A159,66,70 + A146 + E100,99/97,96 + D30",
    "RFD3:scan(brkt+TCRwin)":"rfd3 scan: bracket + D{28,30}|{30,91,93} x E{95,96}|{97,98}|{99,100}",
    "RFD3:xcond(swapTCR)":"rfd3 cross-cond: bracket + OTHER crystal's TCR pattern",
}
TOK2COND = {"mhc":"MHCpocket(7)","tcr2":"TCR2(2)","mhc_tcr2":"MHC+TCR(9)","L1_nterm":"Nterm(3)",
    "L2_nterm_t1":"Nterm+TCR1(4)","L3_nterm_t2":"Nterm+TCR2(5)","L4_expanded":"Nterm+pkt+TCR3(9)",
    "L5_max":"Nterm+fullpkt+TCR4(12)","rfd3_native":"RFD3:native(13)","rfd3_groove":"RFD3:groove(floor)",
    "rfd3_L4":"RFD3:L4(N+pkt+3TCR)","rfd3_mdc2":"RFD3:MDfloor+TCR","rfd3_mdc":"RFD3:MDfloor+TCR",
    "rfd3_anchor":"RFD3:bracket+TCR","rfd3_scan":"RFD3:scan(brkt+TCRwin)","rfd3_xcond":"RFD3:xcond(swapTCR)"}
_TOK_ORDER = ["rfd3_native","rfd3_groove","rfd3_anchor","rfd3_xcond","rfd3_scan","rfd3_mdc2","rfd3_mdc",
    "rfd3_L4","mhc_tcr2","L1_nterm","L2_nterm_t1","L3_nterm_t2","L4_expanded","L5_max","tcr2","mhc"]
def cond_label(fname):
    b = "_" + os.path.basename(fname)
    for tok in _TOK_ORDER:
        if f"_{tok}_" in b: return TOK2COND[tok]
    return None
NICE = {"crystal":"crystal native","relaxed":"MD-relaxed native","TCR2(2)":"grind: TCRx2",
    "MHCpocket(7)":"grind: MHC-pocket","MHC+TCR(9)":"grind: MHC+TCR","Nterm(3)":"L1: N-term",
    "Nterm+TCR1(4)":"L2: N-term+1TCR","Nterm+TCR2(5)":"L3: N-term+2TCR","Nterm+pkt+TCR3(9)":"L4: N-term+pkt+3TCR",
    "Nterm+fullpkt+TCR4(12)":"L5: N-term+fullpkt+4TCR","openmm-relax":"openmm-relax","rosetta-relax":"rosetta-relax",
    "openmm-MD-10ns":"openmm-MD 10ns","openmm-MD-native":"openmm-MD native","RFD3:native(13)":"RFD3: native",
    "RFD3:groove(floor)":"RFD3: groove-floor","RFD3:L4(N+pkt+3TCR)":"RFD3: L4","RFD3:MDfloor+TCR":"RFD3: MD-floor+TCR",
    "RFD3:bracket+TCR":"RFD3: bracket+TCR","RFD3:scan(brkt+TCRwin)":"RFD3: TCR-window scan","RFD3:xcond(swapTCR)":"RFD3: cross-cond"}
def nice(s): return NICE.get(s, s)

# ── break the RFD3 TCR-window scan + cross-cond into their actual sub-conditionings ──
_DA = {"cdr1":("CDR1a","D{28,30}"), "cdr3":("CDR3a","D{30,91,93}")}
_EB = {"lo":("Blo","E{95,96}"), "mid":("Bmid","E{97,98}"), "hi":("Bhi","E{99,100}")}
_SCAN_LABELS=[]
for _dk,(_dn,_dr) in _DA.items():
    for _ek,(_en,_er) in _EB.items():
        _k=f"RFD3:scan {_dn}.{_en}"; _SCAN_LABELS.append(_k)
        COND_DESC[_k]=f"rfd3 scan: bracket(159,66,70,146) + {_dr} + {_er}"
        NICE[_k]=f"RFD3 scan {_dn}.{_en}"
_XCOND_LABELS=[]
for _pid,_other in [("6AMU","6AM5"),("6AM5","6AMU")]:
    _k=f"RFD3:xcond {_pid}<-{_other}"; _XCOND_LABELS.append(_k)
    COND_DESC[_k]=f"rfd3 cross-cond: {_pid} MHC + {_other} TCR contact pattern"
    NICE[_k]=f"RFD3 xcond {_pid}<-{_other}"

def refine_cond(cond, detail):
    """split the collapsed scan/xcond labels into their per-window sub-conditioning."""
    if cond=="RFD3:scan(brkt+TCRwin)":
        p=detail.split("_")                       # scan_<pid>_<da>_<eb>
        if len(p)>=4 and p[2] in _DA and p[3] in _EB:
            return f"RFD3:scan {_DA[p[2]][0]}.{_EB[p[3]][0]}"
    if cond=="RFD3:xcond(swapTCR)":
        p=detail.split("_")                       # xcond_<pid>_as<other>
        if len(p)>=3:
            return f"RFD3:xcond {p[1]}<-{p[2].replace('as','')}"
    return cond

COND_ORDER = ["crystal","relaxed","TCR2(2)","Nterm(3)","Nterm+TCR1(4)","Nterm+TCR2(5)","MHCpocket(7)","MHC+TCR(9)",
    "Nterm+pkt+TCR3(9)","Nterm+fullpkt+TCR4(12)","openmm-relax","rosetta-relax","openmm-MD-10ns","openmm-MD-native",
    "RFD3:groove(floor)","RFD3:native(13)","RFD3:L4(N+pkt+3TCR)","RFD3:bracket+TCR","RFD3:MDfloor+TCR",
    *_SCAN_LABELS, *_XCOND_LABELS]
def group_of(cond):
    if cond in ("crystal","relaxed"): return "native"
    if cond.startswith("RFD3"): return "rfd3"
    if cond in ("openmm-relax","rosetta-relax","openmm-MD-10ns","openmm-MD-native"): return "tool"
    return "v1"

# ── geometry helpers ────────────────────────────────────────────────────────
def _rl(ch): return [r for r in ch if r.id[0]==' ' and 'CA' in r]
def _sq(rs): return ''.join(AA.get(r.resname,'x') for r in rs)
def _ca(rs): return np.array([r['CA'].coord for r in rs])
def _RT(P,Q):
    Pc=P.mean(0);Qc=Q.mean(0);V,S,Wt=np.linalg.svd((P-Pc).T@(Q-Qc));d=np.sign(np.linalg.det(V@Wt))
    R=V@np.diag([1,1,d])@Wt; return R,Qc-Pc@R
def _robust(Po,Pn):
    R,t=_RT(Po,Pn)
    for _ in range(3):
        res=np.linalg.norm(Po@R+t-Pn,axis=1); keep=res<3.0
        if keep.sum()<30: break
        R,t=_RT(Po[keep],Pn[keep])
    res=np.linalg.norm(Po@R+t-Pn,axis=1); keep=res<3.0
    return R,t,float(np.sqrt((res[keep]**2).mean()))
def _rmsd(a,b): return float(np.sqrt(((a-b)**2).sum()/len(a)))
def _off(so,sn):
    best=(-1,0)
    for k in range(-4,14):
        m=sum(1 for i,c in enumerate(so) if 0<=i+k<len(sn) and c==sn[i+k])
        if m>best[0]: best=(m,k)
    return best[1]

# 6AMU native MHC reference (res<=180) + DRG; GIG mapped into same frame
_mU = _pp.get_structure('U', f"{ROOT}/inputs/focus_6am/6AMU.pdb")[0]
_natA = _rl(_mU['A']); _sA = _sq(_natA); _caA = _ca(_natA)
_REFmask = [i for i,r in enumerate(_natA) if r.id[1]<=180]
REF_CA = _caA[_REFmask]; _REFseq = ''.join(_sA[i] for i in _REFmask)
DRG = _ca([r for r in _mU['C'] if r.id[0]==' '])

def to_common(mhc_rs, pep_rs):
    """Superpose a structure's MHC onto the 6AMU reference; return peptide CA in the
    common groove frame + the MHC-fit RMSD (alignment quality)."""
    so=_sq(mhc_rs); k=_off(so,_REFseq)
    cO=_ca(mhc_rs)
    Po=[cO[i] for i in range(len(so)) if 0<=i+k<len(_REFseq) and so[i]==_REFseq[i+k]]
    Pn=[REF_CA[i+k] for i in range(len(so)) if 0<=i+k<len(_REFseq) and so[i]==_REFseq[i+k]]
    if len(Po)<50: return None,None
    R,t,fit=_robust(np.array(Po),np.array(Pn))
    return _ca(pep_rs)@R+t, fit

# GIG (6AM5 peptide) into common frame
_m5 = _pp.get_structure('5', f"{ROOT}/inputs/focus_6am/6AM5.pdb")[0]
GIG,_ = to_common(_rl(_m5['A']), [r for r in _m5['C'] if r.id[0]==' '])

def _detect(path):
    """return (mhc_residues, peptide_residues) using generic chain detection."""
    m=_pp.get_structure('x',path)[0]; chs={c.id:_rl(c) for c in m}
    pep=[c for c,rs in chs.items() if len(rs)==10]
    mhc=[c for c,rs in chs.items() if len(rs)>150 and 'SHSMRYFF' in _sq(rs)[:14]]
    if not pep or not mhc: return None,None
    return chs[mhc[0]], chs[pep[0]]

def _features(p):
    rg=float(np.sqrt(((p-p.mean(0))**2).sum(1).mean()))
    e2e=float(np.linalg.norm(p[0]-p[-1]))
    axis=p[-1]-p[0]; axis=axis/ (np.linalg.norm(axis)+1e-9)
    proj=(p-p[0])@axis; perp=np.linalg.norm((p-p[0])-np.outer(proj,axis),axis=1)
    return rg,e2e,float(perp.max())

def _pid_from(path):
    base=os.path.basename(path); d=os.path.basename(os.path.dirname(os.path.dirname(path)))
    if 'xcond' in path:                       # job dir: ifmhc_rfd3_xcond_<pid>_as<other>
        parts=d.replace('ifmhc_rfd3_','').split('_'); return parts[1]
    return '6AMU' if (base.startswith('6AMU') or '_6AMU' in d or '6AMU' in d) else '6AM5'

def _iter_sources():
    """yield (cond, pid, detail, file) for every structure to load."""
    # natives
    yield ('crystal','6AMU','native', f"{ROOT}/inputs/focus_6am/6AMU.pdb")
    yield ('crystal','6AM5','native', f"{ROOT}/inputs/focus_6am/6AM5.pdb")
    # native relaxed snapshots
    for pid in ['6AMU','6AM5']:
        for g in [f"outputs/focus_relax/snapshots/{pid}_snap*.pdb",
                  f"outputs/_relax_gap_peek/snaps/{pid}_snap*.pdb",
                  f"outputs/relax_campaign/snapshots/{pid}_snap*.pdb"]:
            for f in glob.glob(f"{ROOT}/{g}"): yield ('relaxed',pid,'relaxed',f)
    # v1 de-novo
    for d in ["outputs/grind/pdb","outputs/ladder/pdb","outputs/promising/pdb"]:
        for f in glob.glob(f"{ROOT}/{d}/*.pdb"):
            if "_split" in f: continue
            lab=cond_label(f)
            if lab: yield (lab, '6AMU' if os.path.basename(f).startswith('6AMU') else '6AM5', os.path.basename(f)[:-4], f)
    # tool stages
    for pid in ['6AMU','6AM5']:
        for f in glob.glob(f"{ROOT}/outputs/tamarind/results/ifmhc_relax_{pid}_*/relaxed.pdb"):
            yield ('openmm-relax',pid,'openmm-relax',f)
        for tool in ["rosetta-relax","openmm-MD-10ns","openmm-MD-native"]:
            for f in glob.glob(f"{ROOT}/outputs/tamarind/derived/{tool}/{pid}_*.pdb"):
                yield (tool,pid,tool,f)
    # RFD3
    for f in glob.glob(f"{ROOT}/outputs/tamarind/results/ifmhc_rfd3_*/ifmhc_rfd3_*-*/config_*model_0.pdb"):
        lab=cond_label(f)
        if lab:
            d=os.path.basename(os.path.dirname(os.path.dirname(f))).replace('ifmhc_rfd3_','')
            yield (lab,_pid_from(f),d,f)

def load_all(force=False):
    os.makedirs(CACHE, exist_ok=True)
    fdf=f"{CACHE}/df.parquet"; fco=f"{CACHE}/coords.npy"; ftr=f"{CACHE}/traj.parquet"
    if not force and os.path.exists(fdf) and os.path.exists(fco):
        DF=pd.read_parquet(fdf); COORDS=np.load(fco)
        TRAJ=pd.read_parquet(ftr) if os.path.exists(ftr) else pd.DataFrame()
        return DF, COORDS, TRAJ
    rows=[]; coords=[]; skipped=0
    for cond,pid,detail,f in _iter_sources():
        try:
            cond=refine_cond(cond,detail)                       # split scan/xcond into sub-conditioning
            mhc,pep=_detect(f)
            if mhc is None or len(pep)!=10: skipped+=1; continue
            pc,fit=to_common(mhc,pep)
            if pc is None: skipped+=1; continue
            rg,e2e,mb=_features(pc)
            rows.append(dict(pid=pid,cond=cond,nice=nice(cond),group=group_of(cond),detail=detail,
                             toDRG=round(_rmsd(pc,DRG),3),toGIG=round(_rmsd(pc,GIG),3),fit=round(fit,3),
                             Rg=round(rg,3),e2e=round(e2e,3),maxbulge=round(mb,3),file=f))
            coords.append(pc)
        except Exception:
            skipped+=1
    DF=pd.DataFrame(rows)
    DF["basin"]=np.where(DF.toDRG<DF.toGIG,"DRG","GIG")
    DF["seated"]=DF[["toDRG","toGIG"]].min(axis=1)<4.0
    COORDS=np.array(coords)
    DF=DF.reset_index(drop=True)
    DF.to_parquet(fdf); np.save(fco,COORDS)
    # trajectories
    tp=f"{ROOT}/outputs/tamarind/md_traj.csv"
    TRAJ=pd.read_csv(tp) if os.path.exists(tp) else pd.DataFrame()
    if len(TRAJ): TRAJ.to_parquet(ftr)
    json.dump({"n":len(DF),"skipped":skipped},open(f"{CACHE}/meta.json","w"))
    return DF, COORDS, TRAJ

if __name__=="__main__":
    DF,COORDS,TRAJ=load_all(force=True)
    print(f"loaded {len(DF)} peptides | COORDS {COORDS.shape} | TRAJ {len(TRAJ)} frames")
    print(f"DRG-GIG basin separation: {_rmsd(DRG,GIG):.2f} A")
    print("\ncounts by group:"); print(DF.group.value_counts().to_string())
    print("\nseated% by conditioning:")
    g=DF.groupby('cond').agg(n=('seated','size'),seated=('seated','mean'),toDRG_med=('toDRG','median'),toGIG_med=('toGIG','median'))
    print(g.assign(seated=(g.seated*100).round(0)).round(2).to_string())
