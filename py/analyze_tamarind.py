#!/usr/bin/env python
"""Parse Tamarind MD trajectories (peptide dual-RMSD to DRG/GIG over time) + Rosetta-relax energies.
Writes outputs/tamarind/md_traj.csv and outputs/tamarind/rosetta_scores.csv."""
import glob,os,re,json,csv,warnings,numpy as np
warnings.filterwarnings("ignore")
import mdtraj as md
from Bio.PDB import PDBParser
ROOT="/home/ubuntu/if-mhc"; RES=f"{ROOT}/outputs/tamarind/results"
_p=PDBParser(QUIET=True)
AA3TO1={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
def ca_xtal(pdb):
    m=_p.get_structure("x",pdb)[0]
    return np.array([r['CA'].coord for r in m["C"] if r.id[0]==' ' and 'CA' in r])
DRG=ca_xtal(f"{ROOT}/inputs/focus_6am/6AMU.pdb"); GIG=ca_xtal(f"{ROOT}/inputs/focus_6am/6AM5.pdb")
DRGseq="MMWDRGLGMM"; GIGseq="SMLGIGIVPV"
def kab(P,Q):
    Pc=P-P.mean(0);Qc=Q-Q.mean(0);V,S,Wt=np.linalg.svd(Pc.T@Qc);d=np.sign(np.linalg.det(V@Wt))
    return float(np.sqrt(((Pc@(V@np.diag([1,1,d])@Wt)-Qc)**2).sum()/len(P)))

man={m["jobName"]:m for mf in ["md_pilot_manifest.json","md10_manifest.json"]
     for m in json.load(open(f"{ROOT}/outputs/tamarind/{mf}"))}
def pep_seq_for(job):
    if job.startswith("ifmhc_6AMU_md"): return DRGseq
    if job.startswith("ifmhc_6AM5_md"): return GIGseq
    return man[job]["seq"]   # candidate threaded sequence

rows=[]
for d in sorted(glob.glob(f"{RES}/ifmhc_*md*")):
    job=os.path.basename(d)
    top=f"{d}/topology_no_water.pdb"; xtc=sorted(glob.glob(f"{d}/traj_prod_no_water*.xtc"))
    if not (os.path.exists(top) and xtc): continue
    try:
        t=md.load(xtc[0],top=top)
    except Exception as e:
        print(f"{job}: load fail {e}"); continue
    seq=pep_seq_for(job)
    # find the 10-residue peptide window by sequence match in the topology
    resn=[AA3TO1.get(r.name,'X') for r in t.topology.residues]
    full="".join(resn); i=full.find(seq)
    if i<0:  # fallback: candidates peptide is first 10 residues
        i=0 if not job.startswith(("ifmhc_6AMU_md","ifmhc_6AM5_md")) else full.find(seq[:6])
    pep_res=list(range(i,i+10))
    ca_idx=[a.index for a in t.topology.atoms if a.name=="CA" and a.residue.index in pep_res]
    if len(ca_idx)!=10: print(f"{job}: peptide select got {len(ca_idx)} CA (seq {seq} idx {i})"); continue
    xyz=t.xyz[:,ca_idx,:]*10.0  # nm->A, (frames,10,3)
    for fr in range(0,len(xyz),max(1,len(xyz)//100)):  # subsample ~100 frames
        rows.append({"job":job,"frame":fr,"ns":round(fr*(10.0/len(xyz)),2),
                     "to_DRG":round(kab(xyz[fr],DRG),2),"to_GIG":round(kab(xyz[fr],GIG),2)})
    print(f"{job}: {len(xyz)} frames, pep@res{i}")
with open(f"{ROOT}/outputs/tamarind/md_traj.csv","w",newline="") as o:
    w=csv.DictWriter(o,fieldnames=["job","frame","ns","to_DRG","to_GIG"]);w.writeheader();w.writerows(rows)
print(f"wrote md_traj.csv ({len(rows)} frame-rows)")

# Rosetta scores: best (lowest) total_score across the 3 nstruct
def total_score(sc):
    for ln in open(sc):
        if ln.startswith("SCORE:") and "total_score" not in ln:
            try: return float(ln.split()[1])
            except: pass
    return None
srows=[]
for d in sorted(glob.glob(f"{RES}/ifmhc_rosrlx_*")):
    job=os.path.basename(d); base=job.replace("ifmhc_rosrlx_","")
    scs=[total_score(s) for s in glob.glob(f"{d}/*/score_relaxed.sc")]
    scs=[x for x in scs if x is not None]
    if scs: srows.append({"candidate":base,"rosetta_best":round(min(scs),1),"rosetta_mean":round(np.mean(scs),1),"n":len(scs)})
with open(f"{ROOT}/outputs/tamarind/rosetta_scores.csv","w",newline="") as o:
    w=csv.DictWriter(o,fieldnames=["candidate","rosetta_best","rosetta_mean","n"]);w.writeheader();w.writerows(srows)
print(f"wrote rosetta_scores.csv ({len(srows)} candidates)")
