#!/usr/bin/env python
"""Extract the peptide (Cα) from each submitted Tamarind result and write CA-only PDBs under
outputs/tamarind/derived/<tool>/, so the notebook can load each as a tool-labelled source:
  openmm-relax, rosetta-relax, openmm-MD-10ns (candidate final frame), openmm-MD-native (final frame)."""
import glob,os,json,warnings,numpy as np
warnings.filterwarnings("ignore")
import mdtraj as md
from Bio.PDB import PDBParser
ROOT="/home/ubuntu/if-mhc"; RES=f"{ROOT}/outputs/tamarind/results"; DER=f"{ROOT}/outputs/tamarind/derived"
_p=PDBParser(QUIET=True)
AA3TO1={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
DRGseq="MMWDRGLGMM"; GIGseq="SMLGIGIVPV"
man={m["jobName"]:m for mf in ["md_pilot_manifest.json","md10_manifest.json"] for m in json.load(open(f"{ROOT}/outputs/tamarind/{mf}"))}

def write_ca(coords, out):
    os.makedirs(os.path.dirname(out),exist_ok=True)
    with open(out,"w") as f:
        for i,(x,y,z) in enumerate(coords,1):
            f.write(f"ATOM  {i:5d}  CA  GLY A{i:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
        f.write("END\n")

def pep_ca_from_pdb(pdb, chain="A"):
    m=_p.get_structure("x",pdb)[0]
    if chain not in m: return None
    rs=[r for r in m[chain] if r.id[0]==' ' and 'CA' in r]
    return np.array([r['CA'].coord for r in rs]) if len(rs)==10 else None

def total_score(sc):
    for ln in open(sc):
        if ln.startswith("SCORE:") and "total_score" not in ln:
            try: return float(ln.split()[1])
            except: pass
    return 9e9

n=0
# 1) rosetta-relax: best-energy nstruct, peptide chain A
for d in sorted(glob.glob(f"{RES}/ifmhc_rosrlx_*")):
    base=os.path.basename(d).replace("ifmhc_rosrlx_","")
    subs=[(total_score(s), os.path.dirname(s)) for s in glob.glob(f"{d}/*/score_relaxed.sc")]
    if not subs: continue
    _,best=min(subs); pdb=f"{best}/input_relaxed_0001.pdb"
    c=pep_ca_from_pdb(pdb)
    if c is not None: write_ca(c,f"{DER}/rosetta-relax/{base}.pdb"); n+=1

# 2) openmm-MD final frame (candidate + native), peptide by sequence
def md_final_pep(job):
    d=f"{RES}/{job}"; top=f"{d}/topology_no_water.pdb"; xtc=sorted(glob.glob(f"{d}/traj_prod_no_water*.xtc"))
    if not (os.path.exists(top) and xtc): return None
    t=md.load(xtc[-1],top=top)
    seq = DRGseq if job.startswith("ifmhc_6AMU_md") else GIGseq if job.startswith("ifmhc_6AM5_md") else man[job]["seq"]
    full="".join(AA3TO1.get(r.name,'X') for r in t.topology.residues); i=full.find(seq)
    if i<0: i=0
    idx=[a.index for a in t.topology.atoms if a.name=="CA" and i<=a.residue.index<i+10]
    if len(idx)!=10: return None
    return t.xyz[-1,idx,:]*10.0
for job in [os.path.basename(x) for x in glob.glob(f"{RES}/ifmhc_md_*")]:
    c=md_final_pep(job)
    if c is not None: write_ca(c,f"{DER}/openmm-MD-10ns/{job.replace('ifmhc_md_','')}.pdb"); n+=1
for job in [os.path.basename(x) for x in glob.glob(f"{RES}/ifmhc_6AM*_md_*")]:
    c=md_final_pep(job)
    if c is not None:
        pid="6AMU" if "6AMU" in job else "6AM5"; T=job.split("_")[-1]
        write_ca(c,f"{DER}/openmm-MD-native/{pid}_{T}.pdb"); n+=1
print(f"wrote {n} peptide CA PDBs under {DER}/ (rosetta-relax, openmm-MD-10ns, openmm-MD-native)")
print("counts:", {os.path.basename(x): len(glob.glob(x+'/*.pdb')) for x in glob.glob(f"{DER}/*")})
