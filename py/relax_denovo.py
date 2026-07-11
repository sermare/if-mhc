#!/usr/bin/env python
"""Apply OpenMM relaxation to RFdiffusion DE-NOVO peptide backbones.
Thread the top MPNN sequence onto the de-novo peptide (chain A), build side chains (PDBFixer),
minimize + short MD (CPU), and measure: does relaxation pull the de-novo peptide back toward native
(were they strained?) or do they stay OOD in a valid basin?
Usage: relax_denovo.py <crystal 6AMU|6AM5> <wid> <nworkers> [n_per_crystal]
Writes outputs/denovo_relax/{relaxed pdbs, results.csv}."""
import os, sys, glob, re, csv, warnings
warnings.filterwarnings("ignore")
import numpy as np
from pdbfixer import PDBFixer
from openmm.app import ForceField, Simulation, PDBFile, CutoffNonPeriodic, HBonds, Modeller
from openmm import LangevinMiddleIntegrator, Platform, CustomExternalForce
from openmm.unit import kelvin, picosecond, picoseconds, nanometer, kilojoule_per_mole, kilojoules_per_mole
from Bio.PDB import PDBParser, PDBIO, Select, NeighborSearch

PID=sys.argv[1]; WID=int(sys.argv[2]) if len(sys.argv)>2 else 0
NW=int(sys.argv[3]) if len(sys.argv)>3 else 1
NPER=int(sys.argv[4]) if len(sys.argv)>4 else 15
ROOT="/home/ubuntu/if-mhc"; OUT=f"{ROOT}/outputs/denovo_relax"; os.makedirs(f"{OUT}/pdb",exist_ok=True)
NAT={"6AMU":"MMWDRGLGMM","6AM5":"SMLGIGIVPV"}
AA1to3={'A':'ALA','R':'ARG','N':'ASN','D':'ASP','C':'CYS','Q':'GLN','E':'GLU','G':'GLY','H':'HIS','I':'ILE','L':'LEU','K':'LYS','M':'MET','F':'PHE','P':'PRO','S':'SER','T':'THR','W':'TRP','Y':'TYR','V':'VAL'}
ff=ForceField("amber14-all.xml","implicit/gbn2.xml")
plat=Platform.getPlatformByName("CPU")
_p=PDBParser(QUIET=True)

def topseq(fa):
    ls=open(fa).read().splitlines()
    for i in range(0,len(ls)-1,2):
        if "sample=" in ls[i] and len(ls[i+1].strip())==10: return ls[i+1].strip()
    return None
def ca(struct,chain):
    if chain not in struct: return None
    return np.array([r['CA'].coord for r in struct[chain] if r.id[0]==' ' and 'CA' in r])
def kab(P,Q):
    if P is None or Q is None or P.shape!=Q.shape: return None
    Pc=P-P.mean(0);Qc=Q-Q.mean(0);V,S,Wt=np.linalg.svd(Pc.T@Qc);d=np.sign(np.linalg.det(V@Wt))
    return float(np.sqrt(((Pc@(V@np.diag([1,1,d])@Wt)-Qc)**2).sum()/len(P)))

natca=ca(_p.get_structure("n",f"{ROOT}/inputs/focus_6am/{PID}.pdb")[0],"C")
# candidate backbones: grind + promising L10 for this crystal
cands=sorted(glob.glob(f"{ROOT}/outputs/grind/pdb/{PID}_*_L10_*.pdb")+
             glob.glob(f"{ROOT}/outputs/promising/pdb/{PID}_*.pdb"))
cands=[c for c in cands if "_split" not in c][WID::NW][:NPER]
rows=[]
for pdb in cands:
    name=os.path.basename(pdb)[:-4]
    outpdb=f"{OUT}/pdb/{name}_relaxed.pdb"
    if os.path.exists(outpdb): continue
    fa=pdb.replace("/pdb/","/seqs/").replace(".pdb",".fa")
    if not os.path.exists(fa): continue
    seq=topseq(fa)
    if not seq: continue
    try:
        st=_p.get_structure("x",pdb)[0]
        raw_pep=ca(st,"A")
        # rename chain A (peptide) GLY placeholders -> MPNN sequence so PDBFixer builds correct side chains
        chA=[r for r in st["A"] if r.id[0]==' ']
        for r,aa in zip(chA,seq): r.resname=AA1to3[aa]
        # TRUNCATE to peptide + pocket: keep chain B residues with any atom within 10 A of the peptide
        pep_atoms=[a for r in chA for a in r]
        other_atoms=[a for ch in st for r in ch if ch.id!="A" and r.id[0]==' ' for a in r]
        ns=NeighborSearch(other_atoms)
        near=set()
        for a in pep_atoms:
            for b in ns.search(a.coord,10.0): near.add(b.get_parent())   # nearby residue objects
        class Pocket(Select):
            def accept_residue(self,r): return (r in chA) or (r in near)
        tmp=f"{OUT}/pdb/{name}_threaded.pdb"
        io=PDBIO(); io.set_structure(st); io.save(tmp,Pocket())
        fx=PDBFixer(filename=tmp)
        fx.findMissingResidues(); fx.missingResidues={}
        fx.findMissingAtoms(); fx.addMissingAtoms(); fx.addMissingHydrogens(7.0)
        modeller=Modeller(fx.topology,fx.positions)
        system=ff.createSystem(modeller.topology,nonbondedMethod=CutoffNonPeriodic,
                               nonbondedCutoff=1.0*nanometer,constraints=HBonds)
        # restrain TARGET (chain != first/peptide) heavy atoms so the groove holds and the PEPTIDE relaxes
        top=modeller.topology; pos0=modeller.positions
        # peptide chain = the one with exactly len(seq) residues (10); fall back to first
        pep_chain=next((c for c in top.chains() if sum(1 for _ in c.residues())==len(seq)), next(top.chains()))
        rest=CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        rest.addGlobalParameter("k",1000.0*kilojoules_per_mole/nanometer**2)
        for p in ("x0","y0","z0"): rest.addPerParticleParameter(p)
        nres=0
        for atom in top.atoms():
            if atom.residue.chain is pep_chain: continue
            if atom.element is None or atom.element.symbol=="H": continue
            xyz=pos0[atom.index]; rest.addParticle(atom.index,[xyz.x,xyz.y,xyz.z]); nres+=1
        system.addForce(rest)
        integ=LangevinMiddleIntegrator(300*kelvin,1/picosecond,0.002*picoseconds)
        sim=Simulation(top,system,integ,plat)
        sim.context.setPositions(pos0)
        e0=sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(kilojoule_per_mole)
        sim.minimizeEnergy(maxIterations=1000)
        emin=sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(kilojoule_per_mole)
        sim.context.setVelocitiesToTemperature(300*kelvin)
        sim.step(2000)   # 4 ps MD at 300K (peptide samples; groove restrained)
        emd=sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(kilojoule_per_mole)
        pos=sim.context.getState(getPositions=True).getPositions()
        PDBFile.writeFile(modeller.topology,pos,open(outpdb,"w"))
        rel=_p.get_structure("r",outpdb)[0]
        rel_pep=ca(rel,"A")
        rows.append({"crystal":PID,"backbone":name,"seq":seq,
                     "rmsd_relaxed_vs_denovo":round(kab(rel_pep,raw_pep) or -1,2),
                     "rmsd_denovo_vs_native":round(kab(raw_pep,natca) or -1,2),
                     "rmsd_relaxed_vs_native":round(kab(rel_pep,natca) or -1,2),
                     "E0":round(e0,0),"E_min":round(emin,0),"E_md":round(emd,0)})
        os.remove(tmp)
        print(f"{name}: denovo->nat {rows[-1]['rmsd_denovo_vs_native']} | relaxed->nat {rows[-1]['rmsd_relaxed_vs_native']} | moved {rows[-1]['rmsd_relaxed_vs_denovo']}",flush=True)
    except Exception as e:
        print(f"{name}: FAIL {e}",flush=True)
# append results (per-worker file to avoid clobber)
rf=f"{OUT}/results_{PID}_w{WID}.csv"
if rows:
    with open(rf,"w",newline="") as o:
        w=csv.DictWriter(o,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"wrote {rf} ({len(rows)} relaxed)")
open(f"{OUT}/{PID}_w{WID}.DONE","w").close()
