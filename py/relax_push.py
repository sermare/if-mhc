#!/usr/bin/env python
"""Push relaxation/MD to greater distances: heat the NATIVE peptide (in its restrained pocket) to
increasing temperatures and track how far (Cα-RMSD) it can be driven from native. Answers 'how far
can relaxation/MD push the peptide before it leaves the native basin?'
Usage: relax_push.py <6AMU|6AM5>. Writes outputs/relax_push/{pid}_traj.csv + snapshots."""
import os, sys, warnings, numpy as np
warnings.filterwarnings("ignore")
from pdbfixer import PDBFixer
from openmm.app import ForceField, Simulation, PDBFile, CutoffNonPeriodic, HBonds, Modeller
from openmm import LangevinMiddleIntegrator, Platform, CustomExternalForce
from openmm.unit import kelvin, picosecond, picoseconds, nanometer, kilojoules_per_mole
from Bio.PDB import PDBParser, PDBIO, Select, NeighborSearch
PID=sys.argv[1]; ROOT="/home/ubuntu/if-mhc"; OUT=f"{ROOT}/outputs/relax_push"; os.makedirs(f"{OUT}/pdb",exist_ok=True)
TEMPS=[310,400,500,600]; PER_T_PS=20; SNAP_EVERY=1000  # steps (2 ps)
_p=PDBParser(QUIET=True)
ff=ForceField("amber14-all.xml","implicit/gbn2.xml"); plat=Platform.getPlatformByName("CPU")
def ca(struct,chain):
    return np.array([r['CA'].coord for r in struct[chain] if r.id[0]==' ' and 'CA' in r])
def kab(P,Q):
    Pc=P-P.mean(0);Qc=Q-Q.mean(0);V,S,Wt=np.linalg.svd(Pc.T@Qc);d=np.sign(np.linalg.det(V@Wt))
    return float(np.sqrt(((Pc@(V@np.diag([1,1,d])@Wt)-Qc)**2).sum()/len(P)))
natfull=_p.get_structure("n",f"{ROOT}/inputs/focus_6am/{PID}.pdb")[0]
natca=ca(natfull,"C")
# truncate to peptide(C)+pocket(10A) for speed
chC=[r for r in natfull["C"] if r.id[0]==' ']
pep_atoms=[a for r in chC for a in r]
other=[a for ch in natfull for r in ch if ch.id!="C" and r.id[0]==' ' for a in r]
ns=NeighborSearch(other); near=set()
for a in pep_atoms:
    for b in ns.search(a.coord,10.0): near.add(b.get_parent())
class Pocket(Select):
    def accept_residue(self,r): return (r in chC) or (r in near)
tmp=f"{OUT}/{PID}_pocket.pdb"; io=PDBIO(); io.set_structure(natfull); io.save(tmp,Pocket())
fx=PDBFixer(filename=tmp); fx.findMissingResidues(); fx.missingResidues={}
fx.findMissingAtoms(); fx.addMissingAtoms(); fx.addMissingHydrogens(7.0)
mod=Modeller(fx.topology,fx.positions); top=mod.topology
pep_chain=next((c for c in top.chains() if sum(1 for _ in c.residues())==len(chC)), next(top.chains()))
system=ff.createSystem(top,nonbondedMethod=CutoffNonPeriodic,nonbondedCutoff=1.0*nanometer,constraints=HBonds)
rest=CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
rest.addGlobalParameter("k",1000.0*kilojoules_per_mole/nanometer**2)
for p in ("x0","y0","z0"): rest.addPerParticleParameter(p)
for atom in top.atoms():
    if atom.residue.chain is pep_chain: continue
    if atom.element is None or atom.element.symbol=="H": continue
    xyz=mod.positions[atom.index]; rest.addParticle(atom.index,[xyz.x,xyz.y,xyz.z])
system.addForce(rest)
integ=LangevinMiddleIntegrator(310*kelvin,1/picosecond,0.002*picoseconds)
sim=Simulation(top,system,integ,plat); sim.context.setPositions(mod.positions); sim.minimizeEnergy(maxIterations=500)
# peptide CA atom indices (in order) for RMSD
pep_ca_idx=[a.index for a in top.atoms() if a.residue.chain is pep_chain and a.name=="CA"]
import csv
rows=[]
for T in TEMPS:
    integ.setTemperature(T*kelvin); sim.context.setVelocitiesToTemperature(T*kelvin)
    nsteps=int(PER_T_PS*1000/2)  # 2 fs steps
    for blk in range(nsteps//SNAP_EVERY):
        sim.step(SNAP_EVERY)
        pos=np.array(sim.context.getState(getPositions=True).getPositions().value_in_unit(nanometer))[pep_ca_idx]*10.0
        r=kab(pos,natca); rows.append({"crystal":PID,"T":T,"ps":(blk+1)*SNAP_EVERY*2/1000.0,"rmsd_to_native":round(r,2)})
    print(f"{PID} T={T}: max RMSD so far {max(x['rmsd_to_native'] for x in rows if x['T']==T):.2f} Å",flush=True)
with open(f"{OUT}/{PID}_traj.csv","w",newline="") as o:
    w=csv.DictWriter(o,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"wrote {OUT}/{PID}_traj.csv")
open(f"{OUT}/{PID}.DONE","w").close()
