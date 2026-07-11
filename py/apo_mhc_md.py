#!/usr/bin/env python3
"""Apo-MHC MD: empty groove, no peptide, no TCR (MHC heavy chain + beta2m only). Asks the question the
bound peptide has been hiding: once nothing pins the groove, do the F-pocket side chains Tyr116 / Lys146
FLEX between their GIG-state (Tyr116 trans / Lys146 trans) and DRG-state (Tyr116 g- / Lys146 g-), or stay
locked in the shape they started in? Started from BOTH crystals (6AM5=GIG-shaped, 6AMU=DRG-shaped) x {300,370K}.
 - If apo samples BOTH χ1 states regardless of start  -> side chains are freely flexible; the peptide merely
   SELECTS -> Tyr116/Lys146 are passive followers, not register-encoders (register is purely the peptide).
 - If apo stays locked in its start state                -> intrinsic per-shape rigidity (memory).
Control residue Arg111 (distal). Implicit gbn2, OpenCL. Env: PDB, TAG, T_K, NS(8), CHUNK_PS(20)."""
import os, numpy as np
from openmm import unit, LangevinMiddleIntegrator, Platform
from openmm.app import ForceField, Modeller, Simulation, HBonds, CutoffNonPeriodic, Element, PDBFile
from pdbfixer import PDBFixer

PDB=os.environ["PDB"]; TAG=os.environ["TAG"]; T_K=float(os.environ.get("T_K","300"))
NS=float(os.environ.get("NS","8")); CHUNK_PS=float(os.environ.get("CHUNK_PS","20"))
OUT=f"outputs/apo_mhc/{TAG}.log"; os.makedirs("outputs/apo_mhc",exist_ok=True)
def logln(s): print(s,flush=True); open(OUT,"a").write(s+"\n")
def chi(p):  # dihedral of 4 points -> deg
    b0,b1,b2=p[0]-p[1],p[2]-p[1],p[3]-p[2]
    n1=np.cross(p[1]-p[0],p[2]-p[1]); n2=np.cross(p[2]-p[1],p[3]-p[2])
    m=np.cross(n1,p[2]-p[1])/np.linalg.norm(p[2]-p[1])
    return np.degrees(np.arctan2(m@n2, n1@n2))

# keep only MHC (chain A) + b2m (chain B); drop peptide C, TCR D/E
fx=PDBFixer(PDB)
fx.removeChains([i for i,c in enumerate(fx.topology.chains()) if c.id not in ("A","B")])
fx.removeHeterogens(False)                      # strip crystal waters / ligands
fx.findMissingResidues(); fx.missingResidues={}; fx.findMissingAtoms(); fx.addMissingAtoms(); fx.addMissingHydrogens(7.0)
ff=ForceField("amber14-all.xml","implicit/gbn2.xml"); mod=Modeller(fx.topology,fx.positions)
system=ff.createSystem(mod.topology,nonbondedMethod=CutoffNonPeriodic,nonbondedCutoff=1.0*unit.nanometer,constraints=HBonds)
atoms=list(mod.topology.atoms())
# locate chain A residues 111/116/146 (author resSeq) + a1a2 CA for fold-stability RMSD
def res_atoms(auth):
    for r in mod.topology.residues():
        if r.chain.id=="A" and int(r.id)==auth: return {a.name:a.index for a in r.atoms()}, r.name
    return None,None
r111,n111=res_atoms(111); r116,n116=res_atoms(116); r146,n146=res_atoms(146)
a1a2=[a.index for a in atoms if a.residue.chain.id=="A" and a.name=="CA" and int(a.residue.id)<=180]
logln(f"[{TAG}] apo MHC+b2m, {len(atoms)} atoms | res116={n116} 146={n146} 111={n111} | T={T_K}K NS={NS}")
def chi1(pos,ra):
    P=np.array([pos[ra[n]] for n in ("N","CA","CB","CG")]); return chi(P)
sim=Simulation(mod.topology,system,LangevinMiddleIntegrator(T_K*unit.kelvin,1/unit.picosecond,0.002*unit.picoseconds),Platform.getPlatformByName("OpenCL"))
sim.context.setPositions(mod.positions); sim.minimizeEnergy(maxIterations=300)
p0=sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom)
ref=p0[a1a2]
c0=(chi1(p0,r116),chi1(p0,r146),chi1(p0,r111))
logln(f"[{TAG}] start χ1: Tyr116 {c0[0]:+.0f}  Lys146 {c0[1]:+.0f}  Arg111 {c0[2]:+.0f}")
nchunk=int(CHUNK_PS/0.002); maxch=int(NS*1000/CHUNK_PS)
tr={"116":[],"146":[],"111":[],"rmsd":[]}
for c in range(maxch):
    sim.step(nchunk); pos=sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    tr["116"].append(round(chi1(pos,r116),0)); tr["146"].append(round(chi1(pos,r146),0)); tr["111"].append(round(chi1(pos,r111),0))
    m=pos[a1a2]-pos[a1a2].mean(0); rf=ref-ref.mean(0)
    U,S,Vt=np.linalg.svd(m.T@rf); d=np.sign(np.linalg.det(U@Vt)); R=U@np.diag([1,1,d])@Vt
    tr["rmsd"].append(round(float(np.sqrt(((m@R-rf)**2).sum(1).mean())),2))
    if c%5==0: logln(f"[{TAG}] t={int((c+1)*CHUNK_PS)}ps  Tyr116 {tr['116'][-1]:+.0f}  Lys146 {tr['146'][-1]:+.0f}  Arg111 {tr['111'][-1]:+.0f}  bbRMSD {tr['rmsd'][-1]}")
def wells(a): a=np.array(a); return [int((np.abs(((a-cc+180)%360)-180)<60).mean()*100) for cc in (60,180,-60)]
logln(f"[{TAG}] Tyr116 χ1 wells g+/t/g- = {wells(tr['116'])}  (GIG-state=trans, DRG-state=g-)")
logln(f"[{TAG}] Lys146 χ1 wells g+/t/g- = {wells(tr['146'])}")
logln(f"[{TAG}] Arg111 χ1 wells g+/t/g- = {wells(tr['111'])}  (control)")
logln(f"[{TAG}] α1α2 backbone RMSD to start: end {tr['rmsd'][-1]} max {max(tr['rmsd'])} (fold stable if <~3)")
logln(f"[{TAG}] Tyr116 trace={tr['116']}")
logln(f"[{TAG}] Lys146 trace={tr['146']}")
