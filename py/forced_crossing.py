#!/usr/bin/env python3
"""A2 — forced-crossing rotamer-recruitment test (the definitive coupling probe apo could not give).
Start from GIG (6AM5 pMHC, p10 in F-pocket, swap ~ -3.4). Steer the register CV swap = d(p10,Fc)-d(p9,Fc)
from GIG (-3.4) to DRG (+3.7) so the anchor is FORCED to cross p10->p9 in ONE trajectory. Crucially the MHC
BACKBONE is held but ALL side chains are FREE — so Tyr116/Lys146 CAN respond. Log their χ1 vs swap:
 - if Tyr116 flips trans->g- and Lys146 ->g- AS swap crosses 0  => the DRG rotamers are RECRUITED by the
   anchor position (peptide-induced coupling proven — the thing separate MD ensembles couldn't show).
 - if they stay put while the anchor crosses => even a forced crossing doesn't recruit them (rotamers
   incidental; register is purely the peptide backbone).
Control Arg111. Implicit gbn2, OpenCL. Env: START(6AM5 pMHC pdb), TAG, PULL_NS(2), KCV(20), CHUNK_PS(10)."""
import os, numpy as np
from openmm import unit, LangevinMiddleIntegrator, Platform, CustomExternalForce, CustomCentroidBondForce
from openmm.app import ForceField, Modeller, Simulation, HBonds, CutoffNonPeriodic, Element
from pdbfixer import PDBFixer

START=os.environ.get("START","inputs/focus_6am/6AM5.pdb"); TAG=os.environ.get("TAG","A2_forced")
PULL_NS=float(os.environ.get("PULL_NS","2")); KCV=float(os.environ.get("KCV","20"))
SWAP0=float(os.environ.get("SWAP0","-3.4")); SWAP1=float(os.environ.get("SWAP1","3.7")); T_K=float(os.environ.get("T_K","310"))
CHUNK_PS=float(os.environ.get("CHUNK_PS","10")); FP=[77,80,84,116,123,143]
OUT=f"outputs/forced_crossing/{TAG}.log"; os.makedirs("outputs/forced_crossing",exist_ok=True)
def logln(s): print(s,flush=True); open(OUT,"a").write(s+"\n")
def chi(p):
    n1=np.cross(p[1]-p[0],p[2]-p[1]); n2=np.cross(p[2]-p[1],p[3]-p[2])
    m=np.cross(n1,p[2]-p[1])/np.linalg.norm(p[2]-p[1]); return np.degrees(np.arctan2(m@n2,n1@n2))

fx=PDBFixer(START)
fx.removeChains([i for i,c in enumerate(fx.topology.chains()) if c.id not in ("A","B","C")])  # pMHC only
fx.removeHeterogens(False); fx.findMissingResidues(); fx.missingResidues={}
fx.findMissingAtoms(); fx.addMissingAtoms(); fx.addMissingHydrogens(7.0)
ff=ForceField("amber14-all.xml","implicit/gbn2.xml"); mod=Modeller(fx.topology,fx.positions)
system=ff.createSystem(mod.topology,nonbondedMethod=CutoffNonPeriodic,nonbondedCutoff=1.0*unit.nanometer,constraints=HBonds)
atoms=list(mod.topology.atoms()); pos_nm=mod.positions.value_in_unit(unit.nanometer)
# hold MHC+b2m BACKBONE only (N,CA,C,O) -> fold fixed but side chains (incl Tyr116/Lys146) FREE to respond
rest=CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
rest.addGlobalParameter("k",(5.0*unit.kilocalories_per_mole/unit.angstrom**2).value_in_unit(unit.kilojoules_per_mole/unit.nanometer**2))
for pn in ("x0","y0","z0"): rest.addPerParticleParameter(pn)
for a in atoms:
    if a.residue.chain.id in ("A","B") and a.name in ("N","CA","C","O"):
        x,y,z=pos_nm[a.index]; rest.addParticle(a.index,[x,y,z])
system.addForce(rest)
# swap CV bias on peptide p9/p10 vs F-pocket centroid
fp_ca=[a.index for a in atoms if a.residue.chain.id=="A" and a.name=="CA" and a.residue.id.strip().isdigit() and int(a.residue.id) in FP]
pepca=[a.index for a in atoms if a.residue.chain.id=="C" and a.name=="CA"]; p9,p10=pepca[-2],pepca[-1]
cv=CustomCentroidBondForce(3,"0.5*kcv*((distance(g1,g2)-distance(g3,g2))*10 - s0)^2")
cv.addGlobalParameter("kcv",KCV*4.184); cv.addGlobalParameter("s0",SWAP0)
cv.addGroup([p10]); cv.addGroup(fp_ca); cv.addGroup([p9]); cv.addBond([0,1,2],[]); system.addForce(cv)
# chi1 atom lookup for 116/146/111 (chain A author)
def rat(auth):
    for r in mod.topology.residues():
        if r.chain.id=="A" and int(r.id)==auth: return {a.name:a.index for a in r.atoms()},r.name
    return None,None
r116,n116=rat(116); r146,n146=rat(146); r111,n111=rat(111)
def chi1(pos,ra): return chi(np.array([pos[ra[n]] for n in ("N","CA","CB","CG")]))
def swap_now(pos): P=pos[fp_ca].mean(0); return float(np.linalg.norm(pos[p10]-P)-np.linalg.norm(pos[p9]-P))
sim=Simulation(mod.topology,system,LangevinMiddleIntegrator(T_K*unit.kelvin,1/unit.picosecond,0.002*unit.picoseconds),Platform.getPlatformByName("OpenCL"))
sim.context.setPositions(mod.positions); sim.minimizeEnergy(maxIterations=300)
def getpos(): return sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom)
p0=getpos()
logln(f"[{TAG}] pMHC {len(atoms)} atoms | res116={n116} 146={n146} 111={n111} | start swap {swap_now(p0):+.2f} | steer {SWAP0}->{SWAP1} over {PULL_NS}ns")
logln(f"[{TAG}] start χ1: Tyr116 {chi1(p0,r116):+.0f} Lys146 {chi1(p0,r146):+.0f} Arg111 {chi1(p0,r111):+.0f}")
nwin=int(PULL_NS*1000/CHUNK_PS); per=int(CHUNK_PS/0.002)
for w in range(nwin):
    s0=SWAP0+(SWAP1-SWAP0)*(w+1)/nwin; sim.context.setParameter("s0",s0); sim.step(per)
    if w%5==0 or w==nwin-1:
        pos=getpos()
        logln(f"[{TAG}] s0={s0:+.2f} swap={swap_now(pos):+.2f} | Tyr116χ1 {chi1(pos,r116):+.0f} Lys146χ1 {chi1(pos,r146):+.0f} Arg111χ1 {chi1(pos,r111):+.0f}")
logln(f"[{TAG}] DONE — did Tyr116/Lys146 flip toward g-(~-60) as swap crossed 0? (recruited vs incidental)")
