#!/usr/bin/env python3
"""TCR-causation test. Start = DRG peptide in the UNSHIFTED (6AMT, p10-in) register.
Two systems: (+TCR) the 6AMU TCR grafted in its engaged pose and HELD there (= the TCR load);
(-TCR) control, no TCR. In both the MHC/b2m(/TCR) heavy atoms are position-restrained so the groove
stays intact and only the peptide (chain C) is free. Read the FRAME-INDEPENDENT F-pocket swap
= d(p10,Fc) - d(p9,Fc) each chunk: start ~ -3.4 (unshifted); a shift toward +3.5 = the peptide moved
to the 6AMU (shifted/p9-in) register. If +TCR shifts and -TCR does not => TCR engagement CAUSES the
register shift (Riley's phenomenon). Env: START (pdb), TAG, REPLICAS(3), NS(3.0), CHUNK_PS(50).
Runs OpenMM OpenCL, implicit solvent (gbn2), non-periodic."""
import os, sys, numpy as np
from openmm import unit, LangevinMiddleIntegrator, Platform, CustomExternalForce
from openmm.app import ForceField, Modeller, Simulation, HBonds, CutoffNonPeriodic, Element
from pdbfixer import PDBFixer

START=os.environ["START"]; TAG=os.environ.get("TAG","run")
REPLICAS=int(os.environ.get("REPLICAS","3")); NS=float(os.environ.get("NS","3.0"))
CHUNK_PS=float(os.environ.get("CHUNK_PS","50")); T_K=float(os.environ.get("T_K","310"))
FP=[77,80,84,116,123,143]                       # MHC F-pocket floor residues (author numbering, chain A)
OUT=f"outputs/tcr_causation/{TAG}.log"; os.makedirs("outputs/tcr_causation",exist_ok=True)
def logln(s): print(s,flush=True); open(OUT,"a").write(s+"\n")

fx=PDBFixer(START)
fx.findMissingResidues(); fx.missingResidues={}         # never bridge chain gaps
fx.findMissingAtoms(); fx.addMissingAtoms(); fx.addMissingHydrogens(7.0)
ff=ForceField("amber14-all.xml","implicit/gbn2.xml")
mod=Modeller(fx.topology, fx.positions)
system=ff.createSystem(mod.topology, nonbondedMethod=CutoffNonPeriodic, nonbondedCutoff=1.0*unit.nanometer, constraints=HBonds)
atoms=list(mod.topology.atoms()); pos_nm=mod.positions.value_in_unit(unit.nanometer)
# position-restrain every NON-peptide heavy atom (MHC+b2m+TCR held -> groove intact + TCR holds its load)
rest=CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
rest.addGlobalParameter("k",(5.0*unit.kilocalories_per_mole/unit.angstrom**2).value_in_unit(unit.kilojoules_per_mole/unit.nanometer**2))
for p in ("x0","y0","z0"): rest.addPerParticleParameter(p)
nr=0
for a in atoms:
    if a.residue.chain.id!="C" and a.element!=Element.getBySymbol("H"):
        x,y,z=pos_nm[a.index]; rest.addParticle(a.index,[x,y,z]); nr+=1
system.addForce(rest)
# light peptide restraint during minimize only (clash relief; released for dynamics)
rpep=CustomExternalForce("0.5*kpep*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
rpep.addGlobalParameter("kpep",(10.0*unit.kilocalories_per_mole/unit.angstrom**2).value_in_unit(unit.kilojoules_per_mole/unit.nanometer**2))
for p in ("x0","y0","z0"): rpep.addPerParticleParameter(p)
for a in atoms:
    if a.residue.chain.id=="C" and a.element!=Element.getBySymbol("H"):
        x,y,z=pos_nm[a.index]; rpep.addParticle(a.index,[x,y,z])
system.addForce(rpep)
has_tcr = any(c.id in ("D","E") for c in mod.topology.chains())
logln(f"[{TAG}] atoms={len(atoms)} restrained_non-peptide_heavy={nr} TCR_present={has_tcr} T={T_K}K")

sim=Simulation(mod.topology, system, LangevinMiddleIntegrator(T_K*unit.kelvin,1/unit.picosecond,0.002*unit.picoseconds),
               Platform.getPlatformByName("OpenCL"))
sim.context.setPositions(mod.positions); sim.minimizeEnergy(maxIterations=300)
sim.context.setParameter("kpep",0.0)
minpos=sim.context.getState(getPositions=True).getPositions(asNumpy=True)

# scoring indices (frame-independent): F-pocket CA in chain A; peptide p9/p10 CA in chain C (by order)
fp_idx=[a.index for a in atoms if a.residue.chain.id=="A" and a.name=="CA" and a.residue.id.strip().isdigit() and int(a.residue.id) in FP]
pepca=[a.index for a in atoms if a.residue.chain.id=="C" and a.name=="CA"]
p9,p10=pepca[-2],pepca[-1]
def swap(pos):
    P=np.array(pos.value_in_unit(unit.angstrom)); fc=P[fp_idx].mean(0)
    return float(np.linalg.norm(P[p10]-fc)-np.linalg.norm(P[p9]-fc))
logln(f"[{TAG}] F-pocket CAs found={len(fp_idx)}/6 | start swap after minimize = {swap(minpos):+.2f}  (unshifted~-3.4, shifted~+3.5)")

nchunk=int(CHUNK_PS/0.002); maxch=int(NS*1000/CHUNK_PS)
best_shift=-9.9
for rep in range(REPLICAS):
    sim.context.setPositions(minpos); sim.context.setVelocitiesToTemperature(T_K*unit.kelvin, rep+7)
    tr=[]
    for c in range(maxch):
        sim.step(nchunk)
        s=swap(sim.context.getState(getPositions=True).getPositions(asNumpy=True))
        tr.append(round(s,2)); best_shift=max(best_shift,s)
    logln(f"[{TAG}] rep{rep}: swap trace (every {CHUNK_PS:.0f}ps) = {tr}")
    logln(f"[{TAG}] rep{rep}: end swap {tr[-1]:+.2f} | max {max(tr):+.2f} | min {min(tr):+.2f}")
logln(f"[{TAG}] ==> best (most-shifted) swap over all reps = {best_shift:+.2f}  ({'REACHED shifted/DRG register' if best_shift>1.0 else 'stayed unshifted/GIG-side'})")
