#!/usr/bin/env python3
"""Steered MD: measure whether the TCR LOWERS the register barrier (causation).
Constant-velocity pull on the frame-independent register CV
    swap = d(p10, Fcentroid) - d(p9, Fcentroid)
from the unshifted value (~ -3.4) to the shifted value (~ +3.7) over PULL_NS, with a moving harmonic
restraint (CustomCentroidBondForce). Record the external work W = ∫ F·dλ (trapezoid of the restraint
energy gradient) per pull. Run PULLS independent pulls for each system (+TCR held engaged vs -TCR
control); the MHC(+TCR) heavy atoms are position-restrained so only the peptide moves. If mean work
+TCR < -TCR, the engaged TCR lowers the barrier to the shifted register => TCR-driven register shift.
Env: START, TAG, PULLS(4), PULL_NS(1.5), KCV(20 kcal/mol/A^2), SWAP0(-3.4), SWAP1(3.7), T_K(310)."""
import os, numpy as np
from openmm import unit, LangevinMiddleIntegrator, Platform, CustomExternalForce, CustomCentroidBondForce
from openmm.app import ForceField, Modeller, Simulation, HBonds, CutoffNonPeriodic, Element
from pdbfixer import PDBFixer

START=os.environ["START"]; TAG=os.environ.get("TAG","steer")
PULLS=int(os.environ.get("PULLS","4")); PULL_NS=float(os.environ.get("PULL_NS","1.5"))
KCV=float(os.environ.get("KCV","20")); SWAP0=float(os.environ.get("SWAP0","-3.4")); SWAP1=float(os.environ.get("SWAP1","3.7"))
T_K=float(os.environ.get("T_K","310")); FP=[77,80,84,116,123,143]
OUT=f"outputs/tcr_causation/{TAG}.log"
def logln(s): print(s,flush=True); open(OUT,"a").write(s+"\n")

fx=PDBFixer(START); fx.findMissingResidues(); fx.missingResidues={}
fx.findMissingAtoms(); fx.addMissingAtoms(); fx.addMissingHydrogens(7.0)
ff=ForceField("amber14-all.xml","implicit/gbn2.xml")
mod=Modeller(fx.topology, fx.positions)
system=ff.createSystem(mod.topology, nonbondedMethod=CutoffNonPeriodic, nonbondedCutoff=1.0*unit.nanometer, constraints=HBonds)
atoms=list(mod.topology.atoms()); pos_nm=mod.positions.value_in_unit(unit.nanometer)
KJA2=(unit.kilocalories_per_mole/unit.angstrom**2).conversion_factor_to(unit.kilojoules_per_mole/unit.nanometer**2)
# hold non-peptide heavy atoms
rest=CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)"); rest.addGlobalParameter("k",5.0*KJA2)
for p in ("x0","y0","z0"): rest.addPerParticleParameter(p)
for a in atoms:
    if a.residue.chain.id!="C" and a.element!=Element.getBySymbol("H"):
        x,y,z=pos_nm[a.index]; rest.addParticle(a.index,[x,y,z])
system.addForce(rest)
# register CV bias: 0.5*kcv*( (dist(p10,Fc) - dist(p9,Fc)) - s0 )^2  via CustomCentroidBondForce
fp_idx=[a.index for a in atoms if a.residue.chain.id=="A" and a.name=="CA" and a.residue.id.strip().isdigit() and int(a.residue.id) in FP]
pepca=[a.index for a in atoms if a.residue.chain.id=="C" and a.name=="CA"]; p9,p10=pepca[-2],pepca[-1]
cv=CustomCentroidBondForce(3, "0.5*kcv*((distance(g1,g2)-distance(g3,g2))*10 - s0)^2")  # nm->A inside so s0 in A
cv.addGlobalParameter("kcv", KCV*unit.kilocalories_per_mole.conversion_factor_to(unit.kilojoules_per_mole))  # kJ per A^2 (CV already in A)
cv.addGlobalParameter("s0", SWAP0)
cv.addGroup([p10]); cv.addGroup(fp_idx); cv.addGroup([p9])
cv.addBond([0,1,2], [])
system.addForce(cv)
has_tcr=any(c.id in ("D","E") for c in mod.topology.chains())
logln(f"[{TAG}] atoms={len(atoms)} TCR={has_tcr} PULLS={PULLS} PULL_NS={PULL_NS} KCV={KCV} {SWAP0}->{SWAP1}A")

sim=Simulation(mod.topology, system, LangevinMiddleIntegrator(T_K*unit.kelvin,1/unit.picosecond,0.002*unit.picoseconds),
               Platform.getPlatformByName("OpenCL"))
sim.context.setPositions(mod.positions); sim.minimizeEnergy(maxIterations=300)
eqpos=sim.context.getState(getPositions=True).getPositions(asNumpy=True)
def swap_now():
    P=sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    fc=np.array(P)[fp_idx].mean(0); return float(np.linalg.norm(P[p10]-fc)-np.linalg.norm(P[p9]-fc))

nsteps=int(PULL_NS*1000/0.002); nwin=150; per=nsteps//nwin
works=[]; reached=[]
for pull in range(PULLS):
    sim.context.setPositions(eqpos); sim.context.setVelocitiesToTemperature(T_K*unit.kelvin, pull+int(os.environ.get("SEED0","11")))
    sim.context.setParameter("s0", SWAP0)
    W=0.0; prev_s0=SWAP0
    for w in range(nwin):
        s0=SWAP0+(SWAP1-SWAP0)*(w+1)/nwin
        # work increment: -dE/ds0 evaluated ~ kcv*(s0 - swap) * ds0  (force on the moving restraint center)
        cur=swap_now(); dW=(KCV*4.184)*(s0-cur)*(s0-prev_s0)   # kJ/mol
        W+=dW; prev_s0=s0
        sim.context.setParameter("s0", s0); sim.step(per)
    fin=swap_now(); works.append(W); reached.append(fin)
    logln(f"[{TAG}] pull{pull}: work={W:8.1f} kJ/mol  final swap={fin:+.2f}")
logln(f"[{TAG}] ==> mean work = {np.mean(works):.1f} +/- {np.std(works):.1f} kJ/mol | mean final swap {np.mean(reached):+.2f}")
