#!/usr/bin/env python
"""OpenMM relaxation ensemble for each pMHC-TCR dataset structure -> N snapshots each."""
import csv, os, sys, traceback
from pdbfixer import PDBFixer
from openmm.app import ForceField, Simulation, PDBFile, CutoffNonPeriodic, HBonds
from openmm import LangevinMiddleIntegrator, Platform
from openmm.unit import kelvin, picosecond, picoseconds, nanometer
N_SNAP = int(sys.argv[1]) if len(sys.argv) > 1 else 8
DDIR = "inputs/pmhc_tcr_dataset"
OUT = "outputs/dataset_relax/snapshots"; os.makedirs(OUT, exist_ok=True)
ff = ForceField("amber14-all.xml", "implicit/gbn2.xml")
plat = Platform.getPlatformByName("CPU")   # CUDA build broken on this driver (PTX)
valid = [r["pdb"] for r in csv.DictReader(open(f"{DDIR}/dataset.csv")) if r["valid"] == "True"]
ok = 0
for pid in valid:
    if os.path.exists(f"{OUT}/{pid}_snap{N_SNAP-1:02d}.pdb"):
        ok += 1; print(f"{pid}: already done", flush=True); continue
    try:
        fx = PDBFixer(filename=f"{DDIR}/{pid}.pdb")
        fx.removeHeterogens(False)
        fx.findMissingResidues(); fx.findMissingAtoms(); fx.addMissingAtoms(); fx.addMissingHydrogens(7.0)
        sysm = ff.createSystem(fx.topology, nonbondedMethod=CutoffNonPeriodic,
                               nonbondedCutoff=1.0*nanometer, constraints=HBonds)
        integ = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
        sim = Simulation(fx.topology, sysm, integ, plat)
        sim.context.setPositions(fx.positions)
        sim.minimizeEnergy(maxIterations=500)
        sim.context.setVelocitiesToTemperature(300*kelvin)
        for i in range(N_SNAP):
            sim.step(1000)
            st = sim.context.getState(getPositions=True)
            PDBFile.writeFile(fx.topology, st.getPositions(),
                              open(f"{OUT}/{pid}_snap{i:02d}.pdb", "w"), keepIds=True)
        ok += 1; print(f"{pid}: {N_SNAP} snapshots OK", flush=True)
    except Exception as e:
        print(f"{pid}: FAIL {type(e).__name__}: {str(e)[:80]}", flush=True)
print(f"\nrelaxed {ok}/{len(valid)} structures -> {OUT}")
