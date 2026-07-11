import os
from pdbfixer import PDBFixer
from openmm.app import ForceField, Simulation, PDBFile, CutoffNonPeriodic, HBonds
from openmm import LangevinMiddleIntegrator, Platform
from openmm.unit import kelvin, picosecond, picoseconds, nanometer
N=8; OUT="outputs/focus_relax/snapshots"; os.makedirs(OUT,exist_ok=True)
ff=ForceField("amber14-all.xml","implicit/gbn2.xml")
plat=Platform.getPlatformByName("CPU")   # CUDA build broken on this driver (PTX version)
for pdb in ["6AMT","6AMU","6AM5"]:
    if os.path.exists(f"{OUT}/{pdb}_snap{N-1:02d}.pdb"): print(pdb,"already done",flush=True); continue
    try:
        fx=PDBFixer(f"inputs/focus_6am/{pdb}.pdb"); fx.removeHeterogens(False)
        fx.findMissingResidues();fx.findMissingAtoms();fx.addMissingAtoms();fx.addMissingHydrogens(7.0)
        sysm=ff.createSystem(fx.topology,nonbondedMethod=CutoffNonPeriodic,nonbondedCutoff=1.0*nanometer,constraints=HBonds)
        sim=Simulation(fx.topology,sysm,LangevinMiddleIntegrator(300*kelvin,1/picosecond,0.002*picoseconds),plat)
        sim.context.setPositions(fx.positions); sim.minimizeEnergy(maxIterations=500); sim.context.setVelocitiesToTemperature(300*kelvin)
        for i in range(N):
            sim.step(1000); st=sim.context.getState(getPositions=True)
            PDBFile.writeFile(fx.topology,st.getPositions(),open(f"{OUT}/{pdb}_snap{i:02d}.pdb","w"),keepIds=True)
        print(pdb,N,"snapshots OK",flush=True)
    except Exception as e: print(pdb,"FAIL",type(e).__name__,str(e)[:80],flush=True)
