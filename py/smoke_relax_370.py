import os, time
from pdbfixer import PDBFixer
from openmm.app import ForceField, Simulation, PDBFile, CutoffNonPeriodic, HBonds
from openmm import LangevinMiddleIntegrator, Platform
from openmm.unit import kelvin, picosecond, picoseconds, nanometer

INP = "inputs/focus_6am/6AM5_trim.pdb"
OUT = "outputs/_smoke_370"; os.makedirs(OUT, exist_ok=True)
T = 370.0  # kelvin
N = 4      # short smoke: a few 1-ps snapshots

ff = ForceField("amber14-all.xml", "implicit/gbn2.xml")
t0 = time.time()
fx = PDBFixer(INP); fx.removeHeterogens(False)
fx.findMissingResidues(); fx.findMissingAtoms(); fx.addMissingAtoms(); fx.addMissingHydrogens(7.0)
print("atoms:", fx.topology.getNumAtoms(), "chains:", fx.topology.getNumChains(), flush=True)

sysm = ff.createSystem(fx.topology, nonbondedMethod=CutoffNonPeriodic,
                       nonbondedCutoff=1.0*nanometer, constraints=HBonds)

# try fastest platform, fall back if the Context can't actually be built
# (CUDA build is broken on this driver: CUDA_ERROR_UNSUPPORTED_PTX_VERSION)
sim = None
for name in ("CUDA", "OpenCL", "CPU"):
    try:
        plat = Platform.getPlatformByName(name)
        sim = Simulation(fx.topology, sysm,
                         LangevinMiddleIntegrator(T*kelvin, 1/picosecond, 0.002*picoseconds), plat)
        print("platform:", name, flush=True); break
    except Exception as e:
        print(f"platform {name} unavailable: {type(e).__name__} {str(e)[:60]}", flush=True)
if sim is None:
    raise SystemExit("no usable OpenMM platform")
sim.context.setPositions(fx.positions)

e0 = sim.context.getState(getEnergy=True).getPotentialEnergy()
print("E(pre-min):", e0, flush=True)
sim.minimizeEnergy(maxIterations=500)
e1 = sim.context.getState(getEnergy=True).getPotentialEnergy()
print("E(post-min):", e1, flush=True)

sim.context.setVelocitiesToTemperature(T*kelvin)
for i in range(N):
    sim.step(500)  # 1 ps
    st = sim.context.getState(getPositions=True, getEnergy=True)
    PDBFile.writeFile(fx.topology, st.getPositions(),
                      open(f"{OUT}/6AM5_trim_370_snap{i:02d}.pdb", "w"), keepIds=True)
    print(f"snap{i:02d}  E={st.getPotentialEnergy()}", flush=True)

print(f"SMOKE OK  {N} snapshots @ {T}K in {time.time()-t0:.1f}s -> {OUT}", flush=True)
