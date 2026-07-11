#!/usr/bin/env python3
"""Strip-side-chains test: mutate the peptide to POLY-ALANINE (backbone + Cβ only, no anchor side chains)
and run MD on the pMHC. The register (GIG p10-in / DRG p9-in) is defined by which ANCHOR SIDE CHAIN fills
the F-pocket; removing the side chains asks whether the two-state barrier is created by the anchor identity
or lives in the backbone itself.
 - poly-Ala swap CROSSES 0 / flips  -> barrier is side-chain-anchor-driven (strip anchor -> backbone free).
 - poly-Ala stays at its start swap  -> register preference is in the backbone, independent of anchor.
From both GIG (6AM5) and DRG (6AMU) backbones, x {300,370K}. MHC/b2m backbone held, peptide free.
Env: START(crystal pdb), TAG, T_K, NS(5), CHUNK_PS(20). Implicit gbn2, OpenCL."""
import os, numpy as np
from openmm import unit, LangevinMiddleIntegrator, Platform, CustomExternalForce
from openmm.app import ForceField, Modeller, Simulation, HBonds, CutoffNonPeriodic, Element
from pdbfixer import PDBFixer
from Bio.PDB import PDBParser, PDBIO, Select

START=os.environ["START"]; TAG=os.environ["TAG"]; T_K=float(os.environ.get("T_K","300"))
NS=float(os.environ.get("NS","5")); CHUNK_PS=float(os.environ.get("CHUNK_PS","20")); FP=[77,80,84,116,123,143]
OUT=f"outputs/strip_md/{TAG}.log"; os.makedirs("outputs/strip_md",exist_ok=True)
def logln(s): print(s,flush=True); open(OUT,"a").write(s+"\n")

# --- build pMHC with peptide mutated to poly-ALA (keep backbone N,CA,C,O; drop TCR + waters) ---
p=PDBParser(QUIET=True); m=p.get_structure("x",START)[0]
class Sel(Select):
    def accept_chain(s,c): return c.id in ("A","B","C")
    def accept_residue(s,r): return r.id[0]==" "
    def accept_atom(s,a):
        r=a.get_parent()
        if r.get_parent().id=="C": return a.name in ("N","CA","C","O")   # peptide: backbone only
        return a.element!="H"
# rename peptide residues to ALA
for r in m["C"]:
    if r.id[0]==" ": r.resname="ALA"
tmp=f"outputs/strip_md/_{TAG}_polyA.pdb"; io=PDBIO(); io.set_structure(m); io.save(tmp, Sel())

fx=PDBFixer(tmp); fx.findMissingResidues(); fx.missingResidues={}
fx.findMissingAtoms(); fx.addMissingAtoms(); fx.addMissingHydrogens(7.0)   # rebuilds ALA Cβ + all H
ff=ForceField("amber14-all.xml","implicit/gbn2.xml"); mod=Modeller(fx.topology,fx.positions)
system=ff.createSystem(mod.topology,nonbondedMethod=CutoffNonPeriodic,nonbondedCutoff=1.0*unit.nanometer,constraints=HBonds)
atoms=list(mod.topology.atoms()); pos_nm=mod.positions.value_in_unit(unit.nanometer)
# hold MHC+b2m backbone; peptide free
rest=CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
rest.addGlobalParameter("k",(5.0*unit.kilocalories_per_mole/unit.angstrom**2).value_in_unit(unit.kilojoules_per_mole/unit.nanometer**2))
for pn in ("x0","y0","z0"): rest.addPerParticleParameter(pn)
for a in atoms:
    if a.residue.chain.id in ("A","B") and a.name in ("N","CA","C","O"):
        x,y,z=pos_nm[a.index]; rest.addParticle(a.index,[x,y,z])
system.addForce(rest)
fp_ca=[a.index for a in atoms if a.residue.chain.id=="A" and a.name=="CA" and a.residue.id.strip().isdigit() and int(a.residue.id) in FP]
pepca=[a.index for a in atoms if a.residue.chain.id=="C" and a.name=="CA"]; p9,p10=pepca[-2],pepca[-1]
def swap(pos): P=np.array(pos.value_in_unit(unit.angstrom)); fc=P[fp_ca].mean(0); return float(np.linalg.norm(P[p10]-fc)-np.linalg.norm(P[p9]-fc))
sim=Simulation(mod.topology,system,LangevinMiddleIntegrator(T_K*unit.kelvin,1/unit.picosecond,0.002*unit.picoseconds),Platform.getPlatformByName("OpenCL"))
sim.context.setPositions(mod.positions); sim.minimizeEnergy(maxIterations=300)
def gp(): return sim.context.getState(getPositions=True).getPositions()
s0=swap(gp())
logln(f"[{TAG}] poly-ALA pMHC {len(atoms)} atoms | Fpocket {len(fp_ca)}/6 | start swap {s0:+.2f} (GIG~-3.3 DRG~+3.5)")
nchunk=int(CHUNK_PS/0.002); maxch=int(NS*1000/CHUNK_PS); tr=[]
for c in range(maxch):
    sim.step(nchunk); sw=swap(gp()); tr.append(round(sw,2))
    if c%3==0: logln(f"[{TAG}] t={int((c+1)*CHUNK_PS)}ps swap {sw:+.2f}")
tr=np.array(tr)
logln(f"[{TAG}] ==> swap start {s0:+.2f} end {tr[-1]:+.2f} | min {tr.min():+.2f} max {tr.max():+.2f} | crossed0={'YES' if (tr.min()<0)!=(tr.max()<0) or abs(tr).min()<0.8 else 'no'}")
