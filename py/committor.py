#!/usr/bin/env python3
"""Committor analysis on the equidistant L4 MD frame.
Fire REPLICAS unbiased trajectories (random Maxwell-Boltzmann velocities) from the same minimized
seed; classify each as committing to GIG or DRG (peptide Cα-RMSD < THRESH to that native basin, in the
MHC floor frame). pB(DRG) ~ 0.5 => transition-state ensemble (on the ridge); skewed => off-path.
Env: REPLICAS (4), MAXPS (30), THRESH (1.2), CHUNK_PS (2.0). Runs on OpenMM OpenCL, implicit solvent."""
import os, sys, numpy as np
from openmm import unit, LangevinMiddleIntegrator, Platform
from openmm.app import ForceField, Modeller, Simulation, HBonds, CutoffNonPeriodic, Element
from pdbfixer import PDBFixer

REPLICAS=int(os.environ.get("REPLICAS","4")); MAXPS=float(os.environ.get("MAXPS","30"))
THRESH=float(os.environ.get("THRESH","1.2")); CHUNK_PS=float(os.environ.get("CHUNK_PS","2.0"))
AA3TO1={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V','MSE':'M','HID':'H','HIE':'H','HIP':'H'}
r=np.load("outputs/committor/refs.npz"); GIG=r["GIG"]; DRG=r["DRG"]; SEED=str(r["seed"])
REF_CA=r["REF_CA"]; REFSEQ=str(r["REFseq"])          # core_load's 6AMU MHC ref (authoritative frame)
def _RT(P,Q):
    Pc=P.mean(0);Qc=Q.mean(0);V,S,Wt=np.linalg.svd((P-Pc).T@(Q-Qc));d=np.sign(np.linalg.det(V@Wt));R=V@np.diag([1,1,d])@Wt;return R,Qc-Pc@R
def _off(so,sn):
    best=(-1,0)
    for k in range(-4,14):
        mv=sum(1 for i,c in enumerate(so) if 0<=i+k<len(sn) and c==sn[i+k])
        if mv>best[0]: best=(mv,k)
    return best[1]
def _robust(Po,Pn):
    R,t=_RT(Po,Pn)
    for _ in range(3):
        res=np.linalg.norm(Po@R+t-Pn,axis=1); keep=res<3.0
        if keep.sum()<30: break
        R,t=_RT(Po[keep],Pn[keep])
    return R,t

# --- split the merged single chain into real chains by KNOWN segment sizes (pep|MHC|b2m|TCRa|TCRb) ---
lines=open("outputs/committor/seed_frame.pdb").read().splitlines()
reskeys=[]
for l in lines:
    if l.startswith("ATOM"):
        k=(l[21],l[22:27])
        if not reskeys or reskeys[-1]!=k: reskeys.append(k)
segsz=[10,180,100,115,120]                                  # pep, MHC a1a2, b2m, TCR Va, TCR Vb
assert sum(segsz)==len(reskeys), f"residue count {len(reskeys)} != {sum(segsz)}"
chains="PABDE"; cof={}; i=0
for c,sz in zip(chains,segsz):
    for _ in range(sz): cof[reskeys[i]]=c; i+=1
out=[]
for l in lines:
    if l.startswith(("ATOM","HETATM")):
        l=l[:21]+cof.get((l[21],l[22:27]),l[21])+l[22:]
    out.append(l)
open("outputs/committor/seed_frame_chained.pdb","w").write("\n".join(out)+"\n")
print("re-chained into 5 chains by segment sizes", segsz)

# --- build + minimize once (do NOT fill the inter-domain gaps with modeled residues) ---
fx=PDBFixer("outputs/committor/seed_frame_chained.pdb")
fx.findMissingResidues(); fx.missingResidues={}          # clear: never bridge the trim gaps
fx.findMissingAtoms(); fx.addMissingAtoms()
ff=ForceField("amber14-all.xml","implicit/gbn2.xml")
mod=Modeller(fx.topology, fx.positions)
mod.delete([a for a in mod.topology.atoms() if a.element==Element.getBySymbol("H")])  # strip MD H
mod.addHydrogens(ff, pH=7.0)                                                          # amber-consistent H
system=ff.createSystem(mod.topology, nonbondedMethod=CutoffNonPeriodic, nonbondedCutoff=1.0*unit.nanometer, constraints=HBonds)
# restrain the (register-agnostic) MHC+b2m+TCR heavy atoms so the groove stays intact and only the
# peptide (chain P) is free to explore register — prevents whole-complex drift in implicit solvent.
from openmm import CustomExternalForce
rest=CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
rest.addGlobalParameter("k", (5.0*unit.kilocalories_per_mole/unit.angstrom**2).value_in_unit(unit.kilojoules_per_mole/unit.nanometer**2))
for pnm in ("x0","y0","z0"): rest.addPerParticleParameter(pnm)
pos_nm=mod.positions.value_in_unit(unit.nanometer)
atoms=list(mod.topology.atoms())
nrest=0
for a in atoms:
    if a.residue.chain.id!="P" and a.element!=Element.getBySymbol("H"):   # non-peptide heavy atoms
        x,y,z=pos_nm[a.index]; rest.addParticle(a.index,[x,y,z]); nrest+=1
system.addForce(rest)
# peptide heavy-atom restraint, ON only during minimize (kpep high) so the seed geometry is preserved,
# then released (kpep=0) so the peptide is free during the committor dynamics.
rpep=CustomExternalForce("0.5*kpep*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
rpep.addGlobalParameter("kpep",(50.0*unit.kilocalories_per_mole/unit.angstrom**2).value_in_unit(unit.kilojoules_per_mole/unit.nanometer**2))
for pnm in ("x0","y0","z0"): rpep.addPerParticleParameter(pnm)
npep=0
for a in atoms:
    if a.residue.chain.id=="P" and a.element!=Element.getBySymbol("H"):
        x,y,z=pos_nm[a.index]; rpep.addParticle(a.index,[x,y,z]); npep+=1
system.addForce(rpep)
print(f"restrained {nrest} MHC+TCR heavy atoms (always) + {npep} peptide heavy atoms (minimize only)")
sim=Simulation(mod.topology, system, LangevinMiddleIntegrator(300*unit.kelvin,1/unit.picosecond,0.002*unit.picoseconds),
               Platform.getPlatformByName("OpenCL"))
sim.context.setPositions(mod.positions); sim.minimizeEnergy(maxIterations=200)  # seed held (env+pep restrained)
sim.context.setParameter("kpep", 0.0)                                            # release the peptide for dynamics
minpos=sim.context.getState(getPositions=True).getPositions(asNumpy=True)

# --- scoring in core_load's AUTHORITATIVE frame: robust MHC(α1α2) superposition onto REF_CA, then peptide RMSD ---
# layout after chain split (order preserved): residues 0-9 = peptide, 10-189 = MHC α1α2 (180 res)
resnames=[res.name for res in mod.topology.residues()]
ca_idx=[a.index for a in mod.topology.atoms() if a.name=="CA"]              # one CA per residue, in order
seq="".join(AA3TO1.get(n,'x') for n in resnames)
mhc_seq=seq[10:190]                                                        # design MHC α1α2 sequence
k=_off(mhc_seq, REFSEQ)                                                     # sequence offset to the reference
mo=[i for i in range(len(mhc_seq)) if 0<=i+k<len(REFSEQ) and mhc_seq[i]==REFSEQ[i+k]]
mr=[i+k for i in mo]; Pn=REF_CA[mr]
print(f"MHC↔ref sequence match: {len(mo)}/180 residues (offset {k})")
def score(pos_nm):
    ca=np.array(pos_nm)[ca_idx]*10.0                       # Å, per-residue CA
    R,t=_robust(ca[10:190][mo], Pn)                        # robust MHC superpose (core_load.to_common)
    pep=ca[0:10]@R+t
    return float(np.sqrt(((pep-GIG)**2).sum()/10)), float(np.sqrt(((pep-DRG)**2).sum()/10))

g0,d0=score(minpos.value_in_unit(unit.nanometer))
print(f"seed after minimize: toGIG={g0:.2f} toDRG={d0:.2f} | THRESH={THRESH} MAXPS={MAXPS} REPLICAS={REPLICAS}", flush=True)

nchunk=int(CHUNK_PS/0.002); maxch=int(MAXPS/CHUNK_PS); res=[]
for rep in range(REPLICAS):
    sim.context.setPositions(minpos); sim.context.setVelocitiesToTemperature(300*unit.kelvin, rep+1)
    outcome="none"; g=d=np.nan
    for c in range(maxch):
        sim.step(nchunk)
        pos=sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        g,d=score(pos)
        if g<THRESH and g<d: outcome="GIG"; break
        if d<THRESH and d<g: outcome="DRG"; break
    res.append(outcome); print(f"  rep {rep:3d}: {outcome:4s}  final toGIG={g:.2f} toDRG={d:.2f}", flush=True)
G=res.count("GIG"); D=res.count("DRG"); N=res.count("none")
pB = D/(G+D) if (G+D) else float("nan")
print(f"\n=== COMMITTOR: GIG={G} DRG={D} uncommitted={N} / {REPLICAS} | pB(DRG)={pB:.2f} ===")
print("pB~0.5 => transition-state ensemble (on the ridge); ~0/1 => drains to one basin (off-path)")
open("outputs/committor/result.txt","a").write(f"REPLICAS={REPLICAS} MAXPS={MAXPS} THRESH={THRESH}: GIG={G} DRG={D} none={N} pB={pB}\n")
