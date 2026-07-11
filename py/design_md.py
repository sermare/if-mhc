#!/usr/bin/env python3
"""Local OpenMM MD of an RFdiffusion/MPNN DESIGN complex at a given temperature.
Input = a relaxed design complex (peptide chain + merged [MHC|b2m|TCRa|TCRb]). We re-chain it by
sequence signatures (so PDBFixer never bridges the inter-domain gaps), build an implicit-solvent
system (gbn2, OpenCL), lightly position-restrain the MHC+b2m+TCR heavy atoms (groove intact) and let
the peptide move, then run NS ns at T_K and log the FRAME-INDEPENDENT register readout each chunk:
  swap = d(p10,Fpocket) - d(p9,Fpocket)     (GIG/unshift ~ -3.3 ; DRG/shift ~ +3.5)
plus register-preserving peptide Cα-RMSD toGIG / toDRG (core_load authoritative frame via refs.npz).
Question: under thermal motion at 300/370 K, does a de-novo design peptide drift toward a native
register, or stay put? Env: PDB, TAG, T_K(300), NS(3), CHUNK_PS(50)."""
import os, numpy as np
from openmm import unit, LangevinMiddleIntegrator, Platform, CustomExternalForce
from openmm.app import ForceField, Modeller, Simulation, HBonds, CutoffNonPeriodic, Element
from pdbfixer import PDBFixer

PDB=os.environ["PDB"]; TAG=os.environ["TAG"]; T_K=float(os.environ.get("T_K","300"))
NS=float(os.environ.get("NS","3")); CHUNK_PS=float(os.environ.get("CHUNK_PS","50"))
FP_POS=[76,79,83,115,122,142]                          # 0-based in MHC a1a2 = authors 77,80,84,116,123,143
OUT=f"outputs/design_md/{TAG}.log"; os.makedirs("outputs/design_md",exist_ok=True)
def logln(s): print(s,flush=True); open(OUT,"a").write(s+"\n")
AA3TO1={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V','MSE':'M','HID':'H','HIE':'H','HIP':'H'}
r=np.load("outputs/committor/refs.npz"); GIG=r["GIG"]; DRG=r["DRG"]; REF_CA=r["REF_CA"]; REFSEQ=str(r["REFseq"])
def _RT(P,Q):
    Pc=P.mean(0);Qc=Q.mean(0);V,S,Wt=np.linalg.svd((P-Pc).T@(Q-Qc));d=np.sign(np.linalg.det(V@Wt));R=V@np.diag([1,1,d])@Wt;return R,Qc-Pc@R
def _off(so,sn):
    best=(-1,0)
    for k in range(-6,20):
        mv=sum(1 for i,c in enumerate(so) if 0<=i+k<len(sn) and c==sn[i+k])
        if mv>best[0]: best=(mv,k)
    return best[1]
def _robust(Po,Pn):
    R,t=_RT(Po,Pn)
    for _ in range(3):
        keep=np.linalg.norm(Po@R+t-Pn,axis=1)<3.0
        if keep.sum()<30: break
        R,t=_RT(Po[keep],Pn[keep])
    return R,t

# --- read residues in file order, re-chain by sequence signatures ---
lines=open(PDB).read().splitlines(); res=[]; cur=None
for l in lines:
    if l.startswith(("ATOM","HETATM")):
        key=(l[21],l[22:27])
        if key!=cur: res.append([l[17:20].strip(),[]]); cur=key
        res[-1][1].append(l)
seq="".join(AA3TO1.get(r0[0],'x') for r0 in res)
mhc=seq.find("SHSMRYFF"); b2m=seq.find("MIQRTP",mhc); ta=seq.find("EVEQNSGPL",b2m); tb=seq.find("IAGITQAP",ta)
mhc0=mhc-1 if mhc>0 and seq[mhc-1]=="G" else mhc      # include leading G in the MHC
bounds=[(0,mhc0,"C"),(mhc0,b2m,"A"),(b2m,ta,"B"),(ta,tb,"D"),(tb,len(res),"E")]  # pep,MHC,b2m,TCRa,TCRb
logln(f"[{TAG}] rechain: pep 0-{mhc0} | MHC {mhc0}-{b2m} | b2m {b2m}-{ta} | TCRa {ta}-{tb} | TCRb {tb}-{len(res)}")
out=[]
for i,(rn,rl) in enumerate(res):
    ch=next(c for lo,hi,c in bounds if lo<=i<hi)
    for l in rl: out.append(l[:21]+ch+l[22:])
rp=f"outputs/design_md/{TAG}_chained.pdb"; open(rp,"w").write("\n".join(out)+"\nEND\n")

fx=PDBFixer(rp); fx.findMissingResidues(); fx.missingResidues={}
fx.findMissingAtoms(); fx.addMissingAtoms(); fx.addMissingHydrogens(7.0)
ff=ForceField("amber14-all.xml","implicit/gbn2.xml"); mod=Modeller(fx.topology,fx.positions)
system=ff.createSystem(mod.topology,nonbondedMethod=CutoffNonPeriodic,nonbondedCutoff=1.0*unit.nanometer,constraints=HBonds)
atoms=list(mod.topology.atoms()); pos_nm=mod.positions.value_in_unit(unit.nanometer)
rest=CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
rest.addGlobalParameter("k",(2.0*unit.kilocalories_per_mole/unit.angstrom**2).value_in_unit(unit.kilojoules_per_mole/unit.nanometer**2))
for p in ("x0","y0","z0"): rest.addPerParticleParameter(p)
for a in atoms:
    if a.residue.chain.id!="C" and a.element!=Element.getBySymbol("H"):
        x,y,z=pos_nm[a.index]; rest.addParticle(a.index,[x,y,z])
system.addForce(rest)
# scoring indices
caA=[a.index for a in atoms if a.residue.chain.id=="A" and a.name=="CA"]
fp_idx=[caA[i] for i in FP_POS if i<len(caA)]
pepca=[a.index for a in atoms if a.residue.chain.id=="C" and a.name=="CA"]; p9,p10=pepca[-2],pepca[-1]
mhc_res=[res_ for res_ in mod.topology.residues() if res_.chain.id=="A"]
mhc_seq="".join(AA3TO1.get(rr.name,'x') for rr in mhc_res)
k=_off(mhc_seq,REFSEQ); mo=[i for i in range(len(mhc_seq)) if 0<=i+k<len(REFSEQ) and mhc_seq[i]==REFSEQ[i+k]]
Pn=REF_CA[[i+k for i in mo]]
def score(pos):
    P=np.array(pos.value_in_unit(unit.angstrom))
    fc=P[fp_idx].mean(0); swap=float(np.linalg.norm(P[p10]-fc)-np.linalg.norm(P[p9]-fc))
    mca=P[caA]; R,t=_robust(mca[mo],Pn); pep=P[pepca]@R+t
    return swap, float(np.sqrt(((pep-GIG)**2).sum()/10)), float(np.sqrt(((pep-DRG)**2).sum()/10))

sim=Simulation(mod.topology,system,LangevinMiddleIntegrator(T_K*unit.kelvin,1/unit.picosecond,0.002*unit.picoseconds),Platform.getPlatformByName("OpenCL"))
sim.context.setPositions(mod.positions); sim.minimizeEnergy(maxIterations=300)
s0=score(sim.context.getState(getPositions=True).getPositions())
logln(f"[{TAG}] T={T_K}K NS={NS} Fpocket={len(fp_idx)}/6 | start swap={s0[0]:+.2f} toGIG={s0[1]:.2f} toDRG={s0[2]:.2f}")
nchunk=int(CHUNK_PS/0.002); maxch=int(NS*1000/CHUNK_PS); tr=[]
for c in range(maxch):
    sim.step(nchunk); sw,g,d=score(sim.context.getState(getPositions=True).getPositions())
    tr.append((round(sw,2),round(g,2),round(d,2)))
sw=[x[0] for x in tr]; g=[x[1] for x in tr]; d=[x[2] for x in tr]
logln(f"[{TAG}] swap trace={sw}")
logln(f"[{TAG}] toGIG trace={g}")
logln(f"[{TAG}] toDRG trace={d}")
logln(f"[{TAG}] ==> swap end {sw[-1]:+.2f} (min {min(sw):+.2f} max {max(sw):+.2f}) | toGIG end {g[-1]:.2f} (min {min(g):.2f}) | toDRG end {d[-1]:.2f} (min {min(d):.2f})")
