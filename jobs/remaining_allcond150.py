#!/usr/bin/env python3
"""Print remaining allcond150 work: per-cell (have/target) and total remaining.
Exit code 0 if work remains, 3 if all cells at target (campaign complete)."""
import glob, os, sys
ABS="/global/scratch/users/sergiomar10/if-mhc"
SPEC=f"{ABS}/jobs/allcond150_spec.tsv"; PDB=f"{ABS}/outputs/allcond150/pdb"
total_rem=0; total_have=0; total_tgt=0; lines=[]
for ln in open(SPEC):
    p=ln.rstrip("\n").split("\t")
    if len(p)<4 or not p[0]: continue
    x,cond,style,tgt=p[0],p[1],p[2],int(p[3])
    have=len([f for f in glob.glob(f"{PDB}/{x}_{cond}_j*.pdb") if "traj" not in f])
    rem=max(0,tgt-have)
    total_rem+=rem; total_have+=have; total_tgt+=tgt
    if rem>0: lines.append(f"  {x}_{cond}: {have}/{tgt} (rem {rem})")
print(f"remaining={total_rem} have={total_have}/{total_tgt}")
for l in lines: print(l)
sys.exit(3 if total_rem==0 else 0)
