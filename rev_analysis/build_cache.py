import sys, glob, os
sys.path.insert(0, "/global/scratch/users/sergiomar10/if-mhc/rev_analysis")
import numpy as np, pandas as pd
import score_sd as S
ROOT="/global/scratch/users/sergiomar10/if-mhc"
def cell(p):
    b=os.path.basename(p); cry="6AM5" if b.startswith("6AM5") else "6AMU"
    return cry,b.split(cry+"_",1)[1].split("_j")[0]
def grp(c):
    if c=="null0": return "null"
    if c.startswith("fix"): return "templated"
    if c.startswith("xo_"): return "xover"
    return "denovo"
rows=[]; coords=[]
dirs=("outputs/allcond150/pdb","outputs/null_extra300/pdb","outputs/xover/pdb")
files=[f for d in dirs for f in glob.glob(f"{ROOT}/{d}/*.pdb") if "traj" not in f]
for i,f in enumerate(files):
    r=S._map_peptide(f)
    if r is None: continue
    pa,tcr,pl=r
    if pa is None or len(pa)!=10: continue
    cry,cond=cell(f); sc=S.occupancy(pa)
    e2e=float(np.linalg.norm(pa[0]-pa[9]))                 # end-to-end Ca distance
    rows.append(dict(cry=cry,cond=cond,g=grp(cond),
        toGIG=float(np.sqrt(((pa-S.GIG)**2).sum()/10)),
        toDRG=float(np.sqrt(((pa-S.DRG)**2).sum()/10)),
        fpos=sc["fpocket_pos"],fdist=sc["fpocket_dist"],thread=sc["threading"],
        e2e=e2e,file=os.path.basename(f)))
    coords.append(pa.astype(np.float32))
    if i%2000==0: print(f"  {i}/{len(files)}",flush=True)
DF=pd.DataFrame(rows)
C=np.stack(coords)                                          # (N,10,3)
DF.to_pickle("/global/scratch/users/sergiomar10/if-mhc/rev_analysis/design_cache.pkl")
np.savez_compressed("/global/scratch/users/sergiomar10/if-mhc/rev_analysis/design_coords.npz",
                    coords=C, GIG=S.GIG, DRG=S.DRG)
# native reference end-to-end for the funnel
print(f"cached N={len(DF)}  GIG e2e={np.linalg.norm(S.GIG[0]-S.GIG[9]):.2f}  DRG e2e={np.linalg.norm(S.DRG[0]-S.DRG[9]):.2f}")
print("groups:", DF.g.value_counts().to_dict())
print("CACHE_DONE")
