import glob, os, re, sys
import numpy as np
NAT={"6AMU":"MMWDRGLGMM","6AM5":"SMLGIGIVPV"}
OTHER={"6AMU":"SMLGIGIVPV","6AM5":"MMWDRGLGMM"}
out=sys.argv[1]
from collections import defaultdict
g=defaultdict(list)
for fa in glob.glob(f"{out}/seqs/*.fa"):
    m=re.match(r"(6AM[U5])_pT(\d+)", os.path.basename(fa))
    if not m: continue
    pid,pT=m.group(1),int(m.group(2))
    for i,l in enumerate(open(fa).read().splitlines()):
        if i%2==1 and len(l.strip())==10: g[(pid,pT)].append(l.strip())
print(f"{'struct':7}{'partial_T':10}{'n':6}{'own%':8}{'ownmax':8}{'other%':8}{'gap'}")
for (pid,pT),seqs in sorted(g.items()):
    P=np.array([list(s) for s in seqs])
    io=(P==np.array(list(NAT[pid]))).mean(1)*100
    ix=(P==np.array(list(OTHER[pid]))).mean(1)*100
    print(f"{pid:7}{pT:<10}{len(seqs):<6}{io.mean():<8.1f}{io.max():<8.0f}{ix.mean():<8.1f}{io.mean()-ix.mean():+.1f}")
print("\nGap stays large => structure-specific recovery survives diffusion perturbation.")
