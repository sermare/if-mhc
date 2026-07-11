import glob,sys,os,numpy as np
from collections import defaultdict
# Both peptides are 10-mers -> own-vs-other cross-specificity is position-wise comparable.
DRG="MMWDRGLGMM"; GIG="SMLGIGIVPV"
NAT  ={"6AMU":DRG,"6AM5":GIG,"6AMT":DRG}        # own peptide
OTHER={"6AMU":GIG,"6AM5":DRG,"6AMT":GIG}        # the other peptide
TCR  ={"6AMU":"+TCR","6AM5":"+TCR","6AMT":"no-TCR(ctrl)"}
out=sys.argv[1]
by=defaultdict(list)
for fa in glob.glob(f"{out}/seqs/*.fa"):
    pid=os.path.basename(fa).split("_snap")[0]
    for i,l in enumerate(open(fa).read().splitlines()):
        if i%2==1: by[pid].append(l.strip())
print(f"{'struct':6}{'TCR':14}{'own':12}{'id->OWN':9}{'id->OTHER':11}{'gap':7}{'exact'}")
for pid in ["6AMU","6AM5","6AMT"]:        # TCR-bound pair first, no-TCR control last
    own=NAT[pid]; oth=OTHER[pid]
    seqs=[s for s in by.get(pid,[]) if len(s)==len(own)]
    if not seqs: continue
    P=np.array([list(s) for s in seqs])
    io=(P==np.array(list(own))).mean(1)*100
    ix=(P==np.array(list(oth))).mean(1)*100
    print(f"{pid:6}{TCR[pid]:14}{own:12}{io.mean():<9.1f}{ix.mean():<11.1f}{io.mean()-ix.mean():<+7.1f}{int((io==100).sum())}")
print("\nKey = 6AMU vs 6AM5 (same DMF5 TCR, different peptides).")
print("Gap (own-other) staying large under relaxation => TCR-bound specificity is robust, not crystal-locked.")
