import glob,os,re,sys,numpy as np
from collections import defaultdict
DRG="MMWDRGLGMM"; GIG="SMLGIGIVPV"
PANEL=['ELAGIGILTV','SMLGIGIVPV','NMGGLGIMPV','ILEDRGFNQV','LMFDRGMSLL','MMWDRGLGMM','MMWDRGMGLL','SMAGIGIVDV','IMEDVGWLNV']
NC={}  # (pid,level)->ncontacts
for l in open("inputs/focus_6am/ladder_spec.tsv"):
    p,n,nc,_,_=l.rstrip("\n").split("\t"); NC[(p,n)]=int(nc)
out=sys.argv[1]
g=defaultdict(list); nll=defaultdict(list)
for fa in glob.glob(f"{out}/seqs/*.fa"):
    m=re.match(r"(6AM[U5])_(L\d_[a-z0-9_]+)_\d+\.fa", os.path.basename(fa))
    if not m: continue
    pid,lvl=m.group(1),m.group(2)
    ls=open(fa).read().splitlines()
    for i in range(0,len(ls)-1,2):
        if "sample=" in ls[i] and len(ls[i+1].strip())==10:
            g[(pid,lvl)].append(ls[i+1].strip())
            sm=re.search(r"score=([-\d.]+)",ls[i]); nll[(pid,lvl)].append(float(sm.group(1)) if sm else np.nan)
def mx(seqs,ref):
    P=np.array([list(x) for x in seqs]); return float((P==np.array(list(ref))).mean(1).max()*100)
def bestpanel(seqs):
    return max(mx(seqs,p) for p in PANEL)
print(f"{'crystal':8}{'level':14}{'#contacts':10}{'n':6}{'%uniq':7}{'medNLL':8}{'max>own':9}{'max>panel':10}")
for (pid,lvl) in sorted(g,key=lambda k:(k[0],NC.get(k,0))):
    s=g[(pid,lvl)]; own=DRG if pid=="6AMU" else GIG
    a=np.array([v for v in nll[(pid,lvl)] if not np.isnan(v)])
    print(f"{pid:8}{lvl:14}{NC.get((pid,lvl),'?'):<10}{len(s):<6}{100*len(set(s))/len(s):<7.0f}"
          f"{np.median(a) if len(a) else float('nan'):<8.3f}{mx(s,own):<9.0f}{bestpanel(s):<10.0f}")
print("\nQuestion: does increasing # conditioned contacts raise recovery / lower NLL / change the motif?")
