import glob,os,re,sys
import numpy as np
from collections import defaultdict
DRG="MMWDRGLGMM"; GIG="SMLGIGIVPV"
out=sys.argv[1]
g=defaultdict(list)  # (pid,ablation) -> peptide seqs
for fa in glob.glob(f"{out}/seqs/*.fa"):
    m=re.match(r"(6AM[U5])_(.+)_(\d+)\.fa", os.path.basename(fa))
    if not m: continue
    pid, abl = m.group(1), m.group(2)
    for i,l in enumerate(open(fa).read().splitlines()):
        if i%2==1 and len(l.strip())==10: g[(pid,abl)].append(l.strip())

def ids(seqs,ref):
    P=np.array([list(s) for s in seqs]); r=np.array(list(ref)); idt=(P==r).mean(1)*100
    return idt.mean(), idt.max(), int((idt==100).sum())

ORDER=["tcr1","tcr2","mhc","mhc_tcr1","mhc_tcr2"]
print(f"{'crystal':8}{'conditioning':12}{'n':6}{'->DRG mean/max':16}{'->GIG mean/max':16}{'leans'}")
for pid in ["6AMU","6AM5"]:
    nat = DRG if pid=="6AMU" else GIG
    for abl in ORDER:
        seqs=g.get((pid,abl),[])
        if not seqs: continue
        dm,dx,de=ids(seqs,DRG); gm,gx,ge=ids(seqs,GIG)
        lean="DRG" if dm>gm+2 else ("GIG" if gm>dm+2 else "~tie")
        own="(own)" if (pid=="6AMU" and lean=="DRG") or (pid=="6AM5" and lean=="GIG") else ""
        print(f"{pid:8}{abl:12}{len(seqs):<6}{f'{dm:.1f}/{dx:.0f}':16}{f'{gm:.1f}/{gx:.0f}':16}{lean} {own}")
    print()
print("Question: does adding TCR conditioning steer generations toward each crystal's OWN peptide,")
print("and do the two crystals (same DMF5 TCR) generate DIFFERENT peptides under matched conditioning?")
