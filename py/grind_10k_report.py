import glob,os,re,sys,numpy as np
from collections import defaultdict
DRG="MMWDRGLGMM"; GIG="SMLGIGIVPV"; DRG9="MMWDRGLGM"; GIG9="SMLGIGIVP"
out=sys.argv[1]
for pid in ["6AMU","6AM5"]:
    bylen=defaultdict(list)
    for fa in glob.glob(f"{out}/{pid}/seqs/*.fa"):
        for i,l in enumerate(open(fa).read().splitlines()):
            if i%2==1:
                s=l.strip(); bylen[len(s)].append(s)
    tot=sum(len(v) for v in bylen.values())
    print(f"\n=== {pid}  total={tot} seqs  (T=0.7, across {len(glob.glob(f'{out}/{pid}/seqs/*.fa'))} backbones) ===")
    for L in sorted(bylen):
        s=bylen[L]; u=len(set(s))
        line=f"  L{L}: n={len(s):<5} uniq={u:<5} ({100*u/len(s):.0f}%)"
        for ref,nm in ([(DRG,"DRG"),(GIG,"GIG")] if L==10 else [(DRG9,"DRG9"),(GIG9,"GIG9")] if L==9 else []):
            P=np.array([list(x) for x in s]); idt=(P==np.array(list(ref))).mean(1)*100
            line+=f"  ->{nm} max={idt.max():.0f}% n>=70={int((idt>=70).sum())} n>=60={int((idt>=60).sum())}"
        print(line)
print("\nT=0.7 vs prior T=0.1: expect much higher %unique; watch for any new >=70% DRG/GIG hits.")
