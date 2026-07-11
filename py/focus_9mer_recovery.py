import glob,os,re,sys,numpy as np
# 9-mer epitope variants: DRG 9mer (paper-reported) and GIG 9mer
DRG9="MMWDRGLGM"; GIG9="SMLGIGIVP"
out=sys.argv[1]
from collections import defaultdict
g=defaultdict(list)
for fa in glob.glob(f"{out}/seqs/*.fa"):
    m=re.match(r"(6AM[U5])",os.path.basename(fa)); 
    if not m: continue
    pid=m.group(1)
    for i,l in enumerate(open(fa).read().splitlines()):
        if i%2==1 and len(l.strip())==9: g[pid].append(l.strip())
for pid in ["6AMU","6AM5"]:
    seqs=g.get(pid,[])
    if not seqs: continue
    P=np.array([list(s) for s in seqs])
    idrg=(P==np.array(list(DRG9))).mean(1)*100; igig=(P==np.array(list(GIG9))).mean(1)*100
    bdrg=seqs[int(idrg.argmax())]; bgig=seqs[int(igig.argmax())]
    print(f"\n=== {pid} de-novo 9mers (n={len(seqs)}) ===")
    print(f"  vs DRG9 {DRG9}: mean {idrg.mean():.1f}% max {idrg.max():.0f}% exact {int((idrg==100).sum())} best {bdrg}")
    print(f"  vs GIG9 {GIG9}: mean {igig.mean():.1f}% max {igig.max():.0f}% exact {int((igig==100).sum())} best {bgig}")
