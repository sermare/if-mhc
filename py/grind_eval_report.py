import glob,os,re,sys,numpy as np
from collections import defaultdict
DRG="MMWDRGLGMM"; GIG="SMLGIGIVPV"
out=sys.argv[1]
def idto(seqs,ref):
    if not seqs: return (0,0,0)
    P=np.array([list(s) for s in seqs]); r=np.array(list(ref[:len(seqs[0])])) if len(ref)>=len(seqs[0]) else None
    # only compare when lengths match
    return None
g=defaultdict(list)
for fa in glob.glob(f"{out}/seqs/*.fa"):
    m=re.match(r"(6AM[U5])_([a-z0-9_]+)_L(\d+)_w\d+_\d+", os.path.basename(fa))
    if not m: continue
    pid,cond,L=m.group(1),m.group(2),int(m.group(3))
    for i,l in enumerate(open(fa).read().splitlines()):
        if i%2==1 and len(l.strip())==L: g[(pid,cond,L)].append(l.strip())
def maxid(seqs,ref):
    seqs=[s for s in seqs if len(s)==len(ref)]
    if not seqs: return (0.0,0,None)
    P=np.array([list(s) for s in seqs]); r=np.array(list(ref)); idt=(P==r).mean(1)*100
    return (idt.max(), int((idt>=70).sum()), seqs[int(idt.argmax())])
tot=sum(len(v) for v in g.values())
print(f"[{__import__('datetime').datetime.utcnow().isoformat()}Z] EVAL: {tot} peptide designs over {len(g)} combos")
# overall best toward each native (length-matched: DRG/GIG are 10-mers)
allseq=[s for v in g.values() for s in v]
for nat,name in [(DRG,"DRG"),(GIG,"GIG")]:
    mx,n70,best=maxid(allseq,nat)
    print(f"   ->{name} {nat}: best={mx:.0f}%  (#>=70%: {n70})  closest={best}")
# per length: best toward DRG and GIG (10-mers only meaningful for full match; report any length's best core)
for (pid,cond,L) in sorted(g):
    s=g[(pid,cond,L)]
    if L!=10: continue
    md,_,_=maxid(s,DRG); mg,_,_=maxid(s,GIG)
    if max(md,mg)>=60: print(f"   {pid}/{cond}/L{L} (n={len(s)}): DRG {md:.0f}% | GIG {mg:.0f}%")
