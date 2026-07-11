import glob,sys,numpy as np
NAT={"6AMT":"MMWDRGLGMM","6AMU":"MMWDRGLGMM","6AM5":"SMLGIGIVPV"}
out=sys.argv[1]
for fa in sorted(glob.glob(f"{out}/seqs/*.fa")):
    pid=fa.split("/")[-1].replace(".fa","")
    nat=NAT.get(pid); 
    if not nat: continue
    seqs=[l.strip() for i,l in enumerate(open(fa).read().splitlines()) if i%2==1]
    seqs=[s for s in seqs if len(s)==len(nat)]
    if not seqs: continue
    P=np.array([list(s) for s in seqs]); na=np.array(list(nat))
    ident=(P==na).mean(1)*100; pp=(P==na).mean(0)*100
    best=seqs[int(np.argmax(ident))]
    print(f"\n=== {pid}  native={nat} (n={len(seqs)}) ===")
    print(f"  mean {ident.mean():.1f}%  max {ident.max():.1f}%  exact {int((ident==100).sum())}")
    print(f"  best: {best}  ({(np.array(list(best))==na).mean()*100:.0f}%)")
    print("  per-pos: "+" ".join(f"{nat[i]}{pp[i]:.0f}" for i in range(len(nat))))
