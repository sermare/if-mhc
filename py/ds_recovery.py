#!/usr/bin/env python
"""Per-structure native recovery for the dataset MPNN designs."""
import glob, csv, sys, re
import numpy as np
out, dscsv = sys.argv[1], sys.argv[2]
native = {r["pdb"]: r["peptide"] for r in csv.DictReader(open(dscsv)) if r["valid"]=="True"}
rows = []
for fa in sorted(glob.glob(f"{out}/seqs/*.fa")):
    pid = fa.split("/")[-1].replace(".fa", "")
    nat = native.get(pid)
    if not nat: continue
    seqs = []
    ls = open(fa).read().splitlines()
    for i in range(0, len(ls)-1, 2):
        h, s = ls[i], ls[i+1].strip()
        if "sample=" in h and len(s) == len(nat): seqs.append(s)
    if not seqs: continue
    P = np.array([list(s) for s in seqs]); na = np.array(list(nat))
    ident = (P == na).mean(1)*100
    perpos = (P == na).mean(0)*100
    best = seqs[int(np.argmax(ident))]
    rows.append({"pdb": pid, "native": nat, "len": len(nat), "n": len(seqs),
                 "mean_id": round(ident.mean(),1), "max_id": round(ident.max(),1),
                 "exact": int((ident==100).sum()),
                 "best_design": best, "has_Pro_native": "P" in nat})
rows.sort(key=lambda r: -r["mean_id"])
if rows:
    with open(f"{out}/recovery.csv","w",newline="") as o:
        w=csv.DictWriter(o, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"{'PDB':6}{'native':12}{'len':4}{'mean%':7}{'max%':6}{'exact':6}{'best_design':13}{'natPro'}")
for r in rows:
    print(f"{r['pdb']:6}{r['native']:12}{r['len']:<4}{r['mean_id']:<7}{r['max_id']:<6}{r['exact']:<6}{r['best_design']:13}{'Y' if r['has_Pro_native'] else ''}")
if rows:
    print(f"\nmean across dataset: {np.mean([r['mean_id'] for r in rows]):.1f}%  | structures: {len(rows)}")
