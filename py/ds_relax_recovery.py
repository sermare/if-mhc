#!/usr/bin/env python
"""Relaxed-ensemble native recovery per structure; compare vs single-structure recovery."""
import glob, csv, sys, os, re
from collections import defaultdict
import numpy as np
out, dscsv = sys.argv[1], sys.argv[2]
native = {r["pdb"]: r["peptide"] for r in csv.DictReader(open(dscsv)) if r["valid"] == "True"}
single = {}
sp = "outputs/dataset_protocol/recovery.csv"
if os.path.exists(sp):
    single = {r["pdb"]: float(r["mean_id"]) for r in csv.DictReader(open(sp))}
# pool designs by structure (snapshot files: <pdb>_snapNN.fa)
bystruct = defaultdict(list)
for fa in glob.glob(f"{out}/seqs/*.fa"):
    pdb = os.path.basename(fa).split("_snap")[0]
    ls = open(fa).read().splitlines()
    for i in range(0, len(ls)-1, 2):
        h, s = ls[i], ls[i+1].strip()
        if "sample=" in h: bystruct[pdb].append(s)
rows = []
for pdb, seqs in bystruct.items():
    nat = native.get(pdb)
    seqs = [s for s in seqs if len(s) == len(nat)]
    if not nat or not seqs: continue
    P = np.array([list(s) for s in seqs]); na = np.array(list(nat))
    ident = (P == na).mean(1)*100
    rows.append({"pdb": pdb, "native": nat, "n": len(seqs),
                 "relaxed_mean_id": round(ident.mean(),1), "relaxed_max_id": round(ident.max(),1),
                 "single_mean_id": single.get(pdb, ""), "has_Pro_native": "P" in nat})
rows.sort(key=lambda r: -r["relaxed_mean_id"])
with open(f"{out}/recovery_relaxed.csv", "w", newline="") as o:
    w = csv.DictWriter(o, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"{'PDB':6}{'native':12}{'relax%':8}{'single%':9}{'Δ':7}{'natPro'}")
for r in rows:
    d = (f"{r['relaxed_mean_id']-r['single_mean_id']:+.1f}" if r['single_mean_id'] != "" else "")
    print(f"{r['pdb']:6}{r['native']:12}{r['relaxed_mean_id']:<8}{str(r['single_mean_id']):<9}{d:<7}{'Y' if r['has_Pro_native'] else ''}")
if rows: print(f"\nrelaxed-ensemble mean recovery: {np.mean([r['relaxed_mean_id'] for r in rows]):.1f}% over {len(rows)} structures")
