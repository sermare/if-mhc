#!/usr/bin/env python3
"""Check whether any TCR clone in /home/ubuntu/adimab/tcrs.csv (proprietary Adimab data -- read
column-wise for the comparison only, never dumped wholesale) has a CDR3 (alpha or beta) that appears
as a substring of any protein chain extracted from the crystal structures in
inputs/pmhc_tcr_dataset/*.pdb.

NOTE: an earlier version of this script typed candidate TCR chains via this project's
TCRA_MOTIF/TCRB_MOTIF constant-region motifs (score_denovo_designs.py). Those motifs are specific to
this project's own TCR construct and do NOT generalize to this diverse reference set (many entries
are different clonotypes/alleles, at least one non-human) -- verified directly: e.g. 3HG1's chain E
(a real TCR beta chain) does not contain TCRB_MOTIF at all, which silently dropped a genuine match.
This version checks every chain >=50 residues (excludes only the short peptide chain) with no typing
assumption, so nothing is silently filtered out by a motif that doesn't apply here.
"""
import sys, glob, re
sys.path.insert(0, "/home/ubuntu/if-mhc/py")
import pandas as pd
import score_denovo_designs as S

AA3to1 = S.AA  # reuse this project's 3-letter->1-letter table

def chain_seqs(path):
    """Return {chain_id: sequence} for every protein chain in a PDB (CA-trace order)."""
    order = {}
    for l in open(path):
        if l.startswith("ATOM") and l[12:16].strip() == "CA":
            c = l[21]
            order.setdefault(c, []).append(AA3to1.get(l[17:20].strip(), "x"))
    return {c: "".join(v) for c, v in order.items()}

# ---- load Adimab TCR clones (column-wise only; never print full rows) ----
tcrs = pd.read_csv("/home/ubuntu/adimab/tcrs.csv", sep="\t")
clones = tcrs["Clone"].tolist()
cdr3_a = tcrs["CDR3"].tolist()      # first chain's CDR3
cdr3_b = tcrs["CDR3.1"].tolist()    # second chain's CDR3
print(f"{len(clones)} Adimab TCR clones loaded (CDR3 columns only, not full rows)")

# ---- extract TCR chains from every structure in the reference dataset ----
pdb_dir = "/home/ubuntu/if-mhc/inputs/pmhc_tcr_dataset"
pdbs = sorted(glob.glob(f"{pdb_dir}/*.pdb"))
print(f"{len(pdbs)} reference pMHC-TCR structures to scan")

hits = []
for path in pdbs:
    pdbid = path.split("/")[-1].replace(".pdb", "")
    seqs = chain_seqs(path)
    for clone, ca, cb in zip(clones, cdr3_a, cdr3_b):
        for cdr3, which in [(ca, "CDR3"), (cb, "CDR3.1")]:
            if not isinstance(cdr3, str) or len(cdr3) < 4:
                continue
            for chain_id, seq in seqs.items():
                if len(seq) < 50:   # only excludes the short peptide chain; no TCR-typing assumption
                    continue
                if cdr3 in seq:
                    hits.append({"clone": clone, "adimab_column": which, "cdr3_len": len(cdr3),
                                  "pdb": pdbid, "chain": chain_id, "chain_len": len(seq)})

print(f"\n{len(hits)} substring matches found")
if hits:
    hdf = pd.DataFrame(hits)
    print(hdf.to_string(index=False))
    hdf.to_csv("/home/ubuntu/if-mhc/outputs/analysis/adimab_tcr_overlap.csv", index=False)
    print("\nsaved outputs/analysis/adimab_tcr_overlap.csv")
else:
    print("No overlap between Adimab tcrs.csv CDR3 sequences and inputs/pmhc_tcr_dataset structures.")
