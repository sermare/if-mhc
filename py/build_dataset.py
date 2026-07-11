#!/usr/bin/env python
"""Assemble & validate a pMHC-TCR class I structure dataset for the IF protocol."""
import os, urllib.request, warnings, csv
from collections import OrderedDict
warnings.filterwarnings("ignore")
from Bio.PDB import PDBParser

# Curated class I TCR-pMHC complexes (mostly HLA-A*02:01 restricted)
CANDIDATES = ["2P5E","2P5W","1AO7","1QSF","1QRN","2BNR","2GJ6","2F53","2F54",
              "3QDG","3QEQ","3QFJ","3GSN","1OGA","3HG1","3UTS","3UTQ",
              "5C0A","5C0B","5HHO","5HHM","5EU6","2VLR","4MJI","5NME"]
AA3 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
       'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
DDIR = "inputs/pmhc_tcr_dataset"; os.makedirs(DDIR, exist_ok=True)
p = PDBParser(QUIET=True)

def chains_of(path):
    m = p.get_structure("x", path)[0]
    out = OrderedDict()
    for c in m:
        seq = "".join(AA3[r.resname] for r in c if r.id[0]==' ' and r.resname in AA3)
        if seq: out[c.id] = seq
    return out

def allele(path):
    for l in open(path):
        if l.startswith("COMPND") and ("HLA" in l or "HISTOCOMPATIBILITY" in l):
            return l[10:].strip().rstrip(";")[:40]
        if l.startswith("TITLE") and "HLA" in l: return "see title"
    return "?"

rows=[]
for pid in CANDIDATES:
    fp=f"{DDIR}/{pid}.pdb"
    if not os.path.exists(fp):
        try: urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pid}.pdb", fp)
        except Exception as e: print(f"{pid}: download FAIL {e}"); continue
    try: ch = chains_of(fp)
    except Exception as e: print(f"{pid}: parse FAIL {e}"); continue
    lens = {c:len(s) for c,s in ch.items()}
    # peptide = shortest chain in 8-15; need a long MHC (~250-300) and >=2 TCR-ish (>=180)
    peps = [(c,s) for c,s in ch.items() if 8<=len(s)<=15]
    mhc  = [c for c,n in lens.items() if 250<=n<=300]
    longs= [c for c,n in lens.items() if n>=180]
    ok = bool(peps) and bool(mhc) and len(longs)>=3   # MHC + TCRa + TCRb
    if not peps: print(f"{pid}: no peptide chain (lens={sorted(lens.values())}) -> skip"); continue
    pc, pseq = min(peps, key=lambda x:len(x[1]))
    rows.append({"pdb":pid,"valid":ok,"peptide":pseq,"pep_len":len(pseq),"pep_chain":pc,
                 "n_chains":len(ch),"chain_lens":",".join(f"{c}:{n}" for c,n in lens.items()),
                 "allele":allele(fp)})

with open(f"{DDIR}/dataset.csv","w",newline="") as o:
    w=csv.DictWriter(o,fieldnames=["pdb","valid","peptide","pep_len","pep_chain","n_chains","chain_lens","allele"])
    w.writeheader(); w.writerows(rows)

valid=[r for r in rows if r["valid"]]
print(f"\n=== pMHC-TCR dataset: {len(valid)}/{len(rows)} validated ===")
print(f"{'PDB':6}{'pep_chain':10}{'len':5}{'peptide':18}{'allele'}")
for r in sorted(valid,key=lambda x:x['pdb']):
    print(f"{r['pdb']:6}{r['pep_chain']:^10}{r['pep_len']:<5}{r['peptide']:18}{r['allele'][:30]}")
print(f"\nsaved -> {DDIR}/dataset.csv")
