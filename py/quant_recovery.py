#!/usr/bin/env python
"""Quantify ProteinMPNN native-sequence recovery for the 2P5E peptide (SLLMWITQC)."""
import re, glob, sys
import numpy as np
NATIVE = "SLLMWITQC"; L = len(NATIVE); AA = list("ACDEFGHIKLMNPQRSTVWY")

def load(path):
    seqs = []
    try: ls = open(path).read().splitlines()
    except FileNotFoundError: return seqs, []
    Ts = []
    for i in range(0, len(ls)-1, 2):
        h, s = ls[i], ls[i+1].strip()
        if "sample=" not in h or len(s) != L: continue
        seqs.append(s)
        m = re.search(r"T=([\d.]+)", h); Ts.append(float(m.group(1)) if m else None)
    return seqs, Ts

def report(name, seqs, Ts=None):
    if not seqs: print(f"\n[{name}] no data"); return
    P = np.array([list(s) for s in seqs]); nat = np.array(list(NATIVE))
    ident = (P == nat).mean(1)*100
    exact = int((ident == 100).sum())
    perpos = (P == nat).mean(0)*100
    print(f"\n===== {name}  (n={len(seqs):,}) =====")
    print(f"  exact native recoveries : {exact}  ({100*exact/len(seqs):.4f}%)")
    print(f"  identity to native      : mean {ident.mean():.1f}%  median {np.median(ident):.1f}%  max {ident.max():.1f}%")
    print(f"  per-position recovery   : " + "  ".join(f"P{i+1}({NATIVE[i]}){perpos[i]:.0f}%" for i in range(L)))
    # best designs by identity
    order = np.argsort(ident)[::-1][:3]
    print(f"  closest designs         : " + ", ".join(f"{seqs[i]}({ident[i]:.0f}%)" for i in order))
    if Ts and any(t is not None for t in Ts):
        print("  recovery by temperature :")
        Ts = np.array([t if t is not None else -1 for t in Ts])
        for t in sorted(set(Ts[Ts>0])):
            m = Ts == t
            print(f"     T={t}: mean identity {ident[m].mean():.1f}%  per-pos anchors P2={ (P[m,1]==nat[1]).mean()*100:.0f}%  P9={(P[m,8]==nat[8]).mean()*100:.0f}%")

# 1) full-complex 50K (with M/P, single T=0.3)
s,_ = load("outputs/mpnn_50k/seqs/2P5E.fa"); report("Full-complex 50K (with M,P; T=0.3)", s)
# 2) 100K no-M no-P, 7 temps
s,T = load("outputs/mpnn_100k_7temp/seqs/2P5E.fa"); report("100K (no Met@P1, no Pro; 7 temps)", s, T)
# 3) MHC-only
s,_ = load("outputs/mpnn_mhconly_v20/seqs/2P5E_ABC.fa"); report("MHC-only 50K", s)
# 4) relax ensemble (pooled)
pool=[]
for fa in glob.glob("outputs/relax_campaign/seqs/*.fa"):
    ps,_ = load(fa); pool += ps
report("Relax ensemble (no Met@P1, no Pro; pooled)", pool)
