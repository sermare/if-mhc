#!/usr/bin/env python3
"""Prototype the data-loading for the 3 genuine matched vanilla/noMHC pairs found by
scan_mpnn_provenance.py, before porting into the notebook. Just prints sanity-check stats."""
import re, glob
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("/home/ubuntu/if-mhc")
L = 10
AA = list("ACDEFGHIKLMNPQRSTVWY")

def load_fasta(path, L=L):
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["peptide","score","global_score","seq_recovery","T","sample"])
    lines = p.read_text().splitlines()
    rows = []
    ref = None
    for i in range(0, len(lines)-1, 2):
        h, s = lines[i], lines[i+1].strip()
        if not h.startswith(">") or len(s) != L:
            continue
        if "sample=" not in h:
            ref = s   # reference/original sequence baked into this backbone
            continue
        d = dict(re.findall(r'(\w+)=([-\d.]+)', h))
        rows.append({"peptide": s, "score": float(d.get("score","nan")),
                     "global_score": float(d.get("global_score","nan")),
                     "seq_recovery": float(d.get("seq_recovery","nan")),
                     "T": float(d.get("T","nan")), "sample": int(float(d.get("sample",0)))})
    return pd.DataFrame(rows), ref

# ---- native pair ----
NAT = {"6AM5": "SMLGIGIVPV", "6AMU": "MMWDRGLGMM"}
for pid in ["6AM5","6AMU"]:
    dv, refv = load_fasta(ROOT/f"outputs/focus_6am_50k/seqs/{pid}.fa")
    dn, refn = load_fasta(ROOT/f"outputs/mpnn_nomhc_topcross_50k/seqs/nat_{pid}.fa")
    print(f"native {pid}: vanilla n={len(dv)} ref={refv}  |  noMHC n={len(dn)} ref={refn}  |  true_native={NAT[pid]}")

# ---- ladder + grind matched targets, via manifest ----
manifest = pd.read_csv(ROOT/"outputs/mpnn_nomhc_allbb/manifest.csv")
for campaign in ["ladder","grind"]:
    sub = manifest[manifest.src.str.contains(f"outputs/{campaign}/pdb", na=False)]
    print(f"\n{campaign}: {len(sub)} matched targets")
    row = sub.iloc[0]
    name = Path(row.src).stem
    bbid = row.target
    dv, refv = load_fasta(ROOT/f"outputs/{campaign}/seqs/{name}.fa")
    dn, refn = load_fasta(ROOT/f"outputs/mpnn_nomhc_allbb_deep2/seqs/{bbid}.fa")
    print(f"  sample target={name} (bbid={bbid}): vanilla n={len(dv)} ref={refv}  |  noMHC n={len(dn)} ref={refn}")
    print(f"  refv==refn: {refv==refn}  (should be True/near-true -- same backbone, same baked-in original sequence)")
