#!/usr/bin/env python3
"""Write a FULL-ATOM, common-frame-aligned, properly-chain-split copy of a design or native PDB,
for real structural rendering (PyMOL). Reuses the exact MHC->REF_CA superposition from
score_denovo_designs._map_peptide, but applies R,t to every atom (not just peptide CA), and splits
RFdiffusion's single merged receptor chain back into MHC/B2M/TCRa/TCRb using the design's own .trb
contig (so cartoon rendering doesn't draw false bonds across chain junctions).

Usage:
    python py/align_full_pdb.py <path/to/design_or_native.pdb> <out.pdb>
"""
import sys, os, re
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import score_denovo_designs as S

CHAIN_LABELS = ["A", "B", "D", "E"]  # MHC, B2M, TCRa, TCRb (native convention)

def read_atoms(path):
    return [l for l in open(path) if l.startswith(("ATOM", "HETATM"))]

def contig_lengths(trb_path):
    """[(is_receptor, length), ...] in contig order, from the .trb config."""
    d = np.load(trb_path, allow_pickle=True)
    contig = d["config"]["contigmap"]["contigs"][0]
    out = []
    for tok in contig.split():
        m = re.match(r"^[A-Za-z](\d+)-(\d+)(/\d+)?$", tok)
        if m:
            out.append((True, int(m.group(2)) - int(m.group(1)) + 1)); continue
        m = re.match(r"^(\d+)-(\d+)$", tok)
        if m:
            out.append((False, int(m.group(2)) - int(m.group(1)) + 1)); continue
    return out

def split_and_relabel(lines, trb_path):
    """Design output: chain A=peptide(10), chain B=merged receptor. Split B by contig lengths,
    relabel to peptide='P', receptor sub-chains -> CHAIN_LABELS, renumber each from 1."""
    segs = contig_lengths(trb_path)
    order, plen = [], 10
    for is_recept, n in segs:
        order.append(("recept", n) if is_recept else ("pep", n))
    by_chain = {}
    for l in lines:
        by_chain.setdefault(l[21], []).append(l)
    pep_lines = by_chain.get("A", [])
    recept_lines = by_chain.get("B", [])
    # walk receptor residues in author order, cut into segments per contig_lengths receptor tokens
    recept_res_order = []
    seen = set()
    for l in recept_lines:
        rn = int(l[22:26])
        if rn not in seen:
            seen.add(rn); recept_res_order.append(rn)
    recept_res_order.sort()
    cuts, i = [], 0
    for is_recept, n in segs:
        if not is_recept:
            continue
        cuts.append(recept_res_order[i:i + n]); i += n
    out = []
    for l in pep_lines:
        rn = int(l[22:26])
        out.append(l[:21] + "P" + f"{rn:4d}" + l[26:])
    for label, resnums in zip(CHAIN_LABELS, cuts):
        rmin = min(resnums)
        for l in recept_lines:
            rn = int(l[22:26])
            if rn in resnums:
                newrn = rn - rmin + 1
                out.append(l[:21] + label + f"{newrn:4d}" + l[26:])
    return out

def transform_lines(lines, R, t):
    out = []
    for l in lines:
        x, y, z = float(l[30:38]), float(l[38:46]), float(l[46:54])
        v = np.array([x, y, z]) @ R + t
        out.append(l[:30] + f"{v[0]:8.3f}{v[1]:8.3f}{v[2]:8.3f}" + l[54:])
    return out

def get_alignment(path):
    """Reuse _map_peptide's exact MHC-locate + robust-Kabsch to get (R,t) into the REF_CA groove frame."""
    ch = S._chains(path)
    mhc = None
    for c, rs in ch.items():
        s = "".join(n for n, _ in rs)
        if S.MHC_MOTIF in s:
            mhc = (c, rs, s); break
    if not mhc:
        raise RuntimeError(f"MHC not found in {path}")
    c, rs, ms = mhc
    mca = np.array([x for _, x in rs])
    m = ms.find(S.MHC_MOTIF)
    mhc_local = list(range(m, min(m + 179, len(rs))))
    k = S._offset("".join(ms[i] for i in mhc_local), S.REFSEQ)
    idx = [j for j, i in enumerate(mhc_local) if 0 <= j + k < len(S.REF_CA)]
    R, t = S._robust(mca[[mhc_local[j] for j in idx]], S.REF_CA[[j + k for j in idx]])
    return R, t

def main():
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    path, out = sys.argv[1], sys.argv[2]
    R, t = get_alignment(path)
    lines = read_atoms(path)
    trb = path[:-4] + ".trb"
    if os.path.exists(trb):
        lines = split_and_relabel(lines, trb)  # RFd output: merged receptor chain -> split A/B/D/E, peptide->P
    else:
        lines = [l[:21] + "P" + l[22:] if l[21] == "C" else l for l in lines]  # native: peptide chain C -> P
    lines = transform_lines(lines, R, t)
    with open(out, "w") as f:
        f.writelines(lines)
        f.write("END\n")
    print(out)

if __name__ == "__main__":
    main()
