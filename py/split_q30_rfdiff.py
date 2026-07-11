#!/usr/bin/env python3
"""Re-split Q30 RFdiffusion partial-diffusion outputs into proper chains.

RFdiffusion merges the whole complex into one output chain in contig order. Partial diffusion
preserves that order, so we remap the sequential output residues back onto the seed's contig
segments (A=MHC, B=b2m, C=peptide, D=TCR Va, E=TCR Vb) with original author numbering restored,
using the per-seed contig from outputs/rfdiff_q30/seeds_manifest.tsv.

Writes <name>_T<T>_<idx>_split.pdb next to each raw output. Idempotent; skips existing splits.
"""
import os, re, sys, glob, csv
ROOT = "/home/ubuntu/if-mhc"
OUT  = f"{ROOT}/outputs/rfdiff_q30"
FNAME = re.compile(r"^(?P<seed>.+)_T(?P<T>\d+)_(?P<idx>\d+)\.pdb$")
SEG = re.compile(r"([A-Z])(-?\d+)-(-?\d+)")


def load_segments():
    """seed name -> [(chain, lo, hi), ...] in contig order."""
    segs = {}
    for r in csv.DictReader(open(f"{OUT}/seeds_manifest.tsv"), delimiter="\t"):
        segs[r["name"]] = [(c, int(lo), int(hi)) for c, lo, hi in SEG.findall(r["contig"])]
    return segs


def split_file(inp, outp, segments):
    # ordered unique residues (any chain) in file order
    order, seen = [], set()
    for l in open(inp):
        if l.startswith(("ATOM", "HETATM")):
            key = (l[21], l[22:27])                       # chain + resSeq(+icode)
            if key not in seen:
                seen.add(key); order.append(key)
    total = sum(hi - lo + 1 for _, lo, hi in segments)
    if len(order) != total:
        return f"residue count {len(order)} != contig total {total}"
    # map each original (chain,resSeq) -> (newchain, newresnum)
    m, i = {}, 0
    for ch, lo, hi in segments:
        for n in range(lo, hi + 1):
            m[order[i]] = (ch, n); i += 1
    out = []
    for l in open(inp):
        if l.startswith(("ATOM", "HETATM")):
            key = (l[21], l[22:27])
            ch, nr = m[key]
            out.append(l[:21] + ch + f"{nr:>4d} " + l[27:])   # col22 chain, 23-26 resSeq, 27 icode->space
        elif l.startswith("TER"):
            out.append(l)
    open(outp, "w").writelines(out)
    return None


def main():
    segs = load_segments()
    raw = [f for f in sorted(glob.glob(f"{OUT}/*_T*_*.pdb")) if "_split" not in f]
    ok, skip, err = 0, 0, 0
    for f in raw:
        m = FNAME.match(os.path.basename(f))
        if not m:
            continue
        outp = f[:-4] + "_split.pdb"
        if os.path.exists(outp):
            skip += 1; continue
        seed = m["seed"]
        if seed not in segs:
            print(f"  no manifest segments for seed {seed}"); err += 1; continue
        e = split_file(f, outp, segs[seed])
        if e:
            print(f"  ERR {os.path.basename(f)}: {e}"); err += 1
        else:
            ok += 1
    print(f"split: {ok} written, {skip} existing, {err} errors  ({len(raw)} raw outputs)")


if __name__ == "__main__":
    main()
