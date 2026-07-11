#!/usr/bin/env python
"""Re-split RFdiffusion output (peptide=A, all target merged into B) back into proper chains.
Our trimmed motif order in chain B is: MHC a1a2 (180) -> TCR Va (116) -> TCR Vb (120).
Output chains: P=peptide, A=MHC, D=TCR Va, E=TCR Vb (original numbering restored)."""
import sys
SEGMENTS = [("A", 180, 1), ("B", 100, 0), ("D", 116, 0), ("E", 120, 1)]  # MHC, b2m, Va, Vb

def split(inp, outp):
    # collect residues of merged chain B in file order
    bres = []  # ordered unique resnums in chain B
    seen = set()
    for l in open(inp):
        if l.startswith("ATOM") and l[21] == "B":
            rn = int(l[22:26])
            if rn not in seen:
                seen.add(rn); bres.append(rn)
    # map each B resnum -> (newchain, newresnum)
    bmap = {}; i = 0
    for ch, n, start in SEGMENTS:
        for j in range(n):
            if i < len(bres):
                bmap[bres[i]] = (ch, start + j); i += 1
    out = []
    for l in open(inp):
        if l.startswith(("ATOM", "HETATM")):
            c = l[21]
            if c == "A":                        # peptide -> P
                nl = l[:21] + "P" + l[22:]
            elif c == "B":
                rn = int(l[22:26]); ch, nr = bmap[rn]
                nl = l[:21] + ch + f"{nr:>4d}" + l[26:]
            else:
                nl = l
            out.append(nl)
        elif l.startswith("TER"):
            out.append(l)
    open(outp, "w").writelines(out)

if __name__ == "__main__":
    split(sys.argv[1], sys.argv[2])
    print("wrote", sys.argv[2])
