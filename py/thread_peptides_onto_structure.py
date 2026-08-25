#!/usr/bin/env python3
"""Thread arbitrary equal-length peptides onto a structure's peptide chain.

Generalizes py/thread_kd_peptides_pdb.py to any PDB / chain. Needed because LigandMPNN's score.py
scores the sequence present in the PDB (no --path_to_fasta), so each peptide needs its own file.

Chain C keeps only backbone N/CA/C/O with residue names rewritten: the crystal side chains belong to
the native peptide and would be wrong for any mutant. The first altloc encountered is kept. All other
chains pass through untouched as structural context.
"""
import argparse
from pathlib import Path

BACKBONE = {"N", "CA", "C", "O"}
ONE2THREE = {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN",
             "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
             "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP",
             "Y": "TYR", "V": "VAL"}


def peptide_resids(pdb_text, chain):
    """Ordered unique residue ids of the peptide chain (handles altlocs and insertion codes)."""
    out = []
    for L in pdb_text.splitlines():
        if L.startswith("ATOM") and L[21] == chain and L[12:16].strip() == "CA":
            rid = L[22:27]
            if rid not in out:
                out.append(rid)
    return out


def thread(pdb_text, chain, peptide, resids):
    keep_alt = {}
    lines = []
    for L in pdb_text.splitlines():
        if not L.startswith(("ATOM", "HETATM")):
            if L.startswith(("TER", "END")):
                lines.append(L)
            continue
        if L.startswith("HETATM") or L[21] != chain:
            lines.append(L)
            continue
        if L[12:16].strip() not in BACKBONE:
            continue
        rid, alt, atom = L[22:27], L[16], L[12:16].strip()
        # keep one altloc per (residue, atom): whichever appears first
        k = (rid, atom)
        if alt != " ":
            if k in keep_alt and keep_alt[k] != alt:
                continue
            keep_alt[k] = alt
        aa = peptide[resids.index(rid)]
        lines.append(L[:16] + " " + ONE2THREE[aa] + L[20:])
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--chain", default="C")
    ap.add_argument("--peptides", required=True, help="comma-separated, or path to a one-per-line file")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()

    text = Path(a.pdb).read_text()
    resids = peptide_resids(text, a.chain)
    peps = (Path(a.peptides).read_text().split() if Path(a.peptides).exists()
            else a.peptides.split(","))
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    print(f"{Path(a.pdb).name} chain {a.chain}: {len(resids)} residues")
    for p in peps:
        if len(p) != len(resids):
            print(f"  SKIP {p}: length {len(p)} != chain length {len(resids)}")
            continue
        (out / f"{p}.pdb").write_text(thread(text, a.chain, p, resids))
    print(f"  wrote {len(list(out.glob('*.pdb')))} threaded PDBs to {out}")


if __name__ == "__main__":
    main()
