#!/usr/bin/env python3
"""Thread each KD-tested peptide onto the 2P5E chain-C backbone.

LigandMPNN's score.py scores the sequence present in the PDB (no --path_to_fasta, unlike
ProteinMPNN's score_only), so each peptide needs its own PDB. Chain C keeps only backbone
atoms (N/CA/C/O) with residue names rewritten, since the crystal side chains belong to the
native peptide and would be wrong for any mutant. Altloc A is kept at the two disordered
positions (3, 6). All other chains pass through untouched as structural context.
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path("/home/ubuntu/if-mhc")
PDB_IN = ROOT / "inputs/pmhc_tcr_dataset/2P5E.pdb"
OUT_DIR = ROOT / "outputs/kd_scoring/ligandmpnn/pdbs"
BACKBONE = {"N", "CA", "C", "O"}
ONE2THREE = {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN",
             "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
             "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP",
             "Y": "TYR", "V": "VAL"}


def thread(peptide, out_path):
    lines = []
    for L in PDB_IN.read_text().splitlines():
        if not L.startswith(("ATOM", "HETATM")):
            if L.startswith("TER") or L.startswith("END"):
                lines.append(L)
            continue
        if L[21] != "C" or L.startswith("HETATM"):
            lines.append(L)
            continue
        # chain C: backbone only, altloc A or blank, rename to the target residue
        if L[12:16].strip() not in BACKBONE:
            continue
        if L[16] not in (" ", "A"):
            continue
        resi = int(L[22:26])
        aa = peptide[resi - 1]
        lines.append(L[:16] + " " + ONE2THREE[aa] + L[20:])
    out_path.write_text("\n".join(lines) + "\n")


def main():
    df = pd.read_csv(ROOT / "outputs/analysis/kd_score_correlation.csv")
    peps = sorted(set(df["Peptide"]))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for p in peps:
        assert len(p) == 9, f"{p} is not a 9-mer"
        thread(p, OUT_DIR / f"{p}.pdb")
    print(f"wrote {len(peps)} threaded PDBs to {OUT_DIR}")


if __name__ == "__main__":
    main()
