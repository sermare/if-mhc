#!/usr/bin/env python3
"""v4 Arm 2 seed builder: graft the DONOR peptide C-terminus (the differing/register-defining half)
onto the BASE complex's fixed N-terminus, in the MHC floor frame. Produces a chimeric seed that, by
construction, carries the OTHER register's C-terminus on this frame — the 'pull to the other well'
starting point. Backbone atoms (N,CA,C,O) of C-term residues are replaced; base sequence/identity kept
(structure is what RFdiffusion uses; sequence handled downstream).

Writes outputs/rfdiff_q30_v4arm2/seeds/<name>.pdb and prints the register read-out of each chimera.
"""
import sys, os, numpy as np
sys.path.insert(0, "/home/ubuntu/if-mhc/py")
from Bio.PDB import PDBParser, PDBIO
import q30_analysis as Q

ROOT = "/home/ubuntu/if-mhc"
OUT = f"{ROOT}/outputs/rfdiff_q30_v4arm2/seeds"
os.makedirs(OUT, exist_ok=True)
p = PDBParser(QUIET=True)
BB = {"N", "CA", "C", "O"}
CTERM = [6, 7, 8, 9, 10]          # graft the C-terminal half (matches Arm 1 released region)


def floor_R_t(src, dst):
    """R,t mapping src MHC-floor onto dst MHC-floor (Kabsch over shared floor CA)."""
    def fc(m):
        d = {}
        for n in Q.FLOOR:
            r = [x for x in m["A"] if x.id[1] == n and x.id[0] == " "]
            if r and "CA" in r[0]:
                d[n] = r[0]["CA"].coord
        return d
    s, d = fc(src), fc(dst)
    common = [n for n in s if n in d]
    R, t, rms = Q.kabsch(np.array([s[n] for n in common]), np.array([d[n] for n in common]))
    return R, t, rms


def graft(base_pdb, donor_pdb, name):
    base = p.get_structure("b", base_pdb)[0]
    donor = p.get_structure("d", donor_pdb)[0]
    R, t, rms = floor_R_t(donor, base)                       # donor -> base frame
    dpep = {r.id[1]: r for r in donor["C"] if r.id[0] == " "}
    for r in base["C"]:
        if r.id[0] == " " and r.id[1] in CTERM and r.id[1] in dpep:
            dr = dpep[r.id[1]]
            for a in r:
                if a.name in BB and a.name in dr:
                    a.set_coord(dr[a.name].coord @ R + t)
    outp = f"{OUT}/{name}.pdb"
    io = PDBIO(); io.set_structure(base); io.save(outp)
    return outp, rms


def score(pdb):
    m = p.get_structure("x", pdb)[0]
    A = [r for r in m["A"] if r.id[0] == " "]
    pep = np.array([r["CA"].coord for r in m["C"] if r.id[0] == " "])
    floor = {n: [r for r in A if r.id[1] == n][0]["CA"].coord for n in Q.FLOOR
             if any(r.id[1] == n for r in A)}
    common = [n for n in Q.REF_FLOOR if n in floor]
    R, t, _ = Q.kabsch(np.array([floor[n] for n in common]), np.array([Q.REF_FLOOR[n] for n in common]))
    pcf = pep @ R + t
    fpc = np.argmin(np.linalg.norm(pcf - Q.FPOCKET_CENTROID, axis=1)) + 1
    return Q.rmsd(pcf, Q.REFS["GIG"]), Q.rmsd(pcf, Q.REFS["DRG_shift"]), fpc


if __name__ == "__main__":
    jobs = [("GIGbase_DRGcterm", f"{ROOT}/inputs/focus_6am/6AM5_trim.pdb", f"{ROOT}/inputs/focus_6am/6AMU_trim.pdb"),
            ("DRGbase_GIGcterm", f"{ROOT}/inputs/focus_6am/6AMU_trim.pdb", f"{ROOT}/inputs/focus_6am/6AM5_trim.pdb")]
    print(f"grafting C-term residues {CTERM}\n")
    for name, base, donor in jobs:
        outp, rms = graft(base, donor, name)
        g, d, fpc = score(outp)
        print(f"{name}: floor-superpose RMSD={rms:.2f} | chimera toGIG={g:.2f} toDRG={d:.2f} "
              f"| F-pocket pos={fpc} ({'DRG-like P9' if fpc==9 else 'GIG-like P10' if fpc==10 else 'other'})")
