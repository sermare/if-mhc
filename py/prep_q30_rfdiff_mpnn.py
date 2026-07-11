#!/usr/bin/env python3
"""Stage in-basin diffused backbones for noMHC ProteinMPNN design (Q30 campaign, Stage 3).

Reads outputs/rfdiff_q30/basins.csv, keeps rows with seated==1 (a diffused backbone that landed
in a defined basin), and for each:
  - copies the diffused PDB into outputs/mpnn_q30/pdb_in/bbNNNN.pdb with the **peptide chain C
    relabeled to the native epitope sequence** (GIG=SMLGIGIVPV / DRG=MMWDRGLGMM by landed basin) so
    the FIXED anchor positions carry native identity (RFdiffusion leaves the diffused peptide seq
    arbitrary; MPNN only reads backbone N/CA/C/O + the residue name at fixed positions).
  - records the fixed-position scheme for that basin.

Writes:
  outputs/mpnn_q30/manifest.csv           target,pep_chain,pid,seed,basin,toGIG,toDRG,src
  outputs/mpnn_q30/fixed_positions.jsonl  {target: {"C": [fixed positions]}}  (native held; rest free)

Fixed/free (chain C, 1-indexed), per LANDED basin:
  GIG: fix {1,2,6,10}          free 3,4,5,7,8,9   (p3 = Q30 register contact, designable)
  DRG: fix {1,2,5,6,7,9,10}    free 4,8           (p4 = register contact, designable)
"""
import os, sys, json, csv
import pandas as pd
sys.path.insert(0, "/home/ubuntu/if-mhc/py")
from Bio.PDB import PDBParser, PDBIO, Select

ROOT = "/home/ubuntu/if-mhc"
OUT  = f"{ROOT}/outputs/mpnn_q30"
STAGE = f"{OUT}/pdb_in"
os.makedirs(STAGE, exist_ok=True)

NATIVE = {"GIG": "SMLGIGIVPV", "DRG": "MMWDRGLGMM"}
FIXED  = {"GIG": [1, 2, 6, 10], "DRG": [1, 2, 5, 6, 7, 9, 10]}
AA3 = {'A':'ALA','R':'ARG','N':'ASN','D':'ASP','C':'CYS','Q':'GLN','E':'GLU','G':'GLY','H':'HIS',
       'I':'ILE','L':'LEU','K':'LYS','M':'MET','F':'PHE','P':'PRO','S':'SER','T':'THR','W':'TRP',
       'Y':'TYR','V':'VAL'}
PEP = "C"
_pp = PDBParser(QUIET=True)


class _Prot(Select):
    def accept_residue(self, r):
        return 1 if r.id[0] == " " else 0


def stage_one(src, dst, basin):
    """copy src -> dst, relabeling chain C residues 1..10 to the native epitope for `basin`."""
    m = _pp.get_structure("x", src)[0]
    seq = NATIVE[basin]
    pep = [r for r in m[PEP] if r.id[0] == " "]
    if len(pep) != 10:
        return False
    for r, aa in zip(pep, seq):          # relabel resname to native (backbone atoms untouched)
        r.resname = AA3[aa]
    io = PDBIO(); io.set_structure(m); io.save(dst, _Prot())
    return True


def main():
    bdf = pd.read_csv(f"{OUT.replace('mpnn_q30','rfdiff_q30')}/basins.csv")
    seated = bdf[bdf.seated == 1].reset_index(drop=True)
    # clear prior staging
    for f in os.listdir(STAGE):
        p = os.path.join(STAGE, f)
        if os.path.isfile(p) or os.path.islink(p):
            os.remove(p)

    rows, fixed = [], {}
    for _, r in seated.iterrows():
        basin = r["basin"]
        target = f"bb{len(rows):04d}"
        dst = f"{STAGE}/{target}.pdb"
        if not stage_one(r["file"], dst, basin):
            continue
        rows.append({"target": target, "pep_chain": PEP, "pid": r["pid"], "seed": r["seed"],
                     "basin": basin, "toGIG": r["toGIG"], "toDRG": r["toDRG"], "src": r["file"]})
        fixed[target] = {PEP: FIXED[basin]}

    if not rows:
        print("no seated backbones to stage (basins.csv has no seated==1). Nothing written.")
        return
    with open(f"{OUT}/manifest.csv", "w", newline="") as o:
        w = csv.DictWriter(o, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    json.dump(fixed, open(f"{OUT}/fixed_positions.jsonl", "w"))

    from collections import Counter
    print(f"staged {len(rows)} in-basin backbones -> {STAGE}")
    print("by landed basin:", dict(Counter(x["basin"] for x in rows)))
    print("fixed-position scheme:", {b: FIXED[b] for b in set(x["basin"] for x in rows)})


if __name__ == "__main__":
    main()
