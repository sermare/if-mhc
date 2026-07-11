#!/usr/bin/env python3
"""Stage seeds for the Q30 + F-pocket dual-hotspot partial-diffusion campaign.

For each seed (GIG = 6AM5_trim + 6AM5_snap*, DRG = 6AMU_trim + 6AMU_snap*; 6AMT EXCLUDED):
  - trim to the pd_sweep footprint (A<=180, B all, C1-10 peptide, D<=115, E<=120) so the
    partial-diffusion assembly matches the working template and stays GPU-tractable. All
    hotspots (D30 + floor F-pocket A77/A84/A116/A123) live inside these ranges. Crystal
    *_trim.pdb inputs are already trimmed and used as-is.
  - build contigmap.contigs (chain order A..E preserved, no reindex) and contigmap.provide_seq
    (0-indexed positions of every NON-peptide residue -> receptor sequence fixed, peptide C free).

Writes trimmed relaxed seeds to outputs/rfdiff_q30/seeds/ and a manifest TSV
(name, pid, input_pdb, contig, provide_seq, hotspots) that jobs/run_q30_rfdiff.sh iterates.

Sanity: the derived (contig, provide_seq) for the crystal trims reproduces jobs/run_pd_sweep.sh
exactly (6AM5: provide=0-279,290-524 ; 6AMU: provide=0-278,289-519).
"""
import os, sys, glob, csv
sys.path.insert(0, "/home/ubuntu/if-mhc/py")
from Bio.PDB import PDBParser, PDBIO, Select

ROOT  = "/home/ubuntu/if-mhc"
OUT   = f"{ROOT}/outputs/rfdiff_q30"
SEEDS = f"{OUT}/seeds"
os.makedirs(SEEDS, exist_ok=True)

# dual hotspot (verified against 6AM5/6AMU coords): Q30 N-term + arm-free floor F-pocket
HOTSPOTS = "D30,A77,A84,A116,A123"
# author-numbering upper bounds per chain (None = keep all protein residues of that chain)
KEEP = {"A": 180, "B": None, "C": None, "D": 115, "E": 120}
CHAIN_ORDER = ["A", "B", "C", "D", "E"]
PEP = "C"
_pp = PDBParser(QUIET=True)


class _Keep(Select):
    def accept_residue(self, r):
        c = r.get_parent().id
        if r.id[0] != " " or c not in KEEP:            # protein residues in kept chains only
            return 0
        hi = KEEP[c]
        return 1 if (hi is None or r.id[1] <= hi) else 0


def kept_residues(model):
    """ordered list of (chain, resnum) kept, following CHAIN_ORDER."""
    out = []
    for c in CHAIN_ORDER:
        if c not in model:
            continue
        for r in model[c]:
            if r.id[0] != " ":
                continue
            hi = KEEP[c]
            if hi is None or r.id[1] <= hi:
                out.append((c, r.id[1]))
    return out


def contig_and_provide(model):
    """(contig, provide_seq) from kept residues; assumes each kept chain range is gap-free."""
    per = {}
    for c, n in kept_residues(model):
        per.setdefault(c, []).append(n)
    contig_parts, ordered = [], []
    for c in CHAIN_ORDER:
        if c not in per:
            continue
        nums = per[c]
        contig_parts.append(f"{c}{nums[0]}-{nums[-1]}/0")
        ordered += [(c, n) for n in nums]
    # provide_seq = 0-indexed positions of every non-peptide residue, collapsed to ranges
    prov = [i for i, (c, _) in enumerate(ordered) if c != PEP]
    ranges, s = [], prov[0]
    for a, b in zip(prov, prov[1:] + [None]):
        if b != (a + 1):
            ranges.append(f"{s}-{a}")
            s = b
    return " ".join(contig_parts), ",".join(ranges)


def enumerate_seeds():
    seeds = [("6AM5_trim", "6AM5", f"{ROOT}/inputs/focus_6am/6AM5_trim.pdb", False),
             ("6AMU_trim", "6AMU", f"{ROOT}/inputs/focus_6am/6AMU_trim.pdb", False)]
    for pid in ("6AM5", "6AMU"):
        for f in sorted(glob.glob(f"{ROOT}/outputs/focus_relax/snapshots/{pid}_snap*.pdb")):
            seeds.append((os.path.basename(f)[:-4], pid, f, True))    # needs trimming
    return seeds


def main():
    rows = []
    for name, pid, src, needs_trim in enumerate_seeds():
        model = _pp.get_structure(name, src)[0]
        if needs_trim:
            inp = f"{SEEDS}/{name}_trim.pdb"
            io = PDBIO(); io.set_structure(model); io.save(inp, _Keep())
            model = _pp.get_structure(name, inp)[0]                   # reparse the trimmed copy
        else:
            inp = src
        # verify peptide chain C == 10 residues
        pep = [r for r in model[PEP] if r.id[0] == " "] if PEP in model else []
        if len(pep) != 10:
            print(f"  SKIP {name}: peptide chain {PEP} has {len(pep)} residues (need 10)")
            continue
        contig, provide = contig_and_provide(model)
        rows.append(dict(name=name, pid=pid, input_pdb=os.path.abspath(inp),
                         contig=contig, provide_seq=provide, hotspots=HOTSPOTS))

    man = f"{OUT}/seeds_manifest.tsv"
    with open(man, "w", newline="") as o:
        w = csv.DictWriter(o, fieldnames=["name", "pid", "input_pdb", "contig", "provide_seq", "hotspots"],
                           delimiter="\t", lineterminator="\n")
        w.writeheader(); w.writerows(rows)

    from collections import Counter
    print(f"staged {len(rows)} seeds -> {man}")
    print("by pid:", dict(Counter(r["pid"] for r in rows)))
    for r in rows[:2] + [x for x in rows if x["name"].endswith("_trim")]:
        print(f"\n  {r['name']} ({r['pid']})")
        print(f"    contig     = [{r['contig']}]")
        print(f"    provide_seq= [{r['provide_seq']}]")
        print(f"    hotspots   = [{r['hotspots']}]")


if __name__ == "__main__":
    main()
