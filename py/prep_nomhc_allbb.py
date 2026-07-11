#!/usr/bin/env python3
"""Stage every core_load backbone for a noMHC ProteinMPNN sweep.

For each backbone load_all() accepts (valid 10-mer peptide + MHC that fits the
6AMU reference), detect the peptide chain id, symlink the PDB into a staging dir
under a unique short name (bbNNNN.pdb), and write a manifest carrying
conditioning / pid / group / native-basin geometry / peptide-chain. The manifest
is what lets the logo notebook facet designs by conditioning and score recovery
against the right native epitope (GIG for 6AM5, DRG for 6AMU)."""
import sys, os, csv
sys.path.insert(0, "/home/ubuntu/if-mhc/py")
from core_load import load_all, _pp, _rl, _sq

ROOT = "/home/ubuntu/if-mhc"
OUT  = f"{ROOT}/outputs/mpnn_nomhc_allbb"
STAGE = f"{OUT}/pdb_in"
os.makedirs(STAGE, exist_ok=True)
# clear any prior staging
for f in os.listdir(STAGE):
    p = os.path.join(STAGE, f)
    if os.path.islink(p) or os.path.isfile(p): os.remove(p)

def pep_chain(path):
    """chain id of the 10-mer peptide (mirrors core_load._detect)."""
    m = _pp.get_structure("x", path)[0]
    chs = {c.id: _rl(c) for c in m}
    pep = [c for c, rs in chs.items() if len(rs) == 10]
    mhc = [c for c, rs in chs.items() if len(rs) > 150 and "SHSMRYFF" in _sq(rs)[:14]]
    if not pep or not mhc: return None
    return pep[0]

DF, _, _ = load_all()          # cached; built fresh moments ago
rows = []
miss = 0
for i, r in DF.reset_index(drop=True).iterrows():
    f = r["file"]
    pc = pep_chain(f)
    if pc is None:
        miss += 1; continue
    name = f"bb{len(rows):04d}"
    os.symlink(os.path.realpath(f), f"{STAGE}/{name}.pdb")
    rows.append({"target": name, "pep_chain": pc, "pid": r["pid"], "group": r["group"],
                 "cond": r["cond"], "nice": r["nice"], "toGIG": r["toGIG"],
                 "toDRG": r["toDRG"], "min_native": min(r["toGIG"], r["toDRG"]),
                 "seated": int(r["seated"]), "src": f})

with open(f"{OUT}/manifest.csv", "w", newline="") as o:
    w = csv.DictWriter(o, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

print(f"staged {len(rows)} backbones  (peptide-chain undetected on {miss})")
from collections import Counter
print("pid:", dict(Counter(x["pid"] for x in rows)))
print("pep chains used:", dict(Counter(x["pep_chain"] for x in rows)))
