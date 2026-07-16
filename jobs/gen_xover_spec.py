#!/usr/bin/env python3
"""Generate jobs/xover_spec.tsv — the N-terminal-templated CROSSOVER experiment.

Rationale: GIG and DRG are near-identical over P1-P4 (per-residue Ca separation 0.30/0.61/1.14/0.53 A
-- within native thermal noise) and diverge sharply from P5 on (P6=5.62 A, P10=4.70 A; C-term half mean
3.69 A vs N-term half 0.91 A). So templating the N-TERMINUS carries essentially NO register information,
while leaving the divergent C-terminus fully de novo forces the model to *choose* a register.

This is the inverse of everything run so far:
  - the `fix` ladder templates BOTH ends from the design's OWN crystal (fix8 = C1-4/2-2/C7-10), pinning
    the register-defining C-terminus to its own native -> crossing was structurally impossible (0/773).
  - `xreg_spec.tsv` templates the TARGET C-terminus from a chimera (xg_cterm5 = 5-5/C6-10) -> hands the
    model the answer.
Here the C-terminus is free. Two hotspot arms:
  own   : the scaffold crystal's own max-contact hotspot list (does the free C-term return to its own register?)
  cross : the OTHER crystal's max-contact hotspot list (can contact conditioning actively redirect it?)
"""
TARGET = 100

BASE = {"6AM5": "A1-180/0 B1-100/0 D1-115/0 E1-120/0",
        "6AMU": "A2-180/0 B0-99/0 D2-115/0 E4-120/0"}
# max-contact hotspot lists (as used by the `max` / `fix` cells)
MAX = {
 "6AM5": "A159,A7,A99,A63,A147,A66,A77,A70,A155,D30,E97,A143,A146,A167,A73,A156,A67,A80,A97,A84,E30,E96,A116,A152,A171,D50,E98,A5,A9,A59,A81,A45,A76,A114,A123,D31,E95",
 "6AMU": "E100,A159,A7,D30,A66,A77,A147,A167,A156,A63,A146,A155,E99,A99,A116,D93,A152,A97,A70,A9,A59,A171,E102,A73,A80,A81,A123,A143,D28,D91,D94,A5,A67,A114,E101,A150,E98,A45,A84,A163,D92,D95",
}
OTHER = {"6AM5": "6AMU", "6AMU": "6AM5"}

# peptide contig: template C1-N (shared N-term), leave the rest de novo
PEP = {"nt5": "C1-5/5-5", "nt4": "C1-4/6-6", "nt3": "C1-3/7-7", "nt2": "C1-2/8-8"}

rows = []
for x in ("6AM5", "6AMU"):
    for lvl in ("nt5", "nt4", "nt3", "nt2"):                     # own-hotspot arm: all 4 levels
        rows.append((x, f"xo_{lvl}_own", "xover", TARGET, f"{BASE[x]} {PEP[lvl]}", MAX[x]))
    for lvl in ("nt4", "nt2"):                                   # cross-hotspot arm: the redirect test
        rows.append((x, f"xo_{lvl}_cross", "xover", TARGET, f"{BASE[x]} {PEP[lvl]}", MAX[OTHER[x]]))

out = "/global/scratch/users/sergiomar10/if-mhc/jobs/xover_spec.tsv"
with open(out, "w") as f:
    for r in rows:
        f.write("\t".join(str(v) for v in r) + "\n")
print(f"wrote {out}: {len(rows)} cells x {TARGET} = {len(rows)*TARGET} designs")
for r in rows:
    print(f"  {r[0]}\t{r[1]}\t{r[3]}\t{r[4]}\thotspots={'own' if r[5]==MAX[r[0]] else 'CROSS'}")
