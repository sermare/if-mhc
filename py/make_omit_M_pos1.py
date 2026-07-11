#!/usr/bin/env python
"""Build a ProteinMPNN --omit_AA_jsonl that forbids given AAs at given positions of a chain,
for EVERY structure in a parsed_chains jsonl (the dict needs an entry per name+chain).
Usage: make_omit_M_pos1.py <parsed.jsonl> <out.jsonl> [chain=C] [positions=1] [aas=M]"""
import json, sys
parsed, outp = sys.argv[1], sys.argv[2]
chain  = sys.argv[3] if len(sys.argv) > 3 else "C"
poss   = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "1").split(",")]
aas    = sys.argv[5] if len(sys.argv) > 5 else "M"
d = {}
for line in open(parsed):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    chains = [k[-1] for k in r if k.startswith("seq_chain_")]
    d[r["name"]] = {c: ([[poss, aas]] if c == chain else []) for c in chains}
json.dump(d, open(outp, "w"))
print(f"omit_AA_jsonl: {len(d)} structures | forbid '{aas}' at pos {poss} of chain {chain}")
