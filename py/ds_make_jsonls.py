#!/usr/bin/env python
"""Build per-structure chain_id + omit jsonls for the dataset MPNN run."""
import json, csv, sys
parsed, dscsv, chain_out, omit_out = sys.argv[1:5]
pep = {r["pdb"]: r["pep_chain"] for r in csv.DictReader(open(dscsv)) if r["valid"]=="True"}
chain_id, omit = {}, {}
for line in open(parsed):
    r = json.loads(line); name = r["name"]
    if name not in pep: continue
    chains = [k[-1] for k in r if k.startswith("seq_chain_")]
    pc = pep[name]
    if pc not in chains: continue
    chain_id[name] = [[pc], [c for c in chains if c != pc]]
    omit[name] = {c: ([[[1], "M"]] if c == pc else []) for c in chains}  # no Met at peptide P1
json.dump(chain_id, open(chain_out, "w"))
json.dump(omit, open(omit_out, "w"))
print(f"built jsonls for {len(chain_id)} structures")
