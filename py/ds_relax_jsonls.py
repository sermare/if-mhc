#!/usr/bin/env python
"""Per-snapshot chain_id + omit jsonls for the dataset-relax MPNN (peptide chain per structure)."""
import json, csv, sys
parsed, dscsv, chain_out, omit_out = sys.argv[1:5]
pep = {r["pdb"]: r["pep_chain"] for r in csv.DictReader(open(dscsv)) if r["valid"] == "True"}
chain_id, omit = {}, {}
for line in open(parsed):
    r = json.loads(line); name = r["name"]
    pdb = name.split("_snap")[0]
    if pdb not in pep: continue
    chains = [k[-1] for k in r if k.startswith("seq_chain_")]
    pc = pep[pdb]
    if pc not in chains: continue
    chain_id[name] = [[pc], [c for c in chains if c != pc]]
    omit[name] = {c: ([[[1], "M"]] if c == pc else []) for c in chains}
json.dump(chain_id, open(chain_out, "w")); json.dump(omit, open(omit_out, "w"))
print(f"built jsonls for {len(chain_id)} snapshots")
