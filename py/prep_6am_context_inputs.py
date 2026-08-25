#!/usr/bin/env python3
"""Prepare ProteinMPNN inputs for 6AM5/6AMU under full and MHC-removed context.

Conditions follow jobs/run_context_ablation_mpnn.sh:
  full     A+B+C+D+E  (MHC + b2m + peptide + TCRa + TCRb)
  mhconly  A+B+C      (MHC + b2m + peptide, TCR removed) -- the paper's ablation

Chain C (peptide) is designed in both; everything else is fixed context. Writes one
<out>/<STRUCT>_<cond>/{pdb_in/, parsed.jsonl, chain_id.jsonl} per cell.
"""
import json
import subprocess
from pathlib import Path

ROOT = Path("/home/ubuntu/if-mhc")
MPNN = ROOT / "ProteinMPNN/helper_scripts"
PY = "/home/ubuntu/miniforge3/envs/esmcba/bin/python"
CONDITIONS = {"full": list("ABCDE"), "mhconly": list("ABC")}
STRUCTS = ["6AM5", "6AMU"]
OUT = ROOT / "outputs/context_6am"


def subset_pdb(src, keep, dst):
    lines = [L for L in src.read_text().splitlines()
             if not L.startswith(("ATOM", "HETATM", "TER")) or
             (len(L) > 21 and L[21] in keep)]
    dst.write_text("\n".join(L for L in lines if not L.startswith("HETATM")) + "\n")


def main():
    for s in STRUCTS:
        for cond, keep in CONDITIONS.items():
            cell = OUT / f"{s}_{cond}"
            pdb_in = cell / "pdb_in"
            pdb_in.mkdir(parents=True, exist_ok=True)
            name = f"{s}_{cond}"
            subset_pdb(ROOT / f"inputs/focus_6am/{s}.pdb", set(keep), pdb_in / f"{name}.pdb")
            subprocess.run([PY, str(MPNN / "parse_multiple_chains.py"),
                            "--input_path", str(pdb_in),
                            "--output_path", str(cell / "parsed.jsonl")], check=True)
            designed = ["C"]
            fixed = [c for c in keep if c != "C"]
            (cell / "chain_id.jsonl").write_text(json.dumps({name: [designed, fixed]}) + "\n")
            d = json.loads((cell / "parsed.jsonl").readline() if hasattr(cell / "parsed.jsonl", "readline")
                           else (cell / "parsed.jsonl").read_text().splitlines()[0])
            print(f"{name}: chains={d['num_of_chains']} peptide={d.get('seq_chain_C')} "
                  f"designed={designed} fixed={fixed}")


if __name__ == "__main__":
    main()
