#!/usr/bin/env python3
"""Build ProteinMPNN input jsons for one SKEMPI arm, and verify epitope indexing.

Produces, under outputs/skempi_if/mpnn_inputs/<arm>/:
  parsed_chains.jsonl     from ProteinMPNN's own parser (the source of truth)
  assigned_chains.jsonl   {target: [[designed chains], [fixed chains]]}
  fixed_positions.jsonl   {target: {chain: [1-indexed positions to hold fixed]}}

Only the epitope is designed. For the three class-II complexes whose epitope is
fused to the MHC-beta N-terminus, the whole beta chain is nominally "designed"
and every position past the epitope is pinned via fixed_positions, so the
designed window is identical in both cases.

The parser drops residues with incomplete backbones, which would silently shift
those 1-indexed positions, so the epitope window is re-read out of the parsed
sequence and checked against the manifest before anything is written.
"""
import argparse, json, os, subprocess, sys
import pandas as pd

ROOT = "/global/scratch/users/sergiomar10/if-mhc"
SKDIR = {"skempi": f"{ROOT}/inputs/skempi", "pmhc25": f"{ROOT}/inputs/pmhc25",
         "focus6am": f"{ROOT}/inputs/focus6am"}
PMPNN = "/global/scratch/users/sergiomar10/TCera/ProteinMPNN"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["full", "notcr"])
    ap.add_argument("--dataset", default="skempi", choices=["skempi", "pmhc25", "focus6am"])
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out", default=f"{ROOT}/outputs/skempi_if/mpnn_inputs")
    a = ap.parse_args()

    SK = SKDIR[a.dataset]
    out = f"{a.out}/{a.dataset}/{a.arm}"
    os.makedirs(out, exist_ok=True)
    parsed = f"{out}/parsed_chains.jsonl"

    print(f"[{a.arm}] parsing structures with ProteinMPNN's parser ...", flush=True)
    subprocess.run([a.python, f"{PMPNN}/helper_scripts/parse_multiple_chains.py",
                    "--input_path", f"{SK}/arm_{a.arm}",
                    "--output_path", parsed], check=True)

    man = pd.read_csv(f"{SK}/manifest.csv").set_index("complex")
    percplx = f"{out}/per_complex"
    os.makedirs(percplx, exist_ok=True)
    assigned, fixed = {}, {}
    bad = []

    for line in open(parsed):
        rec = json.loads(line)
        name = rec["name"]
        if name not in man.index:
            print(f"  !! parsed target {name} not in manifest -- skipped")
            continue
        r = man.loc[name]
        pep_ch, pep_len, pep_seq = r["pep_chain"], int(r["pep_len"]), r["pep_seq"]
        chains = sorted(k.split("_")[-1] for k in rec if k.startswith("seq_chain_"))
        seq = rec[f"seq_chain_{pep_ch}"]

        # epitope always occupies the first pep_len parsed residues of pep_chain
        window = seq[:pep_len]
        if window != pep_seq:
            bad.append((name, pep_ch, pep_seq, window, len(seq)))
            continue

        assigned[name] = [[pep_ch], [c for c in chains if c != pep_ch]]
        # every designed chain needs an entry; empty list == nothing pinned
        fixed[name] = {pep_ch: list(range(pep_len + 1, len(seq) + 1))
                       if len(seq) > pep_len else []}

        # one self-contained input set per complex, so each array task runs alone
        with open(f"{percplx}/{name}.parsed.jsonl", "w") as fh:
            fh.write(json.dumps(rec) + "\n")
        json.dump({name: assigned[name]}, open(f"{percplx}/{name}.assigned.jsonl", "w"))
        json.dump({name: fixed[name]}, open(f"{percplx}/{name}.fixed.jsonl", "w"))

    if bad:
        print("\nEPITOPE INDEX MISMATCH -- refusing to write inputs:")
        for name, ch, want, got, L in bad:
            print(f"  {name} chain {ch} (parsed len {L}): manifest={want} parsed={got}")
        sys.exit(1)

    json.dump(assigned, open(f"{out}/assigned_chains.jsonl", "w"))
    json.dump(fixed, open(f"{out}/fixed_positions.jsonl", "w"))
    fused = {k: v for k, v in fixed.items() if any(v.values())}
    print(f"\n[{a.arm}] {len(assigned)} targets verified (epitope window matches manifest); "
          f"{len(fused)} with fused epitopes needing pinned positions")
    for name in sorted(fused):
        r = man.loc[name]
        ch = r["pep_chain"]
        print(f"    {name}: design {ch}[1-{r['pep_len']}] = {r['pep_seq']}, "
              f"pin {ch}[{r['pep_len']+1}-{max(fused[name][ch])}]")


if __name__ == "__main__":
    main()
