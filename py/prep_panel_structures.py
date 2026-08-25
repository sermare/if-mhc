#!/usr/bin/env python3
"""Prep filtered/renamed PDBs + ProteinMPNN parsed.jsonl/chain_id.jsonl for the wider pMHC-TCR panel.

Roles are identified by residue-count RANGE, not by original chain letter or file position --
letter conventions are NOT consistent across the panel (e.g. 3GSN uses H/P/L/A/B instead of
A/B/C/D/E; 5HHO's file order is A,B,D,E,C with the peptide chain last, not third). For n_chains=10
rows (two copies of the assembly in the asymmetric unit) we keep only the FIRST occurrence of each
role by file order, which naturally selects the first copy since the second copy's chains appear
later in the file.

Role ranges (residue count, unique resSeq per chain), validated against inputs/pmhc_tcr_dataset/
dataset.csv across all 25 rows:
    MHC heavy : 270-280
    b2m       : 95-105
    TCRa      : 185-215
    TCRb      : 235-250
    peptide   : given explicitly by dataset.csv's pep_chain column (8-10 residues, too short/
                variable a range to classify by length alone)

Output per structure: pdbs/{PDB}_full.pdb (chains renamed to canonical A=MHC,B=b2m,C=peptide,
D=TCRa,E=TCRb) and pdbs/{PDB}_mhconly.pdb (A=MHC,B=b2m,C=peptide only), plus parsed_full.jsonl/
chain_id_full.jsonl and parsed_mhconly.jsonl/chain_id_mhconly.jsonl for ProteinMPNN.
"""
import csv
import json
import os
import subprocess
import sys

ROOT = "/home/ubuntu/if-mhc"
DATASET_CSV = f"{ROOT}/inputs/pmhc_tcr_dataset/dataset.csv"
PDB_DIR = f"{ROOT}/inputs/pmhc_tcr_dataset"
OUT_DIR = f"{ROOT}/outputs/panel_prep"
MPNN_HELPERS = f"{ROOT}/ProteinMPNN/helper_scripts"
PY = "/home/ubuntu/miniforge3/envs/esmcba/bin/python"

AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "MSE": "M",
}

ROLE_RANGES = {"mhc": (270, 280), "b2m": (95, 105), "tcra": (185, 215), "tcrb": (235, 250)}


def load_targets():
    rows = []
    with open(DATASET_CSV) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def chain_residue_order(pdb_path):
    """Return [(chain_id, [(resseq, icode), ...ordered unique]), ...] in first-appearance order."""
    order = []
    seen_chains = {}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            chain = line[21]
            resseq = line[22:26]
            icode = line[26]
            key = (resseq, icode)
            if chain not in seen_chains:
                seen_chains[chain] = []
                order.append(chain)
            if not seen_chains[chain] or seen_chains[chain][-1] != key:
                seen_chains[chain].append(key)
    return order, seen_chains


def classify_roles(pdb_path, pep_chain, pep_len, peptide_seq):
    order, seen_chains = chain_residue_order(pdb_path)
    counts = {c: len(seen_chains[c]) for c in order}

    if pep_chain not in counts:
        raise ValueError(f"pep_chain {pep_chain} not found in {pdb_path} (chains: {order})")
    if abs(counts[pep_chain] - pep_len) > 1:
        raise ValueError(
            f"{pdb_path}: pep_chain {pep_chain} has {counts[pep_chain]} residues, expected {pep_len}"
        )

    roles = {"peptide": pep_chain}
    used = {pep_chain}
    for role, (lo, hi) in ROLE_RANGES.items():
        found = None
        for c in order:
            if c in used:
                continue
            if lo <= counts[c] <= hi:
                found = c
                break
        if found is None:
            raise ValueError(f"{pdb_path}: no chain found for role {role} (counts={counts})")
        roles[role] = found
        used.add(found)

    return roles, counts


def extract_peptide_seq(pdb_path, chain):
    seq = []
    seen = set()
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[21] != chain:
                continue
            if line[12:16].strip() != "CA":
                continue
            key = (line[22:26], line[26])
            if key in seen:
                continue
            seen.add(key)
            resn = line[17:20].strip()
            seq.append(AA3TO1.get(resn, "X"))
    return "".join(seq)


def write_filtered_pdb(pdb_path, out_path, chain_map):
    """chain_map: {orig_chain: new_chain}. Only lines for chains in chain_map are kept."""
    with open(pdb_path) as f, open(out_path, "w") as out:
        for line in f:
            if line.startswith(("ATOM", "TER")):
                chain = line[21]
                if chain not in chain_map:
                    continue
                new_chain = chain_map[chain]
                line = line[:21] + new_chain + line[22:]
                out.write(line)
        out.write("END\n")


def build_parsed_and_chain_id(pdb_dir, out_prefix, designed_chain, fixed_chains):
    parsed_path = f"{out_prefix}_parsed.jsonl"
    chain_id_path = f"{out_prefix}_chain_id.jsonl"
    subprocess.run(
        [PY, f"{MPNN_HELPERS}/parse_multiple_chains.py",
         "--input_path", pdb_dir, "--output_path", parsed_path],
        check=True, cwd=ROOT,
    )
    with open(parsed_path) as f:
        lines = [json.loads(l) for l in f if l.strip()]
    if not lines:
        raise RuntimeError(f"parse_multiple_chains produced no entries for {pdb_dir}")
    name = lines[0]["name"]
    with open(chain_id_path, "w") as f:
        f.write(json.dumps({name: [[designed_chain], fixed_chains]}) + "\n")
    return parsed_path, chain_id_path, name


def prep_one(row, verbose=True):
    pdb = row["pdb"]
    pep_chain = row["pep_chain"]
    pep_len = int(row["pep_len"])
    src = f"{PDB_DIR}/{pdb}.pdb"
    roles, counts = classify_roles(src, pep_chain, pep_len, row["peptide"])

    seq_check = extract_peptide_seq(src, pep_chain)
    if seq_check != row["peptide"]:
        raise ValueError(f"{pdb}: extracted peptide seq {seq_check} != dataset {row['peptide']}")

    if verbose:
        print(f"{pdb}: roles={roles} counts={{{', '.join(f'{c}:{counts[c]}' for c in counts)}}} "
              f"peptide_ok={seq_check == row['peptide']}")

    struct_dir = f"{OUT_DIR}/{pdb}"
    os.makedirs(f"{struct_dir}/pdbs/full", exist_ok=True)
    os.makedirs(f"{struct_dir}/pdbs/mhconly", exist_ok=True)

    full_map = {roles["mhc"]: "A", roles["b2m"]: "B", roles["peptide"]: "C",
                roles["tcra"]: "D", roles["tcrb"]: "E"}
    mhconly_map = {roles["mhc"]: "A", roles["b2m"]: "B", roles["peptide"]: "C"}

    full_pdb = f"{struct_dir}/pdbs/full/{pdb}.pdb"
    mhconly_pdb = f"{struct_dir}/pdbs/mhconly/{pdb}.pdb"
    write_filtered_pdb(src, full_pdb, full_map)
    write_filtered_pdb(src, mhconly_pdb, mhconly_map)

    full_parsed, full_chain_id, full_name = build_parsed_and_chain_id(
        f"{struct_dir}/pdbs/full", f"{struct_dir}/full", "C", ["A", "B", "D", "E"])
    mhc_parsed, mhc_chain_id, mhc_name = build_parsed_and_chain_id(
        f"{struct_dir}/pdbs/mhconly", f"{struct_dir}/mhconly", "C", ["A", "B"])

    meta = {
        "pdb": pdb, "roles": roles, "counts": counts,
        "peptide": row["peptide"], "peptide_verified": seq_check == row["peptide"],
        "full_parsed": full_parsed, "full_chain_id": full_chain_id, "full_name": full_name,
        "mhconly_parsed": mhc_parsed, "mhconly_chain_id": mhc_chain_id, "mhconly_name": mhc_name,
        "full_pdb": full_pdb, "mhconly_pdb": mhconly_pdb,
        "allele": row.get("allele", ""),
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{struct_dir}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta


if __name__ == "__main__":
    targets = sys.argv[1:]
    rows = load_targets()
    by_pdb = {r["pdb"]: r for r in rows}
    if not targets:
        print("usage: prep_panel_structures.py PDB1 PDB2 ...", file=sys.stderr)
        sys.exit(1)
    ok, fail = [], []
    for pdb in targets:
        row = by_pdb.get(pdb)
        if row is None:
            print(f"SKIP {pdb}: not in dataset.csv", file=sys.stderr)
            fail.append(pdb)
            continue
        try:
            prep_one(row)
            ok.append(pdb)
        except Exception as e:
            print(f"FAIL {pdb}: {e}", file=sys.stderr)
            fail.append(pdb)
    print(f"\nok={len(ok)} fail={len(fail)}")
    if fail:
        print("failed:", fail)
        sys.exit(1)
