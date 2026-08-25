#!/usr/bin/env python3
"""Turn ProteinMPNN / LigandMPNN fasta output into the shared epitope CSV schema.

Both tools emit whole-chain (ProteinMPNN: designed chains joined by '/') or
whole-complex (LigandMPNN: every chain joined by ':') sequences, so the epitope
has to be sliced back out. For LigandMPNN the chain order is the PDB's parse
order, which is re-derived from the structure and asserted against the native
record before any slicing happens -- a silent misalignment here would quietly
produce epitopes taken from the wrong chain.

Output columns match esmif_sample.py:
  complex, arm, model, temp, sample, seq, native, recovery [, score, global_score]
"""
import argparse, os, re, warnings
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import three_to_index, index_to_one

warnings.filterwarnings("ignore")
ROOT = "/global/scratch/users/sergiomar10/if-mhc"
SKDIR = {"skempi": f"{ROOT}/inputs/skempi", "pmhc25": f"{ROOT}/inputs/pmhc25",
         "focus6am": f"{ROOT}/inputs/focus6am"}
AA3 = {"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU","LYS",
       "MET","PHE","PRO","SER","THR","TRP","TYR","VAL"}


def read_fasta(path):
    recs, hdr, seq = [], None, []
    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith(">"):
            if hdr is not None:
                recs.append((hdr, "".join(seq)))
            hdr, seq = line[1:], []
        else:
            seq.append(line)
    if hdr is not None:
        recs.append((hdr, "".join(seq)))
    return recs


def pdb_chain_seqs(path):
    """Ordered [(chain_id, sequence)] as they appear in the file."""
    m = PDBParser(QUIET=True).get_structure("x", path)[0]
    out = []
    for ch in m:
        s = ""
        for r in ch:
            n = r.get_resname()
            if n == "MSE":
                s += "M"
            elif r.id[0] == " " and n in AA3:
                s += index_to_one(three_to_index(n))
        if s:
            out.append((ch.id, s))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--complex", required=True)
    ap.add_argument("--arm", required=True, choices=["full", "notcr"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--tool", required=True, choices=["proteinmpnn", "ligandmpnn"])
    ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--dataset", default="skempi", choices=["skempi", "pmhc25", "focus6am"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    SK = SKDIR[a.dataset]
    man = pd.read_csv(f"{SK}/manifest.csv").set_index("complex").loc[a.complex]
    pep_ch, pep_len, pep_seq = man["pep_chain"], int(man["pep_len"]), man["pep_seq"]
    recs = read_fasta(a.fasta)

    rows = []
    if a.tool == "proteinmpnn":
        # record 0 is the native; designed records carry only the designed
        # chains, and exactly one chain is designed here
        for hdr, seq in recs[1:]:
            d = dict(re.findall(r"(\w+)=([-\d.eE+]+)", hdr))
            assert "/" not in seq, f"expected a single designed chain, got {seq[:40]}"
            rows.append({"seq": seq[:pep_len], "score": d.get("score"),
                         "global_score": d.get("global_score"),
                         "sample": int(float(d.get("sample", len(rows) + 1)))})
        # native comes from the manifest, which mpnn_prep already checked against
        # the parsed structure -- ProteinMPNN's native line reorders chains, so
        # slicing it here would be a second, fragile place to get that wrong
        nat = pep_seq
    else:
        chains = pdb_chain_seqs(f"{SK}/arm_{a.arm}/{a.complex}.pdb")
        nat_parts = recs[0][1].split(":")
        # LigandMPNN emits chains sorted by chain id, which is NOT the order they
        # appear in the file (3GSN is H,P,L,A,B on disk but A,B,H,L,P in the
        # fasta). Sorting here matches its convention; the length check below
        # then confirms the mapping rather than assuming it.
        chains = sorted(chains, key=lambda t: t[0])
        if [len(sq) for _, sq in chains] != [len(p) for p in nat_parts]:
            raise SystemExit(
                f"chain-order mismatch for {a.complex}/{a.arm}: "
                f"pdb={[(c, len(sq)) for c, sq in chains]} fasta={[len(p) for p in nat_parts]}")
        pep_idx = [c for c, _ in chains].index(pep_ch)
        if nat_parts[pep_idx][:pep_len] != pep_seq:
            raise SystemExit(
                f"epitope mismatch for {a.complex}/{a.arm}: "
                f"manifest={pep_seq} fasta={nat_parts[pep_idx][:pep_len]}")
        for hdr, seq in recs[1:]:
            d = dict(re.findall(r"(\w+)=([-\d.eE+]+)", hdr))
            parts = seq.split(":")
            rows.append({"seq": parts[pep_idx][:pep_len],
                         "score": d.get("overall_confidence"),
                         "global_score": d.get("ligand_confidence"),
                         "sample": int(float(d.get("id", len(rows) + 1)))})
        nat = nat_parts[pep_idx][:pep_len]      # already checked against pep_seq

    if nat != pep_seq:
        raise SystemExit(f"native epitope mismatch: manifest={pep_seq} got={nat}")

    df = pd.DataFrame(rows)
    df["recovery"] = [sum(x == y for x, y in zip(s, nat)) / pep_len for s in df["seq"]]
    df.insert(0, "complex", a.complex)
    df.insert(1, "arm", a.arm)
    df.insert(2, "model", a.model)
    df.insert(3, "temp", a.temp)
    df["native"] = nat
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"  collected {len(df)} seqs -> {a.out} "
          f"(uniq={df.seq.nunique()}, rec={df.recovery.mean():.3f})")


if __name__ == "__main__":
    main()
