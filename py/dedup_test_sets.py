#!/usr/bin/env python3
"""Non-redundant test sets for the panel, and the key statistics re-run on each.

The 50-structure panel is not 50 independent observations. It contains 33 unique peptides and 26
unique CDR3-beta sequences: SLLMWITQC appears in 5 crystals, ELAGIGILTV and ALWGFFPVL in 4 each, and
several TCRs (A6, 1G4, DMF5) recur across entries. Pooled statistics therefore weight a peptide or a
receptor by how many times it happens to have been crystallised, which double-counts whatever makes
that particular index peptide easy or hard to recover.

This builds progressively stricter sets and re-runs the anchor result on each, so the reader can see
which conclusions depend on the redundancy and which do not.

  all            every structure
  humanI         human MHC class I only (drops mouse H-2 and MHC class II)
  +uniqueTCR     one structure per TCR family, CDR3-beta clustered so near-identical
                 clonotypes (1-2 substitutions) collapse together
  +uniquePeptide one structure per peptide as well
  A2only         HLA-A*02:01 with unique peptide and unique TCR family

Representative for a collapsed group is the best-resolution member, which is deterministic and
independent of the scores being tested.

  /home/ubuntu/miniforge3/bin/python3 py/dedup_test_sets.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

ROOT = Path("/home/ubuntu/if-mhc")
sys.path.insert(0, str(ROOT / "py"))
from design_corpus import MODELS, MODEL_LABEL  # noqa: E402

A2 = set("""2P5W 1QSF 1QRN 2BNR 2GJ6 2F53 2F54 3QDG 3QEQ 3QFJ 3GSN 1OGA 3UTS 5C0A 5C0B 5HHO 5EU6
2VLR 5NME 1BD2 1LP9 1QSE 2BNQ 2J8U 2JCC 2PYE 2UWE 3D3V 3H9S 3PWP 3QDJ 4FTV 4JFD 4JFE 4JFF 4L3E 4MNQ
5E9D 6AM5 6AMU""".split())
HUMAN_CLASS_I = A2 | {"4MJI", "1MI5", "2AK4"}          # + HLA-B*51:01, B*08:01, B*35


def cdr3_families(m: pd.DataFrame, max_subs: int = 2) -> dict[str, int]:
    """Group structures whose CDR3-beta differ by at most `max_subs` substitutions.

    Exact-match grouping would treat 2P5W (ASSYLGNTGELF) and 2BNR (ASSYVGNTGELF) as independent
    receptors when they differ at one position and are the same 1G4-derived clonotype.
    """
    seqs = m.set_index("pdb")["CDR3b"].fillna("").to_dict()
    fam, nxt = {}, 0
    for pdb, s in seqs.items():
        hit = None
        for other, f in fam.items():
            t = seqs[other]
            if len(t) == len(s) and sum(a != b for a, b in zip(s, t)) <= max_subs:
                hit = f; break
        fam[pdb] = hit if hit is not None else nxt
        nxt += hit is None
    return fam


def build_sets(m: pd.DataFrame) -> dict[str, list[str]]:
    m = m.copy()
    m["fam"] = m["pdb"].map(cdr3_families(m))

    def pick(df, keys):
        """Best-resolution representative per group; deterministic and score-independent."""
        return df.sort_values("resolution_A").groupby(keys, as_index=False).first()["pdb"].tolist()

    hi = m[m.pdb.isin(HUMAN_CLASS_I)]
    a2 = m[m.pdb.isin(A2)]
    return {
        "all": m.pdb.tolist(),
        "humanI": hi.pdb.tolist(),
        "humanI+uniqueTCR": pick(hi, ["fam"]),
        "humanI+uniqueTCR+uniquePeptide": pick(hi, ["fam", "peptide"]),
        "A2+uniqueTCR+uniquePeptide": pick(a2, ["fam", "peptide"]),
    }


def main() -> None:
    m = pd.read_csv(ROOT / "outputs/analysis/panel_dataset_master_table.csv")
    pp = pd.read_parquet(ROOT / "outputs/design_corpus_perpos.parquet")
    sets = build_sets(m)

    print("=== anchor result on progressively non-redundant sets (pMHC+TCR) ===")
    rows = []
    for name, pdbs in sets.items():
        s = pp[(pp.arm == "full") & (pp.pdb.isin(pdbs))]
        interior = s.loc[s.region == "interior", "recovery"]
        rec = {"set": name, "n_struct": len(pdbs),
               "n_peptide": m[m.pdb.isin(pdbs)].peptide.nunique(),
               "interior": interior.mean()}
        for reg in ["P2", "POmega"]:
            v = s.loc[s.region == reg, "recovery"]
            rec[reg] = v.mean()
            rec[f"p_{reg}"] = mannwhitneyu(v, interior)[1]
        rows.append(rec)
    out = pd.DataFrame(rows)
    print(out.round(4).to_string(index=False))

    print("\n=== TCR removal: recovery drop, per set (paired by structure, all models) ===")
    for name, pdbs in sets.items():
        s = pp[pp.pdb.isin(pdbs)]
        w = s.groupby(["complex", "model", "arm"], observed=True)["recovery"].mean().unstack("arm").dropna()
        st, p = wilcoxon(w["mhconly"], w["full"])
        print(f"  {name:32s} n={len(w):3d}  {w['full'].mean():.3f} -> {w['mhconly'].mean():.3f}  p={p:.2e}")

    out.to_csv(ROOT / "outputs/analysis/dedup_test_sets_anchor.csv", index=False)
    pd.DataFrame([{"set": k, "pdbs": " ".join(v)} for k, v in sets.items()]).to_csv(
        ROOT / "outputs/analysis/dedup_test_sets_membership.csv", index=False)
    print(f"\nwrote {ROOT / 'outputs/analysis/dedup_test_sets_anchor.csv'}")
    for k, v in sets.items():
        print(f"  {k:32s} {len(v):2d}: {' '.join(sorted(v))}")


if __name__ == "__main__":
    main()
