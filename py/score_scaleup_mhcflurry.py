#!/usr/bin/env python3
"""MHCflurry HLA-A*02:01 affinity for every unique design in the allele-matched complexes.

Extends the paper's presentation analysis from its own twenty-structure panel to the
disjoint SKEMPI set. Only A*02:01-restricted complexes are scored, matching the paper's
rule that a design is only scored against the allele its own crystal actually presents.

  panel   19 of 20 (4MJI is HLA-B*51:01 and is excluded)
  skempi  19 of 28 (the 'HLA-A2 plus ...' entries in SKEMPI's Protein 1 field);
          HLA-B8/B35, mouse H2-L and the five class II complexes are excluded

Same predictor and cutoff as jobs/run_mhcflurry.sh: Class1PresentationPredictor,
affinity in nM, 500 nM threshold. CPU only.

Two steps, because the mhcflurry env has no parquet engine and the base env has no
mhcflurry -- rather than mutate either shared env, the unique-peptide list is handed
over as a CSV:

  /home/ubuntu/miniforge3/bin/python3 py/score_scaleup_mhcflurry.py --export
  /home/ubuntu/miniforge3/envs/mhcflurry/bin/python py/score_scaleup_mhcflurry.py --score
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "py"))
from design_corpus import load_corpus  # noqa: E402

OUT = ROOT / "outputs/scaleup_mhcflurry.csv"
CELLS = ROOT / "outputs/scaleup_mhcflurry_cells.csv"
AA = set("ACDEFGHIKLMNPQRSTVWY")
PANEL_NON_A2 = {"4MJI"}          # HLA-B*51:01
MHCFLURRY_LEN = range(8, 16)


def a2_complexes_skempi() -> set[str]:
    d = pd.read_csv(ROOT / "inputs/skempi/skempi_tcr_pmhc.csv")
    d["complex"] = d["#Pdb"]
    p1 = d.groupby("complex")["Protein 1"].agg(lambda s: s.mode().iloc[0])
    return set(p1[p1.str.startswith("HLA-A2")].index)


def export() -> None:
    """Base env: write the allele-matched unique (cell, peptide) table for scoring."""
    df = load_corpus(datasets=["panel", "skempi"])
    keep_skempi = a2_complexes_skempi()
    mask = ((df["dataset"] == "panel") & (~df["complex"].isin(PANEL_NON_A2))) | \
           ((df["dataset"] == "skempi") & (df["complex"].isin(keep_skempi)))
    df = df[mask]
    print(f"{df['complex'].nunique()} A*02:01 complexes, {len(df):,} designs")

    uniq = (df.groupby(["dataset", "complex", "arm", "model"], observed=True)["seq"]
              .unique().explode().reset_index(name="peptide"))
    uniq = uniq[uniq["peptide"].map(lambda s: len(s) in MHCFLURRY_LEN and set(s) <= AA)]
    nat = df[["dataset", "complex", "native"]].drop_duplicates()
    uniq = uniq.merge(nat, on=["dataset", "complex"], how="left")
    CELLS.parent.mkdir(parents=True, exist_ok=True)
    uniq.to_csv(CELLS, index=False)
    print(f"wrote {CELLS}  ({len(uniq):,} (cell, peptide) pairs, "
          f"{uniq['peptide'].nunique():,} distinct peptides)")


def main() -> None:
    uniq = pd.read_csv(CELLS)
    peptides = sorted(uniq["peptide"].unique())
    print(f"{len(uniq):,} (cell, peptide) pairs over {len(peptides):,} distinct peptides")

    from mhcflurry import Class1PresentationPredictor
    pred = Class1PresentationPredictor.load()
    t0 = time.time()
    res = pred.predict(peptides=peptides, alleles=["HLA-A*02:01"], verbose=0)
    print(f"predicted in {time.time() - t0:.1f}s")

    res = res[["peptide", "affinity", "presentation_score"]].drop_duplicates("peptide")
    out = uniq.merge(res, on="peptide", how="left")
    # natives too, so each complex can be compared against the epitope it actually presents
    nat_res = pred.predict(peptides=sorted(uniq["native"].unique()),
                           alleles=["HLA-A*02:01"], verbose=0)
    out = out.merge(nat_res[["peptide", "affinity"]].drop_duplicates("peptide")
                    .rename(columns={"peptide": "native", "affinity": "native_affinity"}),
                    on="native", how="left")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT}  ({len(out):,} rows)")

    out["binder"] = out["affinity"] < 500
    print("\n=== fraction of unique designs under 500 nM ===")
    print(out.groupby(["dataset", "model", "arm"], observed=True)["binder"]
             .agg(["mean", "size"]).round(3).to_string())


if __name__ == "__main__":
    if "--export" in sys.argv:
        export()
    else:
        main()
