#!/usr/bin/env python3
"""Fold the SKEMPI design scores into the panel's scoring tables.

The SKEMPI campaign shipped its own MHCflurry and ESMCBA scores
(designs/skempi/t01/scores.zip). The panel notebooks read three tables keyed on the
peptide string, so the two score sets are concatenated into that schema and the UMAP is
refit jointly over the combined embeddings -- a UMAP fit on the panel alone cannot place
the new peptides, and projecting them into an old fit would put them on coordinates the
panel's neighbourhood graph never saw.

Only HLA-A*02:01 is merged. The SKEMPI upload also scores 1MI5 (B*08:01) and 2AK4
(B*35:01), but the panel tables carry no allele column and every panel peptide is A2, so
mixing alleles into one peptide-keyed table would silently compare across grooves.

Inputs
  outputs/analysis/panel_unique_peptides_{esmcba,mhcflurry}.csv  + the embeddings .npy
  designs/skempi/t01/scores/{esmcba_scores,mhcflurry_scores}.csv + esmcba_emb_A0201.npy

Outputs (same paths, backed up once as *.panel20.csv)
  panel_unique_peptides_mhcflurry.csv   peptide, mhcflurry_ic50_nM, mhcflurry_percentile
  panel_unique_peptides_esmcba.csv      peptide, esmcba_prediction, UMAP_1, UMAP_2

  /home/ubuntu/miniforge3/envs/esmcba/bin/python py/merge_extended_scores.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from umap import UMAP

ROOT = Path("/home/ubuntu/if-mhc")
ANA = ROOT / "outputs/analysis"
SC = ROOT / "designs/skempi/t01/scores"
ALLELE_MF = "HLA-A*02:01"
ALLELE_EC = "A0201"


def backup(p: Path) -> None:
    b = p.with_suffix(".panel20.csv")
    if not b.exists() and p.exists():
        shutil.copy(p, b)
        print(f"  backed up {p.name} -> {b.name}")


def merge_arm(suffix: str) -> None:
    """Merge one arm's tables. suffix is '' (pMHC+TCR) or '_mhconly' (TCR removed)."""
    print(f"\n=== arm: {'pMHC only' if suffix else 'pMHC+TCR'} ===")

    # ---------------------------------------------------------------- ESMCBA + embeddings
    panel_ec = pd.read_csv(ANA / f"panel_unique_peptides_esmcba{suffix}.csv")
    panel_emb = np.load(ANA / f"panel_unique_peptides_esmcba_embeddings{suffix}.npy")
    assert len(panel_ec) == len(panel_emb), "panel esmcba table and embeddings disagree"

    sk_ec = pd.read_csv(SC / "esmcba_scores.csv")
    sk_ec = sk_ec[sk_ec.esmcba_allele == ALLELE_EC].reset_index(drop=True)
    sk_emb = np.load(SC / f"esmcba_emb_{ALLELE_EC}.npy")
    assert len(sk_ec) == len(sk_emb), "skempi esmcba table and embeddings disagree"

    # the shipped embedding row order is the sorted unique peptides for that allele
    order = np.argsort(sk_ec["seq"].values, kind="stable")
    if not (sk_ec["seq"].values[order] == np.sort(sk_ec["seq"].values)).all():
        raise ValueError("cannot establish skempi embedding row order")
    sk_ec = sk_ec.iloc[order].reset_index(drop=True)

    new = ~sk_ec["seq"].isin(set(panel_ec["peptide"]))
    print(f"esmcba: panel {len(panel_ec):,} + skempi {int(new.sum()):,} new "
          f"({int((~new).sum()):,} already in the panel) = {len(panel_ec) + int(new.sum()):,}")

    peptides = np.concatenate([panel_ec["peptide"].values, sk_ec.loc[new, "seq"].values])
    preds = np.concatenate([panel_ec["esmcba_prediction"].values,
                            sk_ec.loc[new, "esmcba_pred"].values])
    embeds = np.vstack([panel_emb, sk_emb[new.values]])
    assert len(peptides) == len(embeds)

    print(f"refitting UMAP jointly over {embeds.shape[0]:,} x {embeds.shape[1]} embeddings ...")
    coords = UMAP(n_components=2, n_neighbors=15, random_state=42).fit_transform(embeds)

    backup(ANA / f"panel_unique_peptides_esmcba{suffix}.csv")
    out_ec = pd.DataFrame({"peptide": peptides, "esmcba_prediction": preds,
                           "UMAP_1": coords[:, 0], "UMAP_2": coords[:, 1]})
    out_ec.to_csv(ANA / f"panel_unique_peptides_esmcba{suffix}.csv", index=False)
    np.save(ANA / f"panel_unique_peptides_esmcba_embeddings{suffix}.npy", embeds)
    print(f"wrote panel_unique_peptides_esmcba{suffix}.csv: {len(out_ec):,} rows")

    # ---------------------------------------------------------------- MHCflurry
    panel_mf = pd.read_csv(ANA / f"panel_unique_peptides_mhcflurry{suffix}.csv")
    sk_mf = pd.read_csv(SC / "mhcflurry_scores.csv")
    sk_mf = sk_mf[sk_mf.mhcflurry_allele == ALLELE_MF]
    sk_mf = (sk_mf.rename(columns={"seq": "peptide", "affinity": "mhcflurry_ic50_nM"})
                  [["peptide", "mhcflurry_ic50_nM"]])
    sk_mf["mhcflurry_percentile"] = np.nan          # not reported by the SKEMPI run
    add = sk_mf[~sk_mf["peptide"].isin(set(panel_mf["peptide"]))]

    backup(ANA / f"panel_unique_peptides_mhcflurry{suffix}.csv")
    out_mf = pd.concat([panel_mf, add], ignore_index=True).drop_duplicates("peptide")
    out_mf.to_csv(ANA / f"panel_unique_peptides_mhcflurry{suffix}.csv", index=False)
    print(f"mhcflurry: panel {len(panel_mf):,} + skempi {len(add):,} new = {len(out_mf):,}")

    # ---------------------------------------------------------------- coverage report
    meta = pd.read_csv(ANA / f"panel_unique_peptides_metadata{suffix}.csv")
    cov = (meta.assign(scored=meta["peptide"].isin(set(out_ec["peptide"])) &
                                meta["peptide"].isin(set(out_mf["peptide"])))
               .groupby("pdb")["scored"].mean())
    unscored = cov[cov < 0.99]
    print(f"\nstructures fully scored: {int((cov >= 0.99).sum())}/{len(cov)}")
    if len(unscored):
        print("not fully covered (excluded from notebook 06):")
        for pdb, f in unscored.items():
            print(f"  {pdb}: {f:.1%} of its unique designs have both scores")


def main() -> None:
    for suffix in ["", "_mhconly"]:
        merge_arm(suffix)


if __name__ == "__main__":
    main()
