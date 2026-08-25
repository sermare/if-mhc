#!/usr/bin/env python3
"""Write the SKEMPI/6AM design tables into the panel's on-disk FASTA layout.

The panel notebooks read designs from

    outputs/panel/<PDB>/<arm>/mpnn/seqs/{vanilla,nomhc}_<PDB>.fa
    outputs/panel/<PDB>/<arm>/esmif/seqs/<PDB>.fa
    outputs/panel/<PDB>/<arm>/ligandmpnn/seqs/<PDB>.fa

with arm in {full, mhconly}. The new campaign ships the same designs as gzipped tidy CSVs
under designs/<dataset>/t01/ and calls the TCR-deleted arm `notcr`. Rewriting them into the
layout above lets every notebook and script pick the new complexes up by extending its
structure list -- no loader changes, so the analysis code stays exactly as published.

Two conventions are matched deliberately:

- LigandMPNN records are colon-joined chains and the panel loader takes field 2, so the
  epitope is written as the third field.
- ProteinMPNN and LigandMPNN echo the *input* sequence back as their first record, and the
  panel notebooks are not consistent about it: the design notebook starts reading at record 2
  (skipping the echo) while the recovery notebook starts at record 0 (keeping it). The echo is
  therefore reproduced here exactly as the real tools emit it, so both loaders treat the new
  structures the way they already treat the panel. ESM-IF writes no echo and gets none.

  /home/ubuntu/miniforge3/bin/python3 py/materialize_extended_designs.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path("/home/ubuntu/if-mhc")
PANEL = ROOT / "outputs/panel"
DATASETS = ["skempi", "focus6am"]
ARM_DIR = {"full": "full", "notcr": "mhconly"}

# model key in the CSV -> (subdirectory, filename template)
LAYOUT = {
    "proteinmpnn":       ("mpnn", "vanilla_{pdb}.fa"),
    "proteinmpnn_nomhc": ("mpnn", "nomhc_{pdb}.fa"),
    "esmif":             ("esmif", "{pdb}.fa"),
    "ligandmpnn":        ("ligandmpnn", "{pdb}.fa"),
}


def main() -> None:
    written, missing = 0, []
    for dataset in DATASETS:
        d = ROOT / "designs" / dataset / "t01"
        man = pd.read_csv(d / "manifest.csv")[["complex", "pdb"]]
        pdb_of = dict(zip(man["complex"], man["pdb"]))

        for csv in sorted(d.glob("*.csv.gz")):
            model = csv.name.replace(".csv.gz", "")
            subdir, template = LAYOUT[model]
            df = pd.read_csv(csv, usecols=["complex", "arm", "seq", "recovery", "score", "native"]
                             if model != "esmif"
                             else ["complex", "arm", "seq", "recovery", "native"])
            if "score" not in df.columns:
                df["score"] = float("nan")

            for (cx, arm), g in df.groupby(["complex", "arm"], sort=False):
                pdb = pdb_of[cx]
                out = PANEL / pdb / ARM_DIR[arm] / subdir / "seqs"
                out.mkdir(parents=True, exist_ok=True)
                path = out / template.format(pdb=pdb)

                lines = []
                native = g["native"].iloc[0]
                if model == "ligandmpnn":
                    lines.append(f">{pdb}, T=0.1, seed=41, num_res={len(native)}, "
                                 f"batch_size=32, model_path=./model_params/"
                                 f"ligandmpnn_v_32_010_25.pt")
                    lines.append(f"NA:NA:{native}")
                elif model != "esmif":
                    lines.append(f">{pdb}, score=nan, global_score=nan, "
                                 f"designed_chains=['C'], model_name=v_48_020, seed=37")
                    lines.append(native)

                for i, (seq, rec, sc) in enumerate(
                        zip(g["seq"], g["recovery"], g["score"]), start=1):
                    if model == "ligandmpnn":
                        lines.append(f">{pdb}, id={i}, T=0.1, overall_confidence={sc:.4f}, "
                                     f"seq_rec={rec:.4f}")
                        lines.append(f"NA:NA:{seq}")
                    elif model == "esmif":
                        lines.append(f">esmif_{i - 1}, score=nan, recovery={rec:.4f}, T=0.1")
                        lines.append(seq)
                    else:
                        lines.append(f">T=0.1, sample={i}, score={sc:.4f}, "
                                     f"seq_recovery={rec:.4f}")
                        lines.append(seq)
                path.write_text("\n".join(lines) + "\n")
                written += 1

    # any (structure, arm, model) cell with no designs at all
    for dataset in DATASETS:
        man = pd.read_csv(ROOT / "designs" / dataset / "t01/manifest.csv")
        for pdb in man["pdb"]:
            for arm in ARM_DIR.values():
                for subdir, template in LAYOUT.values():
                    p = PANEL / pdb / arm / subdir / "seqs" / template.format(pdb=pdb)
                    if not p.exists():
                        missing.append(f"{pdb}/{arm}/{p.name}")

    print(f"wrote {written} FASTA files under {PANEL}")
    if missing:
        print(f"\n{len(missing)} (structure, arm, model) cells have no designs:")
        for m in sorted(set(missing)):
            print(f"  {m}")


if __name__ == "__main__":
    main()
