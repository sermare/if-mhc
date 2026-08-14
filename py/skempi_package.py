#!/usr/bin/env python3
"""Package one sweep phase into per-model gzipped CSVs for the repo.

The working directory under outputs/ is ~50 GB and untracked. This writes a
curated, self-describing copy under designs/<dataset>/<temp>/ with one gzipped
CSV per model, small enough to live in git.

GitHub rejects any single file over 100 MB, so a model whose gzipped table
exceeds --max-mb is split by arm, and if still too large, into numbered shards.
Nothing is silently truncated: the row counts are reported and written into the
README next to the data.

  skempi_package.py --dataset skempi --temp 0.1
"""
import argparse, glob, gzip, os, shutil
import pandas as pd

ROOT = "/global/scratch/users/sergiomar10/if-mhc"
OUT = f"{ROOT}/outputs/skempi_if"
MODELS = ["esmif", "proteinmpnn", "proteinmpnn_nomhc", "ligandmpnn"]
MANIFEST = {"skempi": "inputs/skempi/manifest.csv",
            "pmhc25": "inputs/pmhc25/manifest.csv",
            "focus6am": "inputs/focus6am/manifest.csv"}


def write_gz(df, path):
    df.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
    return os.path.getsize(path) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="skempi")
    ap.add_argument("--temp", default="0.1")
    ap.add_argument("--max-mb", type=float, default=90.0)
    a = ap.parse_args()

    phase = f"{a.dataset}/T{a.temp}"
    tag = "t" + a.temp.replace(".", "")
    dst = f"{ROOT}/designs/{a.dataset}/{tag}"
    if os.path.isdir(dst):
        shutil.rmtree(dst)                      # rebuild cleanly, no stale shards
    os.makedirs(dst, exist_ok=True)

    lines, grand = [], 0
    for m in MODELS:
        parts = sorted(glob.glob(f"{OUT}/{phase}/{m}/parts/*.csv"))
        if not parts:
            print(f"{m}: no parts, skipped")
            continue
        df = pd.concat((pd.read_csv(p) for p in parts), ignore_index=True)
        df = df.sort_values(["complex", "arm", "chunk", "sample"]).reset_index(drop=True)
        grand += len(df)

        mb = write_gz(df, f"{dst}/{m}.csv.gz")
        if mb <= a.max_mb:
            print(f"{m:20s} {len(df):9,} rows  {mb:6.1f} MB  -> {m}.csv.gz")
            lines.append(f"| `{m}.csv.gz` | {m} | both | {len(df):,} | {mb:.1f} MB |")
            continue

        # too big for one file: split by arm
        os.remove(f"{dst}/{m}.csv.gz")
        for arm, g in df.groupby("arm"):
            mb = write_gz(g, f"{dst}/{m}__{arm}.csv.gz")
            print(f"{m:20s} {len(g):9,} rows  {mb:6.1f} MB  -> {m}__{arm}.csv.gz (split by arm)")
            lines.append(f"| `{m}__{arm}.csv.gz` | {m} | {arm} | {len(g):,} | {mb:.1f} MB |")

    # ship the epitope definitions and per-cell summary alongside the sequences
    shutil.copy(f"{ROOT}/{MANIFEST[a.dataset]}", f"{dst}/manifest.csv")
    summ = f"{OUT}/{phase}/summary.csv"
    if os.path.exists(summ):
        shutil.copy(summ, f"{dst}/summary.csv")

    n_cplx = len(pd.read_csv(f"{dst}/manifest.csv"))
    readme = f"""# Inverse-folding epitope designs -- {a.dataset}, T={a.temp}

{grand:,} designed epitope sequences over {n_cplx} pMHC-TCR complexes.

Each complex is sampled 10,000 times per model in two arms:

| arm | input structure |
|---|---|
| `full` | epitope + MHC (heavy + B2M) + TCR alpha/beta |
| `notcr` | epitope + MHC only -- TCR chains deleted |

Only the epitope is designed; every other residue is fixed context. The two arms
differ *only* by deletion of the TCR chains, so a difference between them is
attributable to TCR context and nothing else.

## Files

| file | model | arms | rows | size |
|---|---|---|---|---|
""" + "\n".join(lines) + f"""

`manifest.csv` -- per complex: epitope chain, sequence, residue ids, MHC/TCR
chain assignment, chain lengths per arm, and any modified residues rewritten.
`summary.csv` -- per model/arm/complex: recovery, unique count, entropy, top sequence.

## Columns

| column | meaning |
|---|---|
| `complex` | SKEMPI complex id (`PDB_partner1_partner2`) |
| `arm` | `full` (TCR present) or `notcr` (TCR removed) |
| `model` | `esmif`, `proteinmpnn`, `proteinmpnn_nomhc`, `ligandmpnn` |
| `temp` | sampling temperature |
| `seq` | designed epitope |
| `native` | crystallographic epitope |
| `recovery` | per-position identity to `native` |
| `score`, `global_score` | model-reported scores (ESM-IF leaves these blank) |
| `chunk`, `seed`, `sample` | provenance of the draw |

## Models

- **ESM-IF1** (`esm_if1_gvp4_t16_142M_UR50`), batched epitope sampler validated
  against the stock `sample_sequence_in_complex` (identical greedy output, mean
  per-position TVD 0.032 vs a 0.20 sampling-noise floor).
- **ProteinMPNN** `v_48_020` (vanilla weights).
- **ProteinMPNN no-MHC** -- retrained with MHC-containing PDBs excluded.
- **LigandMPNN** `ligandmpnn_v_32_010_25`.

## Known artifact

Position 1 is not usable. All four models place an initiator methionine at the
epitope N-terminus (~92% Met, against natives of L/A/S/Q), giving ~0.06 identity
there regardless of model or arm. This is a free-chain-start prior, not biology.
Analyse P2 onward, or drop P1.

Regenerate with `py/skempi_package.py --dataset {a.dataset} --temp {a.temp}`.
"""
    open(f"{dst}/README.md", "w").write(readme)
    print(f"\n{grand:,} rows packaged into {dst}")
    print("total size:", round(sum(os.path.getsize(f"{dst}/{f}") for f in os.listdir(dst)) / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
