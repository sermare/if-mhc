# Inverse-folding epitope designs -- skempi, T=0.3

2,137,472 designed epitope sequences over 28 pMHC-TCR complexes.

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
| `esmif.csv.gz` | esmif | both | 560,000 | 2.6 MB |
| `proteinmpnn.csv.gz` | proteinmpnn | both | 559,104 | 5.7 MB |
| `proteinmpnn_nomhc.csv.gz` | proteinmpnn_nomhc | both | 559,104 | 5.5 MB |
| `ligandmpnn.csv.gz` | ligandmpnn | both | 459,264 | 3.6 MB |

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

Regenerate with `py/skempi_package.py --dataset skempi --temp 0.3`.
