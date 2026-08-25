#!/usr/bin/env python3
"""Score the NY-ESO-1 KD peptide panel on every NY-ESO-1 crystal, not just one.

notebooks/27_kd_peptide_inverse_folding_scores.ipynb scored the 51 measured peptides against a single
structure (2P5E). That conflates two things: how a peptide scores, and which crystal it was scored
against. Seven independent NY-ESO-1 crystals exist locally, six of them the same SLLMWITQC epitope,
so the same 51 peptides can be scored on all of them. That gives, per peptide, a mean and a spread
across structures, and lets the question be asked directly: is the between-structure spread related
to crystallographic resolution?

Every structure is prepped to the same canonical A/B/C/D/E layout with a 9-mer epitope in chain C,
and every peptide in the panel is a 9-mer, so no re-threading of lengths is needed.

ProteinMPNN and ProteinMPNN (no MHC) use the batched scorer: one featurisation per structure, then
the designed-chain sequence is varied. LigandMPNN is optional (--with-ligandmpnn) because it needs a
separate threaded PDB per peptide and is ~14x more work at this panel size.

  /home/ubuntu/miniforge3/bin/python3 py/score_nyeso_kd_multistructure.py [--with-ligandmpnn]
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path("/home/ubuntu/if-mhc")
sys.path.insert(0, str(ROOT / "py"))
from batch_score_peptides_mpnn import score_peptides  # noqa: E402

OUT = ROOT / "outputs/analysis/nyeso_kd_multistructure_scores.csv"
# every locally available NY-ESO-1 crystal. 2BNQ carries the 9V variant rather than the
# wild-type cysteine, kept and flagged so the effect of the index residue can be separated.
STRUCTS = ["2P5E", "2P5W", "2BNR", "2F53", "2F54", "2PYE", "2BNQ"]
# 10, not the 50 the 13-peptide DMF5 script used. n_orders re-runs the entire network per
# batch -- encoder included, over all ~820 residues, even though only the 9 peptide residues
# ever change -- so at 51 peptides x 7 structures x 2 models, 50 orders means 2,800 full
# forward passes. Validation of this scorer put n_orders=1 at r=0.989 and n_orders=10 at
# r=0.991 against exact scoring, so 10 is already past diminishing returns.
N_ORDERS = 10
LMPNN = ROOT / "LigandMPNN"
PY_ENV = "/home/ubuntu/miniforge3/envs/esmcba/bin/python"


def resolution_of(pdb: str) -> float:
    for line in (ROOT / f"inputs/pmhc_tcr_dataset/{pdb}.pdb").read_text().splitlines():
        if line.startswith("REMARK   2 RESOLUTION"):
            try:
                return float(line.split("RESOLUTION.")[1].split("ANGSTROMS")[0].strip())
            except Exception:
                return float("nan")
    return float("nan")


def native_of(pdb: str) -> str:
    d = pd.read_csv(ROOT / "inputs/pmhc_tcr_dataset/dataset.csv").set_index("pdb")
    return d.loc[pdb, "peptide"]


def ligandmpnn_scores(struct: str, peps: list[str]) -> dict[str, float]:
    """One threaded PDB per peptide, because LigandMPNN's score.py reads sequence from the file."""
    outdir = ROOT / f"outputs/nyeso_scoring/{struct}"
    outdir.mkdir(parents=True, exist_ok=True)
    thread = ROOT / "py/thread_peptides_onto_structure.py"
    pepfile = outdir / "peptides.txt"
    pepfile.write_text("\n".join(peps) + "\n")
    subprocess.run(["/home/ubuntu/miniforge3/bin/python3", str(thread),
                    "--pdb", str(ROOT / f"inputs/pmhc_tcr_dataset/{struct}.pdb"),
                    "--chain", "C", "--peptides", str(pepfile),
                    "--outdir", str(outdir / "threaded")], check=True,
                   stdout=subprocess.DEVNULL)
    sc_dir = outdir / "score_only"
    sc_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([PY_ENV, "score.py", "--model_type", "ligand_mpnn",
                    "--autoregressive_score", "1", "--use_sequence", "1",
                    "--pdb_path_multi", str(outdir / "threaded" / "pdbs.json"),
                    "--out_folder", str(sc_dir), "--chains_to_design", "C",
                    "--number_of_batches", "1"], cwd=str(LMPNN), check=True,
                   stdout=subprocess.DEVNULL)
    out = {}
    for pt in sorted((sc_dir / "-").glob("*.pt")) or sorted(sc_dir.rglob("*.pt")):
        d = torch.load(pt, map_location="cpu")
        seq = "".join(d["native_sequence"]) if isinstance(d.get("native_sequence"), list) else None
        lp, cm = d["log_probs"], d["chain_mask"].bool()
        out[seq or pt.stem] = float(-(lp[cm].max(dim=-1).values.mean()))
    return out


def main() -> None:
    kd = pd.read_csv(ROOT / "outputs/analysis/kd_score_correlation_3model.csv")
    peps = kd["Peptide"].tolist()
    print(f"{len(peps)} KD peptides x {len(STRUCTS)} NY-ESO-1 structures")

    rows = []
    for struct in STRUCTS:
        pdb = ROOT / f"inputs/pmhc_tcr_dataset/{struct}.pdb"
        res, nat = resolution_of(struct), native_of(struct)
        sc = {}
        for weights in ["vanilla", "nomhc"]:
            # batch 16, not the full panel: these are ~820-residue complexes and the decoder
            # activations OOM a 22GB card at 51
            sc[weights] = score_peptides(peps, weights, pdb, design_chain="C",
                                         batch=16, n_orders=N_ORDERS)
        lig = ligandmpnn_scores(struct, peps) if "--with-ligandmpnn" in sys.argv else {}
        for i, pep in enumerate(peps):
            rows.append(dict(structure=struct, resolution_A=res, native=nat,
                             is_native_epitope=pep == nat, peptide=pep,
                             score_vanilla=sc["vanilla"][i], score_nomhc=sc["nomhc"][i],
                             score_ligandmpnn=lig.get(pep, np.nan)))
        print(f"  {struct} ({res:.2f} A, native {nat}): "
              f"vanilla mean {np.nanmean(sc['vanilla']):.3f}, "
              f"nomhc mean {np.nanmean(sc['nomhc']):.3f}")

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}  ({len(df)} rows)")




def noise_floor(struct: str = "2P5W", seeds: tuple[int, ...] = (41, 7, 123)) -> None:
    """Re-score one structure under different decoding-order seeds.

    The quantity of interest is the spread of a peptide's score ACROSS structures. That spread is
    only meaningful relative to how much the score moves when nothing changes but the random
    decoding order, so this measures that floor directly.

      /home/ubuntu/miniforge3/envs/esmcba/bin/python py/score_nyeso_kd_multistructure.py --noise-floor
    """
    kd = pd.read_csv(ROOT / "outputs/analysis/kd_score_correlation_3model.csv")
    peps = kd["Peptide"].tolist()
    pdb = ROOT / f"inputs/pmhc_tcr_dataset/{struct}.pdb"
    cols = {}
    for s in seeds:
        cols[s] = score_peptides(peps, "vanilla", pdb, design_chain="C", batch=16,
                                 n_orders=N_ORDERS, seed=s)
        print(f"  seed {s}: mean {np.nanmean(cols[s]):.4f}")
    m = np.vstack([cols[s] for s in seeds])
    per_pep_sd = m.std(axis=0, ddof=1)
    print(f"\nseed-to-seed SD per peptide on {struct}: mean {per_pep_sd.mean():.4f}, "
          f"max {per_pep_sd.max():.4f}  (n={len(peps)} peptides, {len(seeds)} seeds, "
          f"n_orders={N_ORDERS})")
    pd.DataFrame({"peptide": peps, **{f"seed_{s}": cols[s] for s in seeds},
                  "seed_sd": per_pep_sd}).to_csv(
        ROOT / "outputs/analysis/nyeso_kd_seed_noise_floor.csv", index=False)
    print(f"wrote {ROOT / 'outputs/analysis/nyeso_kd_seed_noise_floor.csv'}")


if __name__ == "__main__":
    if "--noise-floor" in sys.argv:
        noise_floor()
    else:
        main()
