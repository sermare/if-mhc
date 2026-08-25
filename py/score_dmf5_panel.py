#!/usr/bin/env python3
"""Score the DMF5 KD panel (13 10-mers) on both 6AM5 and 6AMU with all three inverse-folding models.

6AM5 carries SMLGIGIVPV ("GIG") and 6AMU carries MMWDRGLGMM ("DRG") -- the same DMF5 TCR engaging
two chemically distinct peptides through different bound conformations. Both index peptides are in
the KD panel, so each structure can be asked whether it prefers its own.

ProteinMPNN/noMHC use the batched scorer (featurize once, vary the designed-chain sequence).
LigandMPNN needs one threaded PDB per peptide because score.py reads sequence from the file.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path("/home/ubuntu/if-mhc")
sys.path.insert(0, str(ROOT / "py"))
from batch_score_peptides_mpnn import score_peptides  # noqa: E402

STRUCTS = {"6AM5": "SMLGIGIVPV", "6AMU": "MMWDRGLGMM"}
LMPNN = ROOT / "LigandMPNN"
PY_ENV = "/home/ubuntu/miniforge3/envs/esmcba/bin/python"
N_ORDERS = 50   # small panel, so buy precision


def ligandmpnn_scores(struct, peps):
    outdir = ROOT / f"outputs/dmf5_scoring/{struct}/score_only"
    outdir.mkdir(parents=True, exist_ok=True)
    chains = "A,B,C,D,E"
    for p in peps:
        if (outdir / f"{p}.pt").exists():
            continue
        subprocess.run(
            [PY_ENV, "score.py", "--model_type", "ligand_mpnn",
             "--checkpoint_ligand_mpnn", "./model_params/ligandmpnn_v_32_010_25.pt",
             "--pdb_path", str(ROOT / f"outputs/dmf5_scoring/{struct}/pdbs/{p}.pdb"),
             "--parse_these_chains_only", chains, "--chains_to_design", "C",
             "--out_folder", str(outdir), "--autoregressive_score", "1", "--use_sequence", "1",
             "--batch_size", "1", "--number_of_batches", str(N_ORDERS), "--seed", "41"],
            cwd=LMPNN, capture_output=True)
    out = {}
    for p in peps:
        f = outdir / f"{p}.pt"
        if not f.exists():
            out[p] = np.nan
            continue
        d = torch.load(f, map_location="cpu", weights_only=False)
        lp = np.asarray(d["log_probs"]); nat = np.asarray(d["native_sequence"])
        sel = np.asarray(d["chain_mask"]) == 1
        out[p] = float((-lp[:, sel, :][:, np.arange(sel.sum()), nat[sel]]).mean())
    return out


def main():
    kd = pd.read_csv("/home/ubuntu/pmhc/modeling/ONG229/comparison/FINAL_dmf5_kd_panel.csv")
    kd = kd[kd.length == 10].reset_index(drop=True)
    peps = kd.peptide.tolist()
    rows = []
    for struct, native in STRUCTS.items():
        pdb = ROOT / f"inputs/focus_6am/{struct}.pdb"
        sc = {}
        for w in ["vanilla", "nomhc"]:
            sc[w] = score_peptides(peps, w, pdb, design_chain="C", batch=13, n_orders=N_ORDERS)
            print(f"{struct} {w}: done", flush=True)
        lm = ligandmpnn_scores(struct, peps)
        print(f"{struct} ligandmpnn: done", flush=True)
        for i, p in enumerate(peps):
            rows.append({"structure": struct, "native_peptide": native, "is_native": p == native,
                         "peptide": p, "score_vanilla": sc["vanilla"][i],
                         "score_nomhc": sc["nomhc"][i], "score_ligandmpnn": lm[p]})
    df = pd.DataFrame(rows).merge(kd[["peptide", "KD_uM", "dG_kcal_mol", "Tm_C", "note"]],
                                  on="peptide", how="left")
    out = ROOT / "outputs/analysis/dmf5_kd_panel_3model_2structure_scores.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}: {len(df)} rows ({df.structure.nunique()} structures x {len(peps)} peptides)")


if __name__ == "__main__":
    main()
