#!/usr/bin/env python3
"""Batched ProteinMPNN scoring of many 9-mer peptides on one fixed backbone.

protein_mpnn_run.py --score_only reloads and re-featurizes per fasta entry and writes one npz
each, which is fine for 51 peptides and hopeless for 100k. This reproduces the identical
quantity -- mean per-residue NLL over the designed chain, teacher-forced, averaged over K random
decoding orders -- but loads the model and featurizes the structure once, then varies only the
designed-chain sequence across the batch dimension.

Equivalence to score_only rests on the same two facts that script relies on: tied_featurize puts
the masked (designed) chains first in S, so S[:, :9] is chain C; and the score is
_scores(S, log_probs, mask*chain_M*chain_M_pos).

Validate with --validate_kd before trusting a large run.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

MPNN = Path("/home/ubuntu/if-mhc/ProteinMPNN")
sys.path.insert(0, str(MPNN))
from protein_mpnn_utils import ProteinMPNN, StructureDatasetPDB, parse_PDB, tied_featurize, _scores  # noqa: E402

ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
CKPT = {"vanilla": MPNN / "vanilla_model_weights/v_48_020.pt",
        "nomhc": MPNN / "nomhc_model_weights/proteinmpnn_nomhc.pt"}


def load_model(weights, device):
    ck = torch.load(CKPT[weights], map_location=device)
    hidden = ck["hidden_dim"] if "hidden_dim" in ck else 128
    layers = ck["num_layers"] if "num_layers" in ck else 3
    m = ProteinMPNN(ca_only=False, num_letters=21, node_features=hidden, edge_features=hidden,
                    hidden_dim=hidden, num_encoder_layers=layers, num_decoder_layers=layers,
                    augment_eps=0.0, k_neighbors=ck["num_edges"])
    m.load_state_dict(ck["model_state_dict"])
    m.to(device).eval()
    return m


def featurize_once(pdb_path, design_chain, batch, device):
    pdb_dict = parse_PDB(str(pdb_path), ca_only=False)
    ds = StructureDatasetPDB(pdb_dict, truncate=None, max_length=20000)
    protein = ds[0]
    all_chains = [k[-1] for k in pdb_dict[0] if k.startswith("seq_chain_")]
    designed = [design_chain]
    fixed = [c for c in all_chains if c != design_chain]
    chain_id_dict = {pdb_dict[0]["name"]: (designed, fixed)}
    batch_clones = [protein for _ in range(batch)]
    out = tied_featurize(batch_clones, device, chain_id_dict, None, None, None, None, None, ca_only=False)
    return out


def score_peptides(peptides, weights, pdb_path, design_chain="C", batch=200, n_orders=10,
                   seed=41, device=None, log_every=0):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(weights, device)
    torch.manual_seed(seed)
    feats = featurize_once(pdb_path, design_chain, batch, device)
    # tied_featurize return order (see protein_mpnn_run.py): X, S, mask, lengths, chain_M,
    # chain_encoding_all, chain_list_list, visible_list_list, masked_list_list,
    # masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, ...
    X, S, mask, chain_M, chain_encoding_all = feats[0], feats[1], feats[2], feats[4], feats[5]
    chain_M_pos, residue_idx = feats[10], feats[12]
    aa2i = {a: i for i, a in enumerate(ALPHABET)}
    L = len(peptides[0])
    scores = np.full(len(peptides), np.nan, dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(peptides), batch):
            chunk = peptides[start:start + batch]
            b = len(chunk)
            Sb = S[:b].clone()
            sub = torch.tensor([[aa2i[a] for a in p] for p in chunk], device=device)
            Sb[:, :L] = sub
            Xb, mb, cMb, cMPb, ceb = X[:b], mask[:b], chain_M[:b], chain_M_pos[:b], chain_encoding_all[:b]
            ridx = residue_idx[:b]
            acc = torch.zeros(b, device=device)
            for _ in range(n_orders):
                randn = torch.randn(cMb.shape, device=device)
                lp = model(Xb, Sb, mb, cMb * cMPb, ridx, ceb, randn)
                acc += _scores(Sb, lp, mb * cMb * cMPb)
            scores[start:start + b] = (acc / n_orders).cpu().numpy()
            if log_every and (start // batch) % log_every == 0:
                print(f"  {start + b}/{len(peptides)}", flush=True)
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="vanilla", choices=list(CKPT))
    ap.add_argument("--pdb", default="/home/ubuntu/if-mhc/inputs/pmhc_tcr_dataset/2P5E.pdb")
    ap.add_argument("--validate_kd", action="store_true")
    ap.add_argument("--peptides_npy", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--n_orders", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0,
                    help="score only the first N peptides; the .npy order is a uniform "
                         "random sample, so a prefix is itself a valid random subsample")
    args = ap.parse_args()

    if args.validate_kd:
        import pandas as pd
        from scipy.stats import pearsonr
        ref = pd.read_csv("/home/ubuntu/if-mhc/outputs/analysis/kd_score_correlation.csv")
        col = {"vanilla": "score_vanilla", "nomhc": "score_nomhc"}[args.weights]
        got = score_peptides(list(ref.Peptide), args.weights, args.pdb,
                             batch=args.batch, n_orders=args.n_orders)
        r, p = pearsonr(got, ref[col])
        print(f"\nvalidate {args.weights}: n={len(got)}  Pearson r={r:.4f}  "
              f"mean|diff|={np.abs(got - ref[col]).mean():.4f}  max|diff|={np.abs(got - ref[col]).max():.4f}")
        for i in range(3):
            print(f"  {ref.Peptide[i]}  mine={got[i]:.4f}  score_only={ref[col][i]:.4f}")
        return

    peps = list(np.load(args.peptides_npy, allow_pickle=True))
    if args.limit:
        peps = peps[:args.limit]
    print(f"scoring {len(peps):,} peptides with {args.weights}", flush=True)
    sc = score_peptides(peps, args.weights, args.pdb, batch=args.batch,
                        n_orders=args.n_orders, log_every=100)
    np.savez_compressed(args.out, peptides=np.array(peps, dtype=object), score=sc)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
