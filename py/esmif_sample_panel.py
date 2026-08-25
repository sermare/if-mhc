#!/usr/bin/env python3
"""Sample peptide sequences with ESM-IF1 (esm_if1_gvp4_t16_142M_UR50), conditioned on the rest of a
pMHC-TCR complex, matching this project's ProteinMPNN convention: one chain (default C, the peptide)
is "designed" (sampled), all other chains are fixed context.

Batches B independent decode chains behind ONE encoder pass (the encoder cost -- ~0.3s for an
~866-residue complex like 3HG1 -- otherwise gets paid on every single-sequence call, which is the
bulk of why esm.inverse_folding.multichain_util.sample_sequence_in_complex is slow in a loop). B=8 is
the default: on this machine's 22GB L4, peak memory scales ~1.1GB/item + ~1.3GB overhead (measured:
B=1 1.86GB, B=8 10.89GB), and per-sequence time improvement mostly saturates by B=8 (0.284s/seq vs
0.607s/seq at B=1) -- B=16+ risks OOM on a shared GPU, not worth it for the marginal further gain.

IMPORTANT shape gotcha (undocumented, differs from the docstring in transformer_decoder.py): with
incremental_state supplied, TransformerDecoder.forward returns logits of shape (batch, vocab, 1) --
NOT (batch, tgt_len, vocab) as its own docstring claims. Index with logits[:, :, -1], not
logits[:, -1, :] (the latter silently produces degenerate/collapsed output -- verified empirically,
see conversation this script was built from).

Resumable: counts existing FASTA records and only samples the remainder, so a killed/restarted run
picks up where it left off (same convention as jobs/run_3hg1_100k.sh etc.).

Usage:
  python py/esmif_sample.py --pdb inputs/pmhc_tcr_dataset/3HG1.pdb --chains A B C D E \
      --target-chain C --num-seqs 10000 --batch-size 8 --temperature 0.1 \
      --out outputs/esmif_3hg1_pilot/seqs/3HG1.fa
"""
import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

ESM_REPO = "/home/ubuntu/if-mhc/esm_repo"
sys.path.insert(0, ESM_REPO)
os.environ.setdefault("TORCH_HOME", "/home/ubuntu/if-mhc/models_cache/torch_hub")

import esm  # noqa: E402
from esm.inverse_folding.util import CoordBatchConverter  # noqa: E402
from esm.inverse_folding.multichain_util import _concatenate_coords  # noqa: E402


def sample_batch(model, all_coords, target_chain_len, batch_size, temperature, device):
    L = all_coords.shape[0]
    bc = CoordBatchConverter(model.decoder.dictionary)
    batch_coords, confidence, _, _, padding_mask = bc(
        [(all_coords, None, None)] * batch_size, device=device
    )
    mask_idx = model.decoder.dictionary.get_idx("<mask>")
    pad_idx = model.decoder.dictionary.get_idx("<pad>")
    cath_idx = model.decoder.dictionary.get_idx("<cath>")

    sampled_tokens = torch.full((batch_size, 1 + L), mask_idx, dtype=torch.long, device=device)
    sampled_tokens[:, 0] = cath_idx
    for i in range(target_chain_len, L):
        sampled_tokens[:, i + 1] = pad_idx

    incremental_state = dict()
    with torch.no_grad():
        encoder_out = model.encoder(batch_coords, padding_mask, confidence)
        for i in range(1, target_chain_len + 1):
            logits, _ = model.decoder(
                sampled_tokens[:, :i], encoder_out, incremental_state=incremental_state
            )
            lg = logits[:, :, -1] / temperature  # (B, vocab) -- see module docstring on shape gotcha
            probs = F.softmax(lg, dim=-1)
            sampled_tokens[:, i] = torch.multinomial(probs, 1).squeeze(-1)

    seqs = []
    for b in range(batch_size):
        seq = "".join(
            model.decoder.dictionary.get_tok(int(a))
            for a in sampled_tokens[b, 1 : 1 + target_chain_len]
        )
        seqs.append(seq)
    return seqs


def count_fa(path):
    if not os.path.exists(path):
        return 0
    n = 0
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--chains", nargs="+", required=True, help="all chains to load, e.g. A B C D E")
    ap.add_argument("--target-chain", required=True, help="chain to sample/design, e.g. C")
    ap.add_argument("--num-seqs", type=int, required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--out", required=True, help="output fasta path (appended to, resumable)")
    ap.add_argument("--seed", type=int, default=37)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    have = count_fa(args.out)
    remaining = args.num_seqs - have
    if remaining <= 0:
        print(f"[esmif] {args.out} already has {have}/{args.num_seqs} -- nothing to do")
        return
    print(f"[esmif] {args.out}: have={have}, generating {remaining} more (target {args.num_seqs})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.eval().to(device)

    structure = esm.inverse_folding.util.load_structure(args.pdb, args.chains)
    coords, native_seqs = esm.inverse_folding.multichain_util.extract_coords_from_complex(structure)
    target_chain_len = coords[args.target_chain].shape[0]
    all_coords = _concatenate_coords(coords, args.target_chain)
    native = native_seqs[args.target_chain]
    print(f"[esmif] target chain {args.target_chain}: native={native} (len={target_chain_len})")

    n_written = 0
    t0 = time.time()
    with open(args.out, "a") as fout:
        while n_written < remaining:
            b = min(args.batch_size, remaining - n_written)
            seqs = sample_batch(model, all_coords, target_chain_len, b, args.temperature, device)
            for i, seq in enumerate(seqs):
                recovery = sum(a == b for a, b in zip(native, seq)) / len(seq)
                idx = have + n_written + i
                fout.write(f">esmif_{idx}, score=nan, recovery={recovery:.4f}, T={args.temperature}\n{seq}\n")
            fout.flush()
            n_written += b
            if n_written % (args.batch_size * 20) == 0 or n_written == remaining:
                dt = time.time() - t0
                print(f"[esmif] {n_written}/{remaining} done ({dt:.0f}s, {dt/n_written:.3f}s/seq)")

    print(f"[esmif] done: {args.out} now has {have + n_written} sequences")


if __name__ == "__main__":
    main()
