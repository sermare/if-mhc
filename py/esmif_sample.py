#!/usr/bin/env python3
"""Batched ESM-IF1 epitope sampling for the SKEMPI TCR/pMHC set.

The stock esm.inverse_folding.sample_sequence_in_complex re-runs the GVP
encoder for every sample, which is the whole cost when only ~10 positions are
being decoded over an ~830-residue complex. Here the encoder runs ONCE per
(complex, arm); the encoder output is broadcast across a batch and the epitope
positions are decoded autoregressively in parallel, which is what makes 10k
samples per complex affordable.

Semantics otherwise match sample_sequence_in_complex: the target chain is
concatenated first, the remaining chains follow behind <pad> spacers, and only
the epitope positions carry <mask>.

Usage:
  esmif_sample.py --arm full --nseq 10000 --temp 0.1 [--complex 1BD2_ABC_DE]
"""
import argparse, os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import esm
from esm.inverse_folding.multichain_util import _concatenate_coords, load_complex_coords
from esm.inverse_folding.util import CoordBatchConverter

ROOT = "/global/scratch/users/sergiomar10/if-mhc"
SKDIR = {"skempi": f"{ROOT}/inputs/skempi", "pmhc25": f"{ROOT}/inputs/pmhc25",
         "focus6am": f"{ROOT}/inputs/focus6am"}
SK = SKDIR["skempi"]   # default; overridden per-run by --dataset
AA = "ACDEFGHIKLMNPQRSTVWY"


def build_encoder_out(model, coords_cat, device):
    """Run the GVP encoder once for the concatenated complex."""
    bc = CoordBatchConverter(model.decoder.dictionary)
    batch_coords, confidence, _, _, padding_mask = bc([(coords_cat, None, None)],
                                                      device=device)
    with torch.no_grad():
        enc = model.encoder(batch_coords, padding_mask, confidence)
    return enc


def expand_encoder_out(enc, batch):
    """Broadcast a batch-1 encoder output across `batch` samples."""
    x = enc["encoder_out"][0]                       # T x 1 x C
    mask = enc["encoder_padding_mask"][0]           # 1 x T
    return {
        "encoder_out": [x.expand(-1, batch, -1)],
        "encoder_padding_mask": [mask.expand(batch, -1)],
        "encoder_embedding": [],
        "encoder_states": [],
    }


def sample_batch(model, enc, total_len, design_idx, batch, temperature, device,
                 aa_token_ids, cath_idx, mask_idx, pad_idx):
    """Decode `batch` epitope sequences in parallel from one encoder pass.

    design_idx are 0-based positions within the concatenated complex; they must
    be a prefix-contiguous block starting at 0 (the target chain leads the
    concatenation), so a single left-to-right pass covers them.
    """
    enc_b = expand_encoder_out(enc, batch)
    # token stream is <cath> + the concatenated complex
    tokens = torch.full((batch, 1 + total_len), pad_idx, dtype=torch.long, device=device)
    tokens[:, 0] = cath_idx
    for i in design_idx:
        tokens[:, i + 1] = mask_idx

    incremental_state = {}
    keep = torch.full((len(model.decoder.dictionary),), float("-inf"), device=device)
    keep[aa_token_ids] = 0.0

    for i in design_idx:
        with torch.no_grad():
            logits, _ = model.decoder(tokens[:, : i + 1], enc_b,
                                      incremental_state=incremental_state)
        # decoder returns B x vocab x tgt_len; take the newest step
        logits = logits[:, :, -1].float() / temperature + keep
        probs = F.softmax(logits, dim=-1)
        tokens[:, i + 1] = torch.multinomial(probs, 1).squeeze(-1)

    toks = tokens[:, 1:][:, design_idx].cpu().numpy()
    d = model.decoder.dictionary
    return ["".join(d.get_tok(int(t)) for t in row) for row in toks]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["full", "notcr"])
    ap.add_argument("--dataset", default="skempi", choices=["skempi", "pmhc25", "focus6am"])
    ap.add_argument("--nseq", type=int, default=10000)
    ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--complex", default=None, help="run a single complex id")
    ap.add_argument("--out", default=f"{ROOT}/outputs/skempi_if/esmif")
    ap.add_argument("--seed", type=int, default=37)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)

    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.eval().to(device)
    d = model.decoder.dictionary
    aa_ids = [d.get_idx(c) for c in AA]
    cath_idx, mask_idx, pad_idx = d.get_idx("<cath>"), d.get_idx("<mask>"), d.get_idx("<pad>")

    sk = SKDIR[a.dataset]
    man = pd.read_csv(f"{sk}/manifest.csv")
    if a.complex:
        man = man[man["complex"] == a.complex]
    print(f"[esmif/{a.arm}] {len(man)} complexes x {a.nseq} seqs @ T={a.temp} on {device}",
          flush=True)

    for _, r in man.iterrows():
        cid, pep_ch, pep_len = r["complex"], r["pep_chain"], int(r["pep_len"])
        dst = f"{a.out}/{cid}__{a.arm}.csv"
        if os.path.exists(dst + ".done"):
            print(f"  skip {cid} (done)", flush=True)
            continue
        t0 = time.time()
        chains = list(r[f"{a.arm}_chains"])
        fpath = f"{sk}/arm_{a.arm}/{cid}.pdb"
        coords, native = load_complex_coords(fpath, chains)
        coords_cat = _concatenate_coords(coords, pep_ch)
        # epitope = first pep_len positions of the target chain, which leads the
        # concatenation (true for both free and MHC-fused epitopes)
        design_idx = list(range(pep_len))
        enc = build_encoder_out(model, coords_cat, device)

        seqs = []
        while len(seqs) < a.nseq:
            b = min(a.batch, a.nseq - len(seqs))
            seqs += sample_batch(model, enc, coords_cat.shape[0], design_idx, b,
                                 a.temp, device, aa_ids, cath_idx, mask_idx, pad_idx)
        nat = native[pep_ch][:pep_len]
        rec = [sum(x == y for x, y in zip(s, nat)) / len(nat) for s in seqs]
        pd.DataFrame({"complex": cid, "arm": a.arm, "model": "esm_if1",
                      "temp": a.temp, "sample": range(len(seqs)),
                      "seq": seqs, "native": nat, "recovery": rec}).to_csv(dst, index=False)
        open(dst + ".done", "w").close()
        print(f"  {cid:14s} L={coords_cat.shape[0]:4d} pep={pep_len} "
              f"n={len(seqs)} uniq={len(set(seqs))} "
              f"rec={np.mean(rec):.3f} {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
