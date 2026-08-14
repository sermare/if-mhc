#!/usr/bin/env python3
"""Check the batched epitope sampler reproduces stock ESM-IF sampling.

esmif_sample.py reuses one encoder pass across a whole batch instead of
re-encoding per sample, so it has to be shown to draw from the same
distribution as esm.inverse_folding.multichain_util.sample_sequence_in_complex.

The stock helper builds its tensors on CPU regardless of where the model lives,
so the head-to-head runs on CPU for both. A second pass then confirms the
batched sampler gives the same distribution on GPU as on CPU.

Checks:
  1. greedy (T -> 0) sequence identical, batched vs stock
  2. per-position amino-acid frequencies agree over N draws at T
  3. batched-on-GPU agrees with batched-on-CPU
"""
import argparse, sys
import numpy as np
import pandas as pd
import torch

import esm
from esm.inverse_folding.multichain_util import (
    _concatenate_coords, load_complex_coords, sample_sequence_in_complex)

sys.path.insert(0, "/global/scratch/users/sergiomar10/if-mhc/py")
from esmif_sample import build_encoder_out, sample_batch, AA, SK

ap = argparse.ArgumentParser()
ap.add_argument("--complex", default="1BD2_ABC_DE")
ap.add_argument("--arm", default="full")
ap.add_argument("--n", type=int, default=32)
ap.add_argument("--temp", type=float, default=0.1)
a = ap.parse_args()

model, _ = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
model = model.eval()
d = model.decoder.dictionary
aa_ids = [d.get_idx(c) for c in AA]
cath, mask, pad = d.get_idx("<cath>"), d.get_idx("<mask>"), d.get_idx("<pad>")

man = pd.read_csv(f"{SK}/manifest.csv").set_index("complex").loc[a.complex]
pep_ch, pep_len = man["pep_chain"], int(man["pep_len"])
chains = list(man[f"{a.arm}_chains"])
coords, native = load_complex_coords(f"{SK}/arm_{a.arm}/{a.complex}.pdb", chains)
cat = _concatenate_coords(coords, pep_ch)
idx = list(range(pep_len))
nat = native[pep_ch][:pep_len]
print(f"{a.complex} arm={a.arm} L={cat.shape[0]} pep={pep_ch}({pep_len}) native={nat}\n")


def batched(dev, n, temp, seed):
    m = model.to(dev)
    torch.manual_seed(seed)
    enc = build_encoder_out(m, cat, dev)
    out = []
    while len(out) < n:
        b = min(64, n - len(out))
        out += sample_batch(m, enc, cat.shape[0], idx, b, temp, dev,
                            aa_ids, cath, mask, pad)
    return out


def freqs(seqs):
    m = np.zeros((pep_len, 20))
    for s in seqs:
        for i, c in enumerate(s):
            if c in AA:
                m[i, AA.index(c)] += 1
    return m / len(seqs)


def rec(seqs):
    return np.mean([sum(x == y for x, y in zip(s, nat)) / pep_len for s in seqs])


# --- check 1: greedy agreement (CPU vs CPU) ------------------------------
model = model.to("cpu")
g_mine = batched("cpu", 1, 1e-6, 0)[0]
torch.manual_seed(0)
g_stock = sample_sequence_in_complex(model, coords, pep_ch, temperature=1e-6)[:pep_len]
ok1 = g_mine == g_stock
print(f"[1] greedy   batched={g_mine}  stock={g_stock}   {'MATCH' if ok1 else 'MISMATCH'}")

# --- check 2: frequency agreement at T (CPU vs CPU) ----------------------
mine_cpu = batched("cpu", a.n, a.temp, 1)
torch.manual_seed(1)
stock = [sample_sequence_in_complex(model, coords, pep_ch, temperature=a.temp)[:pep_len]
         for _ in range(a.n)]
fm, fs = freqs(mine_cpu), freqs(stock)
tvd = 0.5 * np.abs(fm - fs).sum(axis=1)
noise = np.sqrt(1.0 / a.n)
print(f"\n[2] per-position total-variation distance, n={a.n} draws @ T={a.temp}"
      f"   (sampling noise floor ~{noise:.2f})")
for i in range(pep_len):
    print(f"    pos {i+1:2d} native={nat[i]}  TVD={tvd[i]:.3f}   "
          f"batched_top={AA[fm[i].argmax()]}({fm[i].max():.2f})   "
          f"stock_top={AA[fs[i].argmax()]}({fs[i].max():.2f})")
ok2 = tvd.mean() < 2 * noise
print(f"    mean TVD={tvd.mean():.3f} max={tvd.max():.3f} | "
      f"recovery batched={rec(mine_cpu):.3f} stock={rec(stock):.3f}")

# --- check 3: GPU reproduces CPU ----------------------------------------
ok3 = True
if torch.cuda.is_available():
    mine_gpu = batched("cuda", a.n, a.temp, 1)
    tvd3 = 0.5 * np.abs(freqs(mine_gpu) - fm).sum(axis=1)
    ok3 = tvd3.mean() < 2 * noise
    print(f"\n[3] batched GPU vs CPU: mean TVD={tvd3.mean():.3f} "
          f"recovery gpu={rec(mine_gpu):.3f} cpu={rec(mine_cpu):.3f}")
else:
    print("\n[3] no GPU visible -- skipped")

print(f"\nRESULT: greedy {'OK' if ok1 else 'FAIL'} | "
      f"freq {'OK' if ok2 else 'FAIL'} | gpu {'OK' if ok3 else 'FAIL'}")
sys.exit(0 if (ok1 and ok2 and ok3) else 1)
