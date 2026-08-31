#!/usr/bin/env python3
"""Score the designed peptides with ESMCBA (nb06 replication).

Reuses the model definition from ESMCBA/embeddings_generation.py verbatim -- an
ESM-C 300M trunk with a mean-pooled linear regression head -- but reads peptides
from a file instead of argv (10k peptides will not fit on a command line) and
skips the UMAP stage. Checkpoints are per-allele and are resolved by name from
the HuggingFace repo, with --encoding epitope semantics: the peptide alone is
fed, the allele is carried by the checkpoint.

Saves predictions and the 960-d pooled embeddings for downstream analysis.
"""
import os, re, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = "/global/scratch/users/sergiomar10/if-mhc"
# Phase selection: which (dataset, temperature) run to analyse. Defaults to the
# T=0.1 SKEMPI phase so existing invocations keep working; set SK_DATASET /
# SK_TEMP to point the same analysis at another phase.
DATASET = os.environ.get("SK_DATASET", "skempi")
TEMP = os.environ.get("SK_TEMP", "0.1")
TAG = "t" + TEMP.replace(".", "")
SUF = f"_{DATASET}_T{TEMP}"

os.environ.setdefault("HF_HOME", "/global/scratch/users/sergiomar10/hf_cache")
from huggingface_hub import hf_hub_download, list_repo_files
from esm.models.esmc import ESMC

REPO = "smares/ESMCBA"


class ESMBA(nn.Module):
    """ESM-based regressor returning both predictions and pooled embeddings."""

    def __init__(self, base_model, dropout=0.3):
        super().__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(dropout)
        self.regression_head = nn.Linear(960, 1)

    def forward(self, input_ids, attention_mask=None, return_embedding=False):
        outputs = self.base_model.forward(input_ids)
        hidden = outputs.hidden_states[-1].to(torch.float32)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / (mask.sum(1) + 1e-8)
        pred = self.regression_head(self.dropout(pooled)).squeeze(-1)
        return (pred, pooled.detach()) if return_embedding else pred


def resolve_checkpoint(allele, files):
    """Pick the per-allele final checkpoint from the HF repo listing."""
    cands = [f for f in files
             if f.endswith(".pth") and f"HLA{allele}" in f and "_final" in f]
    if not cands:
        cands = [f for f in files if f.endswith(".pth") and allele in f]
    if not cands:
        raise SystemExit(f"no ESMCBA checkpoint for {allele}")
    return sorted(cands, key=len)[0]


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    T = pd.read_csv(f"{ROOT}/outputs/skempi_if/peptides_to_score{SUF}.csv")
    files = list_repo_files(REPO)
    base = ESMC.from_pretrained("esmc_300m").to(dev).eval()

    out = []
    for allele, g in T.groupby("esmcba_allele"):
        peps = sorted(set(g.seq.astype(str)))
        ckpt_name = resolve_checkpoint(allele, files)
        print(f"\n{allele}: {len(peps):,} unique peptides\n  checkpoint {ckpt_name}", flush=True)
        path = hf_hub_download(REPO, ckpt_name)
        model = ESMBA(base).to(dev)
        ck = torch.load(path, map_location=dev)
        filtered = {k: v for k, v in ck.items() if not k.startswith("mask_head")}
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        head = [k for k in filtered if "regression_head" in k]
        print(f"  loaded: {len(filtered)} tensors, regression_head present={bool(head)}, "
              f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        model.eval()

        class DS(Dataset):
            def __len__(self): return len(peps)
            def __getitem__(self, i): return peps[i]

        def collate(batch):
            return batch, base.tokenizer(batch, return_tensors="pt", padding=True)

        preds, embs, seqs = [], [], []
        with torch.no_grad():
            for s, enc in DataLoader(DS(), batch_size=256, collate_fn=collate):
                p, e = model(enc.input_ids.to(dev),
                             attention_mask=enc.attention_mask.to(dev),
                             return_embedding=True)
                preds.append(p.float().cpu()); embs.append(e.float().cpu()); seqs += list(s)
        pr = torch.cat(preds).numpy(); em = torch.cat(embs).numpy()
        np.save(f"{ROOT}/outputs/skempi_if/esmcba_emb_{allele}{SUF}.npy", em)
        out.append(pd.DataFrame({"seq": seqs, "esmcba_pred": pr, "esmcba_allele": allele}))
        print(f"  pred range {pr.min():.3f}..{pr.max():.3f}  mean {pr.mean():.3f}", flush=True)

    R = pd.concat(out, ignore_index=True)
    R.to_csv(f"{ROOT}/outputs/skempi_if/esmcba_scores{SUF}.csv", index=False)
    print(f"\nscored {len(R):,} peptides -> outputs/skempi_if/esmcba_scores.csv")


if __name__ == "__main__":
    main()
