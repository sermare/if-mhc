#!/usr/bin/env python3
"""Score the peptides that are in the metadata tables but not yet in the ESMCBA tables.

The panel and SKEMPI corpora arrived with their own ESMCBA runs; the 6AM5/6AMU designs did
not. Rather than re-embed ~30k peptides that are already scored, this fills only the gap and
then refits the UMAP over the combined set (the UMAP is a joint fit, so it has to be redone
whenever rows are added -- projecting into an old fit would place new peptides on coordinates
that fit never saw).

Same checkpoint, pooling and UMAP settings as py/esmcba_score.py, so the appended rows are
computed identically to the ones already there.

  /home/ubuntu/miniforge3/envs/esmcba/bin/python py/esmcba_score_missing.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from esm.models.esmc import ESMC
from torch.utils.data import DataLoader, Dataset
from umap import UMAP

ROOT = Path("/home/ubuntu/if-mhc")
ANA = ROOT / "outputs/analysis"
HLA = "A0201"


class ESMBA(nn.Module):
    def __init__(self, base_model, dropout=0.3):
        super().__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(dropout)
        self.regression_head = nn.Linear(960, 1)

    def forward(self, input_ids, attention_mask=None, return_embedding=False):
        outputs = self.base_model.forward(input_ids)
        hidden = outputs.hidden_states[-1].to(torch.float32)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        pred = self.regression_head(self.dropout(pooled)).squeeze(-1)
        return (pred, pooled.detach()) if return_embedding else pred


def load_model():
    ckpts = glob.glob(str(Path.home() /
                         f".cache/huggingface/hub/models--smares--ESMCBA/snapshots/"
                         f"*/*HLA{HLA}_*{HLA}_final.pth"))
    assert ckpts, f"no cached ESMCBA checkpoint for {HLA}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"checkpoint: {ckpts[0]}\ndevice: {device}")
    model = ESMBA(ESMC.from_pretrained("esmc_300m").to(device)).to(device)
    ckpt = torch.load(ckpts[0], map_location=device)
    model.load_state_dict({k: v for k, v in ckpt.items()
                           if not k.startswith("mask_head")}, strict=False)
    model.eval()
    return model, device


class PeptideDataset(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        return self.seqs[i]


def score(model, device, peptides):
    tok = model.base_model.tokenizer

    def collate(batch):
        return batch, tok(batch, return_tensors="pt", padding=True)

    loader = DataLoader(PeptideDataset(peptides), batch_size=32, shuffle=False,
                        collate_fn=collate)
    seqs, preds, embs = [], [], []
    with torch.no_grad():
        for i, (b, enc) in enumerate(loader):
            p, e = model(enc.input_ids.to(device),
                         attention_mask=enc.attention_mask.to(device), return_embedding=True)
            seqs.extend(b); preds.append(p.cpu()); embs.append(e.cpu())
            if i % 20 == 0:
                print(f"  batch {i}/{len(loader)}", flush=True)
    return seqs, torch.cat(preds).numpy(), torch.cat(embs).numpy()


def main() -> None:
    model = device = None
    for suffix in ["", "_mhconly"]:
        meta = pd.read_csv(ANA / f"panel_unique_peptides_metadata{suffix}.csv")
        ec = pd.read_csv(ANA / f"panel_unique_peptides_esmcba{suffix}.csv")
        emb = np.load(ANA / f"panel_unique_peptides_esmcba_embeddings{suffix}.npy")
        assert len(ec) == len(emb), f"{suffix}: table/embedding mismatch"

        missing = sorted(set(meta["peptide"]) - set(ec["peptide"]))
        print(f"\n=== arm {suffix or '(full)'}: {len(missing)} peptides to score ===")
        if not missing:
            print("nothing to do")
            continue
        if model is None:
            model, device = load_model()

        seqs, preds, embs = score(model, device, missing)
        peptides = np.concatenate([ec["peptide"].values, np.array(seqs)])
        allpred = np.concatenate([ec["esmcba_prediction"].values, preds])
        allemb = np.vstack([emb, embs])

        print(f"refitting UMAP over {allemb.shape[0]:,} embeddings ...")
        coords = UMAP(n_components=2, n_neighbors=15, random_state=42).fit_transform(allemb)
        pd.DataFrame({"peptide": peptides, "esmcba_prediction": allpred,
                      "UMAP_1": coords[:, 0], "UMAP_2": coords[:, 1]}).to_csv(
            ANA / f"panel_unique_peptides_esmcba{suffix}.csv", index=False)
        np.save(ANA / f"panel_unique_peptides_esmcba_embeddings{suffix}.npy", allemb)
        print(f"wrote panel_unique_peptides_esmcba{suffix}.csv: {len(peptides):,} rows")


if __name__ == "__main__":
    main()
