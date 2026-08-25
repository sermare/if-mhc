#!/usr/bin/env python3
"""Score the panel's unique peptides with ESMCBA (HLA-A0201 checkpoint) -- embeddings, regression
prediction, and one jointly-fit UMAP over the whole set. Run in the esmcba conda env.

Usage: /home/ubuntu/miniforge3/envs/esmcba/bin/python3 py/esmcba_score.py
"""
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from esm.models.esmc import ESMC
from torch.utils.data import Dataset, DataLoader
from umap import UMAP

ROOT = Path("/home/ubuntu/if-mhc")
OUT_DIR = ROOT / "outputs/analysis"
HLA = "A0201"

ckpt_matches = glob.glob(str(Path.home() / f".cache/huggingface/hub/models--smares--ESMCBA/snapshots/*/*HLA{HLA}_*{HLA}_final.pth"))
assert ckpt_matches, f"no cached ESMCBA checkpoint found for {HLA}"
model_path = ckpt_matches[0]
print(f"using checkpoint: {model_path}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

class ESMBA(nn.Module):
    def __init__(self, base_model, dropout=0.3):
        super().__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(dropout)
        self.regression_head = nn.Linear(960, 1)

    def forward(self, input_ids, attention_mask=None, return_embedding=False):
        outputs = self.base_model.forward(input_ids)
        hidden_states = outputs.hidden_states[-1].to(torch.float32)
        mask = attention_mask.unsqueeze(-1).float()
        masked_hidden = hidden_states * mask
        sum_emb = masked_hidden.sum(dim=1)
        sum_mask = mask.sum(dim=1)
        pooled = sum_emb / (sum_mask + 1e-8)
        dropped = self.dropout(pooled)
        pred = self.regression_head(dropped).squeeze(-1)
        if return_embedding:
            return pred, pooled.detach()
        return pred

ckpt = torch.load(model_path, map_location=device)
base = ESMC.from_pretrained("esmc_300m").to(device)
model = ESMBA(base).to(device)
filtered = {k: v for k, v in ckpt.items() if not k.startswith("mask_head")}
model.load_state_dict(filtered, strict=False)
model.eval()
base_model = model.base_model

peptides = [l.strip() for l in open(OUT_DIR / "panel_unique_peptides_for_scoring.txt") if l.strip()]
print(f"scoring {len(peptides)} peptides with ESMCBA ({HLA})")

class PeptideDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

def collate_fn(batch):
    enc = base_model.tokenizer(batch, return_tensors="pt", padding=True)
    return batch, enc

loader = DataLoader(PeptideDataset(peptides), batch_size=32, shuffle=False, collate_fn=collate_fn)

all_seqs, all_preds, all_embeds = [], [], []
with torch.no_grad():
    for i, (seqs, enc) in enumerate(loader):
        input_ids = enc.input_ids.to(device)
        attn_mask = enc.attention_mask.to(device)
        preds, embeds = model(input_ids, attention_mask=attn_mask, return_embedding=True)
        all_seqs.extend(seqs)
        all_preds.append(preds.cpu())
        all_embeds.append(embeds.cpu())
        if i % 20 == 0:
            print(f"  batch {i}/{len(loader)}", flush=True)

all_preds = torch.cat(all_preds).numpy()
all_embeds = torch.cat(all_embeds).numpy()
print(f"done: {all_embeds.shape}")

emb_path = OUT_DIR / "panel_unique_peptides_esmcba_embeddings.npy"
np.save(emb_path, all_embeds)
print(f"wrote {emb_path}")

reducer = UMAP(n_components=2, n_neighbors=15, random_state=42)
umap_coords = reducer.fit_transform(all_embeds)

result_df = pd.DataFrame({
    "peptide": all_seqs,
    "esmcba_prediction": all_preds,
    "UMAP_1": umap_coords[:, 0],
    "UMAP_2": umap_coords[:, 1],
})
out = OUT_DIR / "panel_unique_peptides_esmcba.csv"
result_df.to_csv(out, index=False)
print(f"wrote {out}: {len(result_df)} rows")
