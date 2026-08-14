#!/usr/bin/env python3
"""Score the native epitopes themselves, as the reference the designs lack.

No design matches its native exactly (the P1 methionine artifact guarantees it),
so the design table alone cannot answer "are these better binders than the real
epitope?". This scores the 21 native epitopes with both predictors.
"""
import os, sys
import numpy as np, pandas as pd
ROOT="/global/scratch/users/sergiomar10/if-mhc"
T=pd.read_csv(f"{ROOT}/outputs/skempi_if/peptides_to_score.csv")
nat=T[["complex","native","mhcflurry_allele","esmcba_allele"]].drop_duplicates()
nat=nat.rename(columns={"native":"seq"})
print(f"{len(nat)} native epitopes")
if sys.argv[1]=="mhcflurry":
    os.environ.setdefault("MHCFLURRY_DATA_DIR","/global/scratch/users/sergiomar10/mhcflurry_models")
    from mhcflurry import Class1PresentationPredictor
    p=Class1PresentationPredictor.load(); out=[]
    for al,g in nat.groupby("mhcflurry_allele"):
        r=p.predict(peptides=g.seq.tolist(), alleles=[al], verbose=0)
        r=r[["peptide","affinity","presentation_score"]].rename(columns={"peptide":"seq"})
        r["mhcflurry_allele"]=al; out.append(r)
    pd.concat(out).to_csv(f"{ROOT}/outputs/skempi_if/native_mhcflurry.csv",index=False)
else:
    sys.path.insert(0,f"{ROOT}/py")
    os.environ.setdefault("HF_HOME","/global/scratch/users/sergiomar10/hf_cache")
    import torch
    from huggingface_hub import hf_hub_download, list_repo_files
    from esm.models.esmc import ESMC
    from score_esmcba import ESMBA, resolve_checkpoint, REPO
    dev="cpu"; files=list_repo_files(REPO)
    base=ESMC.from_pretrained("esmc_300m").to(dev).eval(); out=[]
    for al,g in nat.groupby("esmcba_allele"):
        m=ESMBA(base).to(dev)
        ck=torch.load(hf_hub_download(REPO,resolve_checkpoint(al,files)),map_location=dev)
        m.load_state_dict({k:v for k,v in ck.items() if not k.startswith("mask_head")},strict=False)
        m.eval()
        enc=base.tokenizer(g.seq.tolist(),return_tensors="pt",padding=True)
        with torch.no_grad():
            p_,_=m(enc.input_ids,attention_mask=enc.attention_mask,return_embedding=True)
        out.append(pd.DataFrame({"seq":g.seq.tolist(),"esmcba_pred":p_.float().numpy(),"esmcba_allele":al}))
    pd.concat(out).to_csv(f"{ROOT}/outputs/skempi_if/native_esmcba.csv",index=False)
print("done")
