#!/usr/bin/env python3
"""Build the unique-peptide table for external scoring (nb06 replication).

One row per (complex, arm, model, peptide) with its sampling count, restricted to
complexes whose allele both MHCflurry and ESMCBA support. The two mouse H-2Ld
complexes and the five class II complexes are dropped: MHCflurry is class I and
has no H-2Ld model, and ESMCBA's checkpoints are per-HLA with no counterpart.
"""
import glob
import pandas as pd

ROOT = "/global/scratch/users/sergiomar10/if-mhc"
DES = f"{ROOT}/designs/skempi/t01"

ALLELE = {"HLA-A2": ("HLA-A*02:01", "A0201"),
          "HLA-B35": ("HLA-B*35:01", "B3501"),
          "HLA-B8": ("HLA-B*08:01", "B0801")}


def allele_of(annotation):
    for key, val in ALLELE.items():
        if annotation.upper().startswith(key.upper() + " "):
            return val
    return (None, None)


mut = pd.read_csv(f"{ROOT}/inputs/skempi/skempi_tcr_pmhc_mutations.csv")
ann = mut.drop_duplicates("#Pdb").set_index("#Pdb")["Protein 1"]
man = pd.read_csv(f"{ROOT}/inputs/skempi/manifest.csv")
man["ann"] = man["complex"].map(ann)
man[["mhcflurry_allele", "esmcba_allele"]] = man["ann"].apply(
    lambda s: pd.Series(allele_of(str(s))))
keep = man.dropna(subset=["mhcflurry_allele"])
print(f"{len(keep)}/{len(man)} complexes have a supported class I allele:")
print(keep.groupby("mhcflurry_allele")["complex"].agg(["size", list]).to_string())
dropped = man[man.mhcflurry_allele.isna()]
print(f"\ndropped ({len(dropped)}): "
      + ", ".join(f"{r['complex']} [{r['ann']}]" for _, r in dropped.iterrows()))

amap = keep.set_index("complex")[["mhcflurry_allele", "esmcba_allele", "pep_seq"]]
frames = []
for f in sorted(glob.glob(f"{DES}/*.csv.gz")):
    model = f.split("/")[-1].replace(".csv.gz", "")
    d = pd.read_csv(f, usecols=["complex", "arm", "seq"])
    d = d[d["complex"].isin(amap.index)]
    g = (d.groupby(["complex", "arm", "seq"]).size()
         .reset_index(name="count"))
    g["model"] = model
    frames.append(g)

T = pd.concat(frames, ignore_index=True)
T = T.join(amap, on="complex")
T = T.rename(columns={"pep_seq": "native"})
T["is_native"] = T["seq"] == T["native"]
T.to_csv(f"{ROOT}/outputs/skempi_if/peptides_to_score.csv", index=False)

pairs = T[["seq", "mhcflurry_allele"]].drop_duplicates()
pairs.to_csv(f"{ROOT}/outputs/skempi_if/mhcflurry_input_pairs.csv", index=False)
print(f"\n{len(T):,} (complex,arm,model,peptide) rows")
print(f"{len(pairs):,} unique (peptide, allele) pairs to score")
print(f"peptide lengths present: {sorted(T.seq.str.len().unique())}")
