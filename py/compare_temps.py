#!/usr/bin/env python3
"""Direct T=0.1 vs T=0.3 comparison of every panel conclusion, from saved tables."""
import numpy as np, pandas as pd
from scipy import stats
ROOT="/global/scratch/users/sergiomar10/if-mhc"; OUT=f"{ROOT}/outputs/skempi_if"
MODELS=["esmif","proteinmpnn","proteinmpnn_nomhc","ligandmpnn"]

def load(t):
    R=pd.read_csv(f"{OUT}/panel_replication_positions_skempi_T{t}.csv")
    C=R.groupby(["complex","arm","group"]).agg(recovery=("recovery","mean"),
        chem_match=("chem_match","mean"), ent20=("ent20","mean")).reset_index()
    return R,C

rows=[]
for t in ["0.1","0.3"]:
    R,C=load(t)
    piv=C[C.arm=="full"].pivot(index="complex",columns="group",values="recovery")
    pc =C[C.arm=="full"].pivot(index="complex",columns="group",values="chem_match")
    w=lambda a,b,alt="greater": stats.wilcoxon(a,b,alternative=alt)[1]
    d=C.pivot_table(index="complex",columns="arm",values="recovery")
    cell=R[R.group!="P1"].groupby(["complex","model","arm"]).agg(
        recovery=("recovery","mean"),L=("L","first")).reset_index()
    ci=cell[(cell.arm=="full")&(~cell["complex"].isin(
        ["3C60_CD_AB","3QIB_ABP_CD","4OZG_ABJ_GH","4P23_CD_AB","4P5T_CD_AB"]))&cell.L.isin([9,10])]
    site=R[R.group!="P1"].pivot_table(index=["complex","arm","pos"],columns="model",values="recovery").dropna()
    rhos=[stats.spearmanr(site[a],site[b])[0] for i,a in enumerate(MODELS) for b in MODELS[i+1:] if a in site and b in site]
    rows.append(dict(T=t,
      P2_full=piv["P2"].mean(), interior_full=piv["interior"].mean(),
      P2_p=min(1,2*w(piv["P2"],piv["interior"])),
      Pom_id=piv["Pomega"].mean(), Pom_p=min(1,2*w(piv["Pomega"],piv["interior"])),
      Pom_chem=pc["Pomega"].mean(), Pom_chem_p=min(1,2*w(pc["Pomega"],pc["interior"])),
      TCR_delta=(d["full"]-d["notcr"]).mean(), TCR_n=int((d["full"]>d["notcr"]).sum()),
      TCR_p=w(d["full"],d["notcr"]),
      len_all=stats.pearsonr(cell[cell.arm=="full"].L,cell[cell.arm=="full"].recovery)[0],
      len_classI_910=stats.pearsonr(ci.L,ci.recovery)[0],
      len_classI_p=stats.pearsonr(ci.L,ci.recovery)[1],
      site_rho=np.mean(rhos)))
D=pd.DataFrame(rows).set_index("T").T
print("="*78); print("PANEL CONCLUSIONS: T=0.1 vs T=0.3 (28 SKEMPI complexes, same protocol)"); print("="*78)
print(D.round(4).to_string())

print("\n" + "="*78); print("DIVERSITY: what the temperature actually changed"); print("="*78)
for t,tag in [("0.1","t01"),("0.3","t03")]:
    tot=[]
    for m in MODELS:
        f=f"{ROOT}/designs/skempi/{tag}/{m}.csv.gz"
        try: d=pd.read_csv(f,usecols=["complex","arm","seq"])
        except Exception: continue
        u=d.groupby(["complex","arm"]).seq.nunique().mean()
        tot.append((m,u))
    print(f"  T={t}: mean unique sequences per (complex,arm) cell -> " +
          ", ".join(f"{m.split('_')[0][:9]}={u:.0f}" for m,u in tot))
