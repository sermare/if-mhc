#!/usr/bin/env python3
"""Emit 00_native_register_components.ipynb — component-wise structural comparison of the two native
pMHC-TCR complexes (6AM5/GIG vs 6AMU/DRG), on the crystals and across their unbiased MD (300 & 370 K).
Answers: (Q1) what was actually simulated (apo vs holo, and what fed RFdiffusion); (Q2) where the two
registers differ, component by component, with figures + tables — crystal deltas and MD distributions."""
import nbformat as nbf
nb = nbf.v4.new_notebook(); C = []
def md(t): C.append(nbf.v4.new_markdown_cell(t))
def co(t): C.append(nbf.v4.new_code_cell(t))

md(r"""# Where do the two native registers differ? — component-wise Cα-RMSD (crystals + MD)

**System.** HLA-A\*02 · DMF5 TCR. Two 10-mer epitopes in the *same* groove, *different* register:
**GIG** `SMLGIGIVPV` (crystal **6AM5**, P10 F-pocket anchor) vs **DRG** `MMWDRGLGMM` (crystal **6AMU**,
P9 anchor). This notebook decomposes the 6AM5↔6AMU difference into every structural sub-object — whole
complex, MHC (α1α2 / α3 / β-sheet floor / F-pocket), peptide, TCR (α/β, docking pose), and the six CDR
loops — on the **crystals** and across the **unbiased native MD** (300 K & 370 K, both epitopes).

---
## Q1 — What was simulated, and what fed RFdiffusion?

Reading the MD scripts (`py/apo_mhc_md.py`, `py/design_md.py`, `py/tcr_md.py`) and seed extractors
(`py/extract_md_seeds.py`, `py/extract_full_complex_seeds.py`):

| MD run | script | system simulated | peptide present? | TCR present? |
|---|---|---|---|---|
| **Native MD 300/370 K** (this notebook) | tamarind OpenMM ff19SB | **full holo complex** MHC+β2m+**peptide**+TCRα+TCRβ | ✅ yes | ✅ yes |
| **Apo-MHC MD** | `apo_mhc_md.py` | **MHC α1α2α3 + β2m only** (peptide *and* TCR removed) | ❌ no | ❌ no |
| Design MD | `design_md.py` | relaxed design complex, peptide free, MHC/TCR restrained | ✅ yes | ✅ (restrained) |

**So the answer is: no.** There was **no** "apo (no-peptide) MHC–TCR complex" MD that was then used as an
RFdiffusion scaffold. The only *apo / no-epitope* simulation run was **MHC + β2m alone** (`apo_mhc_md.py`)
— and even that dropped the **TCR too**, not just the peptide. It was a **side-chain rotamer-flexibility
probe** (does Tyr116 / Lys146 in the F-pocket flip between GIG- and DRG-states with the groove empty?),
**not** a backbone that seeded RFdiffusion. Its finding: the pocket rotamers stay locked in their start
state — the register is peptide-induced, not intrinsic groove flexibility.

Every RFdiffusion campaign was seeded from a **peptide-containing** template — crystals, relaxed
snapshots, or **native/design MD frames that always retain the peptide** (`extract_md_seeds.py` writes
pMHC = MHC α1α2 + β2m + **peptide**; `extract_full_complex_seeds.py` keeps the TCR too). RFdiffusion then
partial-diffused or re-generated the peptide **inside a groove that already held one** — it was never run
on top of a relaxed empty/no-epitope complex.

---
## Q2 — Component-wise register difference

Two Cα-RMSD frames are reported (see `py/native_md_components.py` for the exact residue sets):

* **Conformation** — object Kabsch-superposed onto its counterpart by its *own* Cα → pure internal shape
  difference (a rigidity check).
* **Groove (pose)** — everything superposed on the invariant **MHC α1α2** Cα (the register-preserving
  *common groove frame* the whole project uses), then measured with **no further fit** → where the object
  sits relative to the groove. For the peptide this reproduces the canonical **2.87 Å** register split.
* **CDR loops** — framework-superposed (V-domain minus CDR3), isolating loop shape from docking pose.

Residues are addressed by **alignment to the 6AM5 crystal frame**, so the correspondence is gap-aware and
identical for crystals and the merged-chain MD topology.

> **Bottom line (spoiler, matches prior work):** the two complexes are structurally **identical within
> thermal noise everywhere except the peptide**. The MHC groove is a rigid socket (0.2–0.4 Å), and the
> TCR V-domains and *all six CDR loops* — including CDR3α/β — differ by only 0.3–0.8 Å. **The peptide
> (2.87 Å) is the one clear difference.** (An earlier draft of this notebook reported CDR α-loops at
> ~3.9 Å; that was a residue-alignment artifact in the loop code — comparing raw positional index instead
> of aligned residues across two chains that start at different author numbers. It is fixed here, and the
> corrected CDR3α = 0.59 Å reproduces the prior `plot_tcr_pose_space.py` value of 0.65 Å.)""")

co(r"""import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/ubuntu/if-mhc/py")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.patches import Patch
import native_md_components as M
plt.rcParams.update({"figure.dpi": 120, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.axisbelow": True})

# palette: GIG/6AM5 = teal, DRG/6AMU = amber; 300 K solid, 370 K light
COL = {"6AM5": "#1b9e9e", "6AMU": "#d9822b"}
TCOL = {("6AM5","300K"):"#127d7d", ("6AM5","370K"):"#7fd0d0",
        ("6AMU","300K"):"#b5651d", ("6AMU","370K"):"#f0b878"}

rcs = M.ref_chain_seqs()
C5 = M.load_crystal("6AM5")._set_canon(rcs)
CU = M.load_crystal("6AMU")._set_canon(rcs)
NF = 150
MD = {}
for pid, T, job in M.MD_JOBS:
    models, _ = M.load_md(job, NF); MD[(pid, T)] = models
print("crystals loaded; MD frames/traj:", {f"{p}_{t}": len(v) for (p,t),v in MD.items()})
print("GIG↔DRG peptide register separation (in-groove, crystals):",
      round(M.distance(C5, CU, "pep", "groove"), 2), "Å")""")

md("""## A · Crystal 6AM5 vs 6AMU — the static register difference, component by component""")

co(r"""ctab = pd.DataFrame(M.crystal_table(C5, CU))
ctab_disp = ctab.rename(columns={"rmsd":"6AM5↔6AMU Cα-RMSD (Å)","mode":"frame"})[
    ["component","frame","6AM5↔6AMU Cα-RMSD (Å)"]]
def _hl(v):
    return ("background-color:#fde7d6" if v>=2 else
            "background-color:#fff6e6" if v>=1 else "background-color:#e7f5f5")
_styler = ctab_disp.style
_mapfn = getattr(_styler, "map", None) or _styler.applymap  # pandas >=2.1 renamed applymap->map
_mapfn(_hl, subset=["6AM5↔6AMU Cα-RMSD (Å)"]).set_caption(
    "Table A — per-component Cα-RMSD between the two native crystals. "
    "Teal ≈ rigid (<1 Å) · amber = shifts (≥2 Å).").format({"6AM5↔6AMU Cα-RMSD (Å)":"{:.2f}"})""")

co(r"""# Figure A — crystal component RMSD, grouped MHC / peptide / TCR
order = ["MHC groove floor (basin)","MHC F-pocket (6)","MHC F-pocket (ext 10)","MHC groove α1α2",
         "MHC α3","MHC (α1α2α3)","Whole complex","Peptide (conformation)",
         "Peptide (in-groove/register)","TCR β","Whole TCR","TCR α","TCR (docking pose, groove frame)",
         "CDR1b","CDR2b","CDR3b","CDR1a","CDR2a","CDR3a"]
# normalise unicode in component names to match module output
name_map = {r["component"]: r["component"] for _,r in ctab.iterrows()}
vals = {r["component"]: r["rmsd"] for _,r in ctab.iterrows()}
labels = [c for c in [x.replace("a1a2","α1α2").replace("alpha3","α3") for x in vals] ]
# build directly from ctab in a sensible fixed order
grp = [("MHC / groove", ["MHC groove floor (basin)","MHC F-pocket (6)","MHC F-pocket (ext 10)",
                          "MHC groove a1a2","MHC alpha3","MHC (a1a2a3)"], "#1b9e9e"),
       ("peptide (the difference)", ["Peptide (conformation)","Peptide (in-groove/register)","Whole complex"], "#7570b3"),
       ("TCR body", ["TCR beta","TCR alpha","TCR Vdom (conformation)",
                     "TCR Vdom (docking pose, groove frame)","TCR (docking pose, groove frame)"], "#d9822b"),
       ("CDR loops", ["CDR1b","CDR2b","CDR3b","CDR1a","CDR2a","CDR3a"], "#b5651d")]
fig, ax = plt.subplots(figsize=(11,5)); x=0; ticks=[]; tlab=[]
for gname, comps, c in grp:
    for comp in comps:
        v = vals.get(comp, np.nan)
        ax.bar(x, v, color=c, edgecolor="k", linewidth=0.4)
        ax.text(x, v+0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)
        ticks.append(x); tlab.append(comp); x+=1
    x+=0.8
ax.axhline(2.87, ls="--", lw=1, color="#7570b3"); ax.text(0.2, 2.95, "peptide register split 2.87 Å",
    color="#7570b3", fontsize=8)
ax.axhline(1.0, ls=":", lw=0.8, color="gray")
ax.set_xticks(ticks); ax.set_xticklabels([t.replace("a1a2","α1α2").replace("alpha3","α3") for t in tlab],
    rotation=55, ha="right", fontsize=8)
ax.set_ylabel("6AM5↔6AMU Cα-RMSD (Å)")
ax.set_title("Crystal register difference by component — only the PEPTIDE clears the line; MHC groove, "
             "TCR V-domains and all CDR loops are rigid (<1 Å)")
ax.legend(handles=[Patch(color=c,label=g) for g,_,c in grp], fontsize=8, ncol=4, loc="upper center")
plt.tight_layout(); plt.savefig("/home/ubuntu/if-mhc/outputs/native_md_rmsd/figA_crystal_components.png",
    dpi=140, bbox_inches="tight"); plt.show()""")

md(r"""**Read-out (crystals).** The **MHC groove is a rigid, register-agnostic socket**: floor **0.22 Å**,
F-pocket **0.20 Å**, α1α2 **0.39 Å** — all far below the peptide's **2.87 Å** register split. The **TCR is
also essentially rigid**: V-domain conformation **0.74 Å**, and *every* CDR loop — CDR1/2/3α = 0.80/0.38/
**0.59 Å**, CDR1/2/3β = 0.27/0.41/0.47 Å — sits below 1 Å (framework-superposed). The only elevated TCR
number is the **whole-TCR docking pose 3.23 Å**, and that is dominated by the distal **constant domains**
pivoting on the elbow: restricting to the **V-domains gives 2.09 Å**, matching the prior MHC-superposed
Vα/Vβ value of 1.97 Å — which is *within* the TCR's own thermal spread (see reconciliation below). So the
**one clear structural difference between the two registers is the peptide** (in-groove 2.87 Å; even its
own-superposed *shape* 2.41 Å). β2m, MHC α3, and the TCR are the same complex.""")

md(r"""### Reconciliation — which is right, and by what method?

**The prior conclusion ("same except the peptide") is correct.** Two independent earlier analyses established it:

* **`plot_perresidue_rmsd.py`** — per-residue 6AM5↔6AMU deviation of the **MHC**, superposed on the
  invariant β-sheet floor. Backbone: F-pocket/arm **0.35 Å** vs elsewhere **0.32 Å** (ratio **1.1×** — flat
  everywhere). Every larger all-atom spike (Arg111 4.1 Å, Lys121, Gln72 …) is a **distal surface
  side-chain rotamer** 10–35 Å from the peptide anchor; only Tyr116/Lys146 (F-pocket) move near the
  peptide, and those are peptide-*induced* rotamers (apo-MHC MD never samples them). → the MHC does not
  structurally encode the register.
* **`plot_tcr_pose_space.py`** — framework-superposed **CDR3** loop conformation: crystal GIG↔DRG =
  **0.65 Å**, native-MD centroid separation 0.53 Å vs within-register spread 0.78–0.95 Å → **"SWAMPED by
  thermal spread."** TCR docking pose (MHC-superposed Vα/Vβ) = **1.97 Å**, again below the ~2 Å within-
  register MD spread → also swamped.

**My corrected numbers agree exactly** (CDR3α 0.59 ≈ 0.65; TCR V-pose 2.09 ≈ 1.97). The "clear TCR
difference" in the first draft was **not real** — it came from pairing CDR-loop residues by raw position
index across two α-chains that start at different author residues (6AM5 D 1–190 vs 6AMU D 2–198) and carry
internal gaps, so the loop was compared against shifted residues. Fixing the correspondence (canonical
alignment to the 6AM5 frame, the same mapping used for the MHC and peptide) collapses it to <1 Å.

**Method verdict:** trust the **alignment-based, register-preserving** comparison (used here and in
`plot_perresidue_rmsd.py` / `groove_frame_rmsd.py`). Positional-index matching silently mis-pairs residues
whenever two chains differ in numbering — which is exactly what produced the spurious 3.9 Å.""")

md(r"""## B · Unbiased MD (300 K & 370 K) — do the components hold their register under thermal motion?

For every MD frame we measure each component's Cα-RMSD to **both** crystals (`toGIG` = to 6AM5,
`toDRG` = to 6AMU), in the same frame as the crystal comparison. A component that *carries* the register
sits near its own crystal and far from the other; a rigid, register-agnostic component sits near **both**.

**Caveat on these MD trajectories (found later, applies here too):** each run is only 10 ns production
after 0.2 ns equilibration. Re-analysis of the same 4 trajectories found 3 of 4 still drifting (not
plateaued) with autocorrelation times up to 1.72 ns — giving effective independent-sample counts as low
as ~6-14 (not the raw 500 frames/traj used below). Component comparisons that are large relative to
crystal-level differences (Å-scale, not sub-Å) are unaffected; anything read at sub-0.3 Å precision from
this MD should be treated as provisional pending the longer (50 ns) re-runs in progress.""")

co(r"""# assemble MD summary + per-frame tables (reuse module logic)
ALL = [(n,k,m) for n,k,m in M.COMPONENTS] + [(n,f"{c}:{l}","framework") for n,c,l in M.CDRS]
rows=[]; frames=[]
for pid,T,job in M.MD_JOBS:
    models=MD[(pid,T)]
    for name,kind,mode in ALL:
        if mode=="framework":
            ch,lk=kind.split(":"); g,d=M.md_cdr_series(models,C5,CU,ch,lk)
        else:
            g,d=M.md_series(models,C5,CU,kind,mode)
        gm,dm=M.summ(g),M.summ(d)
        rows.append(dict(traj=f"{pid}_{T}",pid=pid,temp=T,component=name,mode=mode,
                         toGIG_mean=gm[0],toGIG_med=gm[1],toGIG_sd=gm[2],
                         toDRG_mean=dm[0],toDRG_med=dm[1],toDRG_sd=dm[2]))
        for i in range(len(models)):
            frames.append(dict(traj=f"{pid}_{T}",pid=pid,temp=T,component=name,toGIG=g[i],toDRG=d[i]))
S=pd.DataFrame(rows); F=pd.DataFrame(frames)
print("summary rows:",len(S),"| frame rows:",len(F))""")

co(r"""# Table B — peptide + key components, mean±sd toGIG / toDRG per trajectory
key=["Peptide (in-groove/register)","MHC groove floor (basin)","MHC F-pocket (6)",
     "TCR (docking pose, groove frame)","CDR3a","CDR3b"]
def fmt(m,s): return f"{m:.2f}±{s:.2f}"
tb=[]
for comp in key:
    for pid,T,_ in M.MD_JOBS:
        r=S[(S.component==comp)&(S.pid==pid)&(S["temp"]==T)].iloc[0]
        tb.append(dict(component=comp,traj=f"{pid}_{T}",
                       toGIG=fmt(r.toGIG_mean,r.toGIG_sd),toDRG=fmt(r.toDRG_mean,r.toDRG_sd)))
pd.DataFrame(tb).pivot(index="component",columns="traj",values=["toGIG","toDRG"]).style.set_caption(
    "Table B — MD Cα-RMSD (mean±sd Å) of each component to the GIG (6AM5) and DRG (6AMU) crystals. "
    "Cognate traj (e.g. 6AM5_* vs toGIG) should be small; non-cognate large if the component carries register.")""")

co(r"""# Figure B1 — per-component MD distributions: toGIG (top) and toDRG (bottom), 4 trajectories
comps=["Peptide (in-groove/register)","MHC groove floor (basin)","MHC F-pocket (6)","MHC α3",
       "TCR (docking pose, groove frame)","CDR3a","CDR3b","Whole complex"]
comps=[c.replace("α3","alpha3") for c in comps]
trajs=[(p,t) for p,t,_ in M.MD_JOBS]
fig,axes=plt.subplots(2,len(comps),figsize=(2.05*len(comps),6),sharex=True)
for j,comp in enumerate(comps):
    for row,metric in enumerate(["toGIG","toDRG"]):
        ax=axes[row,j]
        data=[F[(F.component==comp)&(F.pid==p)&(F["temp"]==t)][metric].values for p,t in trajs]
        bp=ax.boxplot(data,positions=range(len(trajs)),widths=0.6,patch_artist=True,showfliers=False)
        for k,(p,t) in enumerate(trajs):
            bp["boxes"][k].set(facecolor=TCOL[(p,t)],alpha=0.9,linewidth=0.5)
        for med in bp["medians"]: med.set(color="k",linewidth=1)
        if row==0: ax.set_title(comp.replace("alpha3","α3").replace("(in-groove/register)","\n(register)").replace(" (docking pose, groove frame)","\ndock-pose"),fontsize=7.5)
        if j==0: ax.set_ylabel(f"{metric}\nCα-RMSD (Å)")
        ax.set_xticks(range(len(trajs))); ax.set_xticklabels([f"{p[-1]}·{t[:3]}" for p,t in trajs],rotation=90,fontsize=6.5)
fig.suptitle("MD Cα-RMSD distributions to GIG(6AM5) [top] and DRG(6AMU) [bottom] — box=IQR, line=median  "
             "(x: 5/U · temp)",fontsize=11,y=1.0)
plt.tight_layout(); plt.savefig("/home/ubuntu/if-mhc/outputs/native_md_rmsd/figB1_md_distributions.png",
    dpi=140,bbox_inches="tight"); plt.show()""")

co(r"""# Figure B2 — register scatter (toGIG vs toDRG) for peptide, F-pocket, CDR3a: does MD cross?
panels=[("Peptide (in-groove/register)","peptide — the ONLY register carrier"),
        ("MHC F-pocket (6)","F-pocket — register-agnostic"),
        ("CDR3a","CDR3α — register-agnostic (corrected)")]
fig,axes=plt.subplots(1,3,figsize=(13,4.3))
for ax,(comp,title) in zip(axes,panels):
    for p,t in trajs:
        sub=F[(F.component==comp)&(F.pid==p)&(F["temp"]==t)]
        ax.scatter(sub.toGIG,sub.toDRG,s=10,alpha=0.55,color=TCOL[(p,t)],
                   label=f"{p}·{t}",edgecolor="none")
    lim=max(ax.get_xlim()[1],ax.get_ylim()[1]); ax.plot([0,lim],[0,lim],ls="--",lw=0.8,color="gray")
    # crystal points
    cg=M.distance(C5,CU,*[(k,m) for n,k,m in M.COMPONENTS if n==comp][0]) if comp!="CDR3a" else None
    ax.set_xlabel("Cα-RMSD to GIG / 6AM5 (Å)"); ax.set_ylabel("Cα-RMSD to DRG / 6AMU (Å)")
    ax.set_title(title,fontsize=9.5); ax.legend(fontsize=6.5,loc="upper right")
axes[0].annotate("on diagonal = equidistant\n(no register preference)",(0.05,0.95),
    xycoords="axes fraction",va="top",fontsize=7,color="gray")
fig.suptitle("Register scatter — a register carrier separates off the diagonal (near its own crystal, far "
             "from the other); a rigid part hugs the diagonal",y=1.02,fontsize=10.5)
plt.tight_layout(); plt.savefig("/home/ubuntu/if-mhc/outputs/native_md_rmsd/figB2_register_scatter.png",
    dpi=140,bbox_inches="tight"); plt.show()""")

co(r"""# Figure B3 — "rigidity spectrum": crystal Δ vs MD thermal spread, per component
# For each component: crystal 6AM5↔6AMU RMSD (bar) and the mean within-traj sd (error) — is the register
# split bigger than thermal noise?
comp_order=["MHC groove floor (basin)","MHC F-pocket (6)","MHC groove a1a2","MHC alpha3",
            "Whole complex","Peptide (in-groove/register)","TCR Vdom (docking pose, groove frame)",
            "TCR (docking pose, groove frame)","CDR3b","CDR2b","CDR1b","CDR3a","CDR2a","CDR1a"]
comp_order=[c for c in comp_order if c.replace("α3","alpha3") in set(S.component)]
xc=[]; delta=[]; spread=[]; cols=[]
for comp in comp_order:
    cc=comp.replace("α3","alpha3")
    cr=ctab[ctab.component==cc]
    if cr.empty: continue
    d=cr.rmsd.values[0]
    sd=S[S.component==cc][["toGIG_sd","toDRG_sd"]].values.mean()
    xc.append(comp); delta.append(d); spread.append(sd)
    cols.append("#1b9e9e" if d<1 else "#7570b3" if d<2.5 else "#d9822b")
fig,ax=plt.subplots(figsize=(11,4.6)); xi=np.arange(len(xc))
ax.bar(xi,delta,color=cols,edgecolor="k",linewidth=0.4,label="crystal 6AM5↔6AMU Δ")
ax.errorbar(xi,delta,yerr=spread,fmt="none",ecolor="k",elinewidth=1,capsize=3,label="MD thermal sd (mean)")
ax.set_xticks(xi); ax.set_xticklabels([c.replace("alpha3","α3") for c in xc],rotation=55,ha="right",fontsize=8)
ax.set_ylabel("Cα-RMSD (Å)")
ax.set_title("Register split vs thermal noise — only the PEPTIDE's crystal Δ clears its MD spread; MHC "
             "parts, the TCR pose and every CDR loop do not (their Δ ≲ thermal sd)")
ax.legend(fontsize=8); plt.tight_layout()
plt.savefig("/home/ubuntu/if-mhc/outputs/native_md_rmsd/figB3_rigidity_spectrum.png",dpi=140,bbox_inches="tight")
plt.show()""")

co(r"""# Full summary table (all components, all trajectories) + save CSVs
S_out=S.copy()
for c in ["toGIG_mean","toGIG_med","toGIG_sd","toDRG_mean","toDRG_med","toDRG_sd"]:
    S_out[c]=S_out[c].round(2)
S_out.to_csv("/home/ubuntu/if-mhc/outputs/native_md_rmsd/md_component_rmsd_summary.csv",index=False)
ctab.to_csv("/home/ubuntu/if-mhc/outputs/native_md_rmsd/crystal_component_rmsd.csv",index=False)
print("saved crystal_component_rmsd.csv + md_component_rmsd_summary.csv + 4 figures to "
      "outputs/native_md_rmsd/")
S_out[S_out.component.isin(["Peptide (in-groove/register)","MHC groove floor (basin)","MHC F-pocket (6)",
    "MHC α3".replace("α3","alpha3"),"TCR (docking pose, groove frame)","CDR3a","CDR3b","Whole complex"])]\
    .sort_values(["component","traj"]).reset_index(drop=True)""")

md(r"""## Conclusions

**Q1.** MD was run on the **full holo complex** (MHC + β2m + **peptide** + TCR) at 300 & 370 K. The only
*no-peptide* simulation was **apo MHC + β2m** (no TCR either) — a rotamer-flexibility probe, **not** an
RFdiffusion scaffold. **No RFdiffusion run was seeded on a relaxed no-epitope complex**; every RFd seed
retained the peptide.

**Q2 — is there a clear component-wise difference between the two registers? Yes — and it is the peptide,
essentially only the peptide.** This confirms the prior "same except the peptide" conclusion.

* **MHC is a rigid, register-agnostic socket.** Groove floor 0.22 Å, F-pocket 0.20 Å, α1α2 0.39 Å, α3
  1.17 Å — all well under the peptide split, and in MD each MHC part sits equally close to *both* crystals
  (hugs the diagonal). Crystal Δ does **not** clear the MD thermal spread → MHC does not encode register.
* **The peptide carries the register** — 2.87 Å in-groove between crystals, and in MD each epitope stays in
  its own basin (cognate ≈ 0.7–1.1 Å, non-cognate ≈ 3 Å) at both temperatures; 370 K widens the spread but
  never crosses. This is the C-terminal (P9↔P10) F-pocket anchor swap.
* **The TCR is also rigid between the two structures.** V-domain conformation 0.74 Å; **all six CDR loops
  0.3–0.8 Å** (CDR3α 0.59, CDR3β 0.47). The V-domain docking pose (2.09 Å) is *within* the TCR's own
  ~2 Å within-register thermal spread → not a register signal. The whole-TCR 3.23 Å is just the distal
  constant domains pivoting on the elbow. In MD every CDR (α and β) hugs the diagonal.
* **Correction:** an earlier draft reported CDR α-loops at ~3.9 Å — a residue-alignment artifact (positional
  index vs aligned residue across differently-numbered chains). Fixed; corrected CDR3α 0.59 Å reproduces the
  prior `plot_tcr_pose_space.py` 0.65 Å.

Consistent with `plot_perresidue_rmsd.py`, `plot_tcr_pose_space.py`, and the campaign summary: *structurally
the register is a peptide-C-terminal event on a rigid MHC and a rigid TCR*. (The TCR's causal role is
**dynamic** — steered MD shows engagement lowers the crossing barrier ~17 % — not a static conformational
difference.) Data + figures: `outputs/native_md_rmsd/`.""")

nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "esmfold2", "display_name": "esmfold2", "language": "python"}
with open("/home/ubuntu/if-mhc/00_native_register_components.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote 00_native_register_components.ipynb")
