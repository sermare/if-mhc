#!/usr/bin/env python3
"""Emit 09_q30_conditioning_arms.ipynb — the campaign-question notebook.

Renders, from py/q30_analysis.load_all(): the floor-frame basin scatter (headline), 3-reference
RMSD, N- vs C-terminal split, contact-satisfaction histograms, F-pocket register readout,
clustering + trace overlays, per-residue RMSF, and the physical-sanity filter. Re-runnable: it
reads whatever designs the arms have produced. Execute with:
  esmfold2/bin/jupyter nbconvert --to notebook --execute --inplace 09_q30_conditioning_arms.ipynb
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
def md(t): C.append(nbf.v4.new_markdown_cell(t))
def co(t): C.append(nbf.v4.new_code_cell(t))

md("""# Q30 conditioning — did interaction-only conditioning steer the basin?

**The question this campaign tests:** when we re-diffuse *only the peptide* inside a fixed pMHC-TCR
complex, conditioned on specific interactions (Q30α, CDR3β-Phe, the A/B-pocket N-anchor, an
N-terminal sliding window), do the rebuilt backbones come **off the toGIG≈toDRG diagonal and into a
basin** — or do they drift equidistant from both (the prior failure)? This is **not** generic
structure QC.

**Make-or-break — the frame.** Every design is superposed on the **MHC β-sheet groove-floor Cα only**
(helices and the mobile α2 short arm His145–Val152 excluded; 6AM5↔6AMU floor RMSD ≈ 0.21 Å). Peptide
Cα-RMSD is then measured to **three** references — a single reference cannot resolve register:
`GIG` (6AM5, P10 F-pocket), `DRG_shift` (6AMU, P9 F-pocket), `DRG_unshift` (6AMT, excluded from seeding).

Arms: **fpocket** (Q30+floor F-pocket, prior), **tcr** (Q30α+Phe-β), **apoc** (+A/B-pocket N-anchor),
**nslide** (N-terminal window scan, motif-scaffold).""")

co("""import sys, importlib, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/ubuntu/if-mhc/py")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import q30_analysis as Q; importlib.reload(Q)
plt.rcParams.update({"figure.dpi": 110, "font.size": 9})
ARMS = ["fpocket", "tcr", "apoc", "nslide"]
ACOL = {"fpocket":"#9e9e9e","tcr":"#1f77b4","apoc":"#d62728","nslide":"#2ca02c"}
df, CO = Q.load_all(ARMS)
print(f"{len(df)} designs | coords {CO.shape} | refs: GIG,DRG_shift,DRG_unshift")
df.groupby(["arm","pid"]).size()""")

md("""## 1 · Headline — GIG vs DRG basin scatter (floor frame)
The correct headline object. Points on the `y=x` diagonal = drift (no basin selected); points pulled
into a corner = a basin was chosen. Green box = seated (`min(toGIG,toDRG) < 4 Å`).""")
co("""fig, axes = plt.subplots(1, len(ARMS), figsize=(4*len(ARMS), 4), sharex=True, sharey=True)
for ax, arm in zip(np.atleast_1d(axes), ARMS):
    s = df[df.arm == arm]
    if len(s):
        for pid, mk in [("6AM5","o"),("6AMU","^")]:
            ss = s[s.pid == pid]
            ax.scatter(ss.to_GIG, ss.to_DRG_shift, s=28, marker=mk, alpha=.7,
                       c=ACOL[arm], edgecolor="k", linewidth=.3, label=pid)
    lim = 1 + np.nanmax([df.to_GIG.max(), df.to_DRG_shift.max()]) if len(df) else 12
    ax.plot([0,lim],[0,lim],"k--",lw=.7,alpha=.5); ax.axhline(4,c="g",lw=.5,ls=":"); ax.axvline(4,c="g",lw=.5,ls=":")
    ax.set_title(f"{arm}  (n={len(s)})"); ax.set_xlabel("Cα-RMSD to GIG (Å)")
np.atleast_1d(axes)[0].set_ylabel("Cα-RMSD to DRG_shift (Å)"); np.atleast_1d(axes)[0].legend(fontsize=7)
plt.tight_layout(); plt.show()""")
md("**Quantified:** fraction seated, fraction off-diagonal, and basin split per arm × epitope.")
co("""def quant(s):
    return pd.Series(dict(n=len(s), seated=round(s.seated.mean(),2) if len(s) else np.nan,
        off_diag_med=round(s.off_diagonal.median(),2) if len(s) else np.nan,
        pct_GIG=round((s.basin=="GIG").mean(),2) if len(s) else np.nan,
        toGIG_med=round(s.to_GIG.median(),2) if len(s) else np.nan,
        toDRG_med=round(s.to_DRG_shift.median(),2) if len(s) else np.nan))
df.groupby(["arm","pid"]).apply(quant)""")

md("""## 2 · Three-reference RMSD
A single backbone can't resolve register; comparing to GIG / DRG_shift / DRG_unshift shows *which*
register (and whether the unshifted DRG, though never seeded, is nearest).""")
co("""cols = [c for c in ["to_GIG","to_DRG_shift","to_DRG_unshift"] if c in df]
g = df.groupby(["arm","pid"])[cols].median().round(2)
g["nearest_ref"] = df.groupby(["arm","pid"]).apply(
    lambda s: pd.Series(np.median(s[cols].values,0), index=cols).idxmin())
g""")

md("""## 3 · N- vs C-terminal split RMSD (highest-signal addition)
The register is a **C-terminal** event; the N-terminus is what we conditioned. If conditioning works,
the N-half should track the target while the C-half is where the basin is (or isn't) resolved.""")
co("""fig, axes = plt.subplots(1, len(ARMS), figsize=(4*len(ARMS), 3.6), sharex=True, sharey=True)
for ax, arm in zip(np.atleast_1d(axes), ARMS):
    s = df[df.arm == arm]
    if len(s):
        ax.scatter(s.Nterm_GIG, s.Cterm_GIG, s=22, c=ACOL[arm], alpha=.6, edgecolor="k", linewidth=.3, label="vs GIG")
        ax.scatter(s.Nterm_DRG_shift, s.Cterm_DRG_shift, s=22, marker="x", c="k", alpha=.5, label="vs DRG")
    ax.set_title(arm); ax.set_xlabel("N-term (P1-5) RMSD Å")
np.atleast_1d(axes)[0].set_ylabel("C-term (P6-10) RMSD Å"); np.atleast_1d(axes)[0].legend(fontsize=7)
plt.tight_layout(); plt.show()
df.groupby("arm")[[c for c in ["Nterm_GIG","Cterm_GIG","Nterm_DRG_shift","Cterm_DRG_shift"] if c in df]].median().round(2)""")

md("""## 4 · Contact satisfaction — did conditioning actually work? (the direct test)
Per design, min **Cβ–Cβ** distance from the peptide to each conditioned residue (backbone-only
designs → Cβ, RFdiffusion's hotspot convention; ≤8 Å = contact formed). This closes the loop: did the
rebuilt backbone form the contacts we conditioned on?""")
co("""fig, axes = plt.subplots(1, len(ARMS), figsize=(4*len(ARMS), 3.4))
for ax, arm in zip(np.atleast_1d(axes), ARMS):
    s = df[df.arm == arm]
    if not len(s):
        ax.set_title(f"{arm} (n=0)"); continue
    byres = {}
    for d in s.contacts:
        for k, v in (d or {}).items(): byres.setdefault(k, []).append(v)
    for k, vals in byres.items():
        ax.hist(vals, bins=np.arange(0,20,1.5), histtype="step", lw=1.6, label=k)
    ax.axvline(8, c="k", ls="--", lw=.8); ax.set_title(f"{arm}"); ax.set_xlabel("min Cβ-Cβ to hotspot (Å)")
    ax.legend(fontsize=6)
np.atleast_1d(axes)[0].set_ylabel("designs"); plt.tight_layout(); plt.show()
df.groupby("arm").contacts_frac.mean().round(2).rename("frac hotspots ≤8Å")""")

md("""## 5 · F-pocket register readout (RMSD-independent)
Which peptide Cα sits at the F-pocket centroid: **P10 = unshifted / GIG-like**, **P9 = DRG-shifted**.
The single cleanest register number.""")
co("""fp = df.groupby(["arm","pid"]).fpocket_pos.value_counts().unstack(fill_value=0)
display(fp)
fig, ax = plt.subplots(figsize=(6,3))
for arm in ARMS:
    s = df[df.arm==arm]
    if len(s): ax.hist(s.fpocket_pos, bins=np.arange(6.5,11.5,1), histtype="step", lw=1.8, label=arm, color=ACOL[arm])
ax.set_xlabel("peptide position in F-pocket centroid"); ax.set_ylabel("designs"); ax.legend(); plt.show()""")

md("""## 6 · Clustering + aggregate trace overlays
Cluster on pairwise peptide Cα-RMSD (already in the floor frame → direct). Report cluster count,
population, and each cluster's mean RMSD to GIG/DRG; overlay each cluster's Cα traces with the three
references (the quantified version of the trace panels).""")
co("""from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
N = len(CO)
if N >= 3:
    D = np.zeros((N,N))
    for i in range(N):
        D[i,i+1:] = np.sqrt(((CO[i]-CO[i+1:])**2).sum(-1).mean(-1))
    D += D.T
    Z = linkage(squareform(D, checks=False), method="average")
    cl = fcluster(Z, t=1.5, criterion="distance")   # 1.5 Å average-linkage cut
    df["cluster"] = cl
    summ = df.groupby("cluster").agg(n=("seated","size"), toGIG=("to_GIG","mean"),
            toDRG=("to_DRG_shift","mean"), arms=("arm", lambda x:",".join(sorted(set(x))))).round(2)
    summ = summ.sort_values("n", ascending=False)
    display(summ.head(12))
    print(f"{len(summ)} clusters at 1.5Å")
else:
    df["cluster"] = 1; summ = None; print("too few designs to cluster yet")""")
co("""if len(CO) >= 3 and summ is not None:
    top = summ.head(min(4, len(summ))).index
    fig = plt.figure(figsize=(4*len(top), 3.6))
    for k, cid in enumerate(top):
        ax = fig.add_subplot(1, len(top), k+1, projection="3d")
        for i in df.index[df.cluster == cid][:30]:
            p = CO[i]; ax.plot(p[:,0],p[:,1],p[:,2], color="0.6", alpha=.4, lw=.8)
        for nm, c in [("GIG","#1f77b4"),("DRG_shift","#d62728"),("DRG_unshift","#2ca02c")]:
            r = Q.REFS[nm]
            if len(r)==10: ax.plot(r[:,0],r[:,1],r[:,2], color=c, lw=2.2, label=nm)
        ax.set_title(f"cluster {cid} (n={summ.loc[cid,'n']})"); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        if k==0: ax.legend(fontsize=6)
    plt.tight_layout(); plt.show()""")

md("""## 7 · Per-residue RMSF (tight bundle vs spray)
Positional spread of each peptide Cα across the ensemble in the floor frame. Expect **low at the
conditioned N-terminus / anchors, high at the free C-terminus** — the number behind a tight bundle
vs a spray.""")
co("""fig, ax = plt.subplots(figsize=(6.5,3.4))
for arm in ARMS:
    idx = df.index[df.arm == arm]
    if len(idx) >= 2:
        sub = CO[idx]; rmsf = np.sqrt(((sub - sub.mean(0))**2).sum(-1).mean(0))
        ax.plot(range(1,11), rmsf, "-o", ms=3, color=ACOL[arm], label=f"{arm} (n={len(idx)})")
ax.set_xlabel("peptide position"); ax.set_ylabel("Cα RMSF (Å)"); ax.set_xticks(range(1,11)); ax.legend(fontsize=7)
ax.axvspan(0.5,3.5,color="0.9",zorder=0); ax.text(2,ax.get_ylim()[1]*.9,"N (conditioned)",fontsize=7,ha="center")
plt.tight_layout(); plt.show()""")

md("""## 8 · Physical sanity filter
Cα–Cα virtual bond lengths (ideal ~3.8 Å; flag <3.0 or >4.5) and self-clashes (non-adjacent
Cα < 3.0 Å). Broken designs (the tails flying off) are flagged and should be excluded from the
basin claims above.""")
co("""san = df.groupby("arm").agg(n=("broken","size"), bad_bond=("bad_bond","sum"),
        clash=("clash","sum"), broken=("broken","sum"))
san["pct_broken"] = (100*san.broken/san.n).round(1)
display(san)
if df.broken.sum(): display(df[df.broken==1][["arm","seed","cond","idx","min_ref","file"]].head(20))
else: print("no broken geometry flagged")""")

md("""## 9 · Verdict
Per arm, on **clean** designs only: did conditioning pull backbones off the diagonal into a basin, and
were the conditioned contacts formed?""")
co("""ok = df[df.broken == 0]
verdict = ok.groupby("arm").agg(n=("seated","size"), pct_seated=("seated","mean"),
        off_diag_med=("off_diagonal","median"), contacts=("contacts_frac","mean"),
        fpocket_mode=("fpocket_pos", lambda x: x.mode().iloc[0] if len(x) else np.nan)).round(2)
display(verdict)
print("Reading: high off_diag_med + high pct_seated + high contacts = conditioning steered a basin.")
print("NOTE: while arms are still running these are dominated by near-native low-T/crystal echoes;")
print("re-run this notebook as relaxed-seed and higher-noise designs accumulate.")""")

nb["cells"] = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
with open("/home/ubuntu/if-mhc/09_q30_conditioning_arms.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote /home/ubuntu/if-mhc/09_q30_conditioning_arms.ipynb")
