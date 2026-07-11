#!/usr/bin/env python3
"""Scatter of every NATIVE-derived structure (crystal, relaxation, MD) on the register map:
toGIG vs toDRG, colored by method/params, shaped by starting structure. Annotates the most extreme
excursions (closest any native-derived structure comes to the OTHER register).
"""
import sys; sys.path.insert(0, "/home/ubuntu/if-mhc/py")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import core_load as CL
plt.rcParams.update({"figure.dpi": 130, "font.size": 9})

DF, CO, TRAJ = CL.load_all()
SEP = CL._rmsd(CL.GIG, CL.DRG)

# --- assemble native-derived points: (toGIG, toDRG, method, start) ---
pts = []
lab = {"crystal": "crystal (native)", "relaxed": "OpenMM relax-snapshot",
       "openmm-relax": "OpenMM relax (Tamarind)"}
nat = DF[DF.group.isin(["native", "tool"])]
for _, r in nat.iterrows():
    pts.append(dict(toGIG=r.toGIG, toDRG=r.toDRG, method=lab.get(r.cond, r.cond), start=r.pid))
md = TRAJ[TRAJ.job.str.match(r"ifmhc_6AM[5U]_md_\d+K")].copy()
md["start"] = md.job.str.extract(r"(6AM[5U])"); md["T"] = md.job.str.extract(r"(\d+K)")
for _, r in md.iterrows():
    pts.append(dict(toGIG=r.to_GIG, toDRG=r.to_DRG, method=f"MD {r['T']}", start=r.start))
P = pd.DataFrame(pts)

# --- extreme excursions: closest any native-derived structure gets to the OTHER register ---
P["own"]   = np.where(P.start == "6AM5", P.toGIG, P.toDRG)
P["other"] = np.where(P.start == "6AM5", P.toDRG, P.toGIG)
print(f"GIG↔DRG separation = {SEP:.2f} Å | {len(P)} native-derived structures\n")
print("MOST EXTREME toward the OTHER register (per starting structure):")
for st in ["6AM5", "6AMU"]:
    s = P[P.start == st]
    e = s.loc[s.other.idxmin()]
    print(f"  {st}: closest to other = {e.other:.2f} Å  (method='{e.method}', toGIG={e.toGIG:.2f}, toDRG={e.toDRG:.2f})")
print("\nMOST EXTREME away from own native basin (largest own-RMSD):")
for st in ["6AM5", "6AMU"]:
    s = P[P.start == st]; e = s.loc[s.own.idxmax()]
    print(f"  {st}: max own-RMSD = {e.own:.2f} Å  (method='{e.method}')")

# --- plot ---
MCOL = {"crystal (native)":"#111111", "OpenMM relax-snapshot":"#ff7f0e",
        "OpenMM relax (Tamarind)":"#8c564b", "MD 300K":"#1f77b4", "MD 370K":"#e41a1c"}
MK = {"6AM5": "o", "6AMU": "^"}   # starting structure
fig, ax = plt.subplots(figsize=(7.6, 7))
for method, g in P.groupby("method"):
    for st, gg in g.groupby("start"):
        ax.scatter(gg.toGIG, gg.toDRG, s=(70 if "crystal" in method else 16),
                   c=MCOL.get(method, "0.5"), marker=MK[st], alpha=.75,
                   edgecolor="k", linewidth=.3, label=f"{method} · {st}")
VIEW = 5.5   # zoom to the meaningful region; a few failed-relaxation outliers sit far beyond
ax.plot([0, VIEW], [0, VIEW], "k--", lw=.7, alpha=.5)
ax.set_xlim(-0.3, VIEW); ax.set_ylim(-0.3, VIEW)
noff = int(((P.toGIG > VIEW) | (P.toDRG > VIEW)).sum())
ax.text(0.02, 0.02, f"{noff} OpenMM-relax outliers >{VIEW}Å off-plot\n(up to ~16Å — likely failed relaxations)",
        transform=ax.transAxes, fontsize=6.5, color="#8c564b", va="bottom")
ax.scatter([0.4],[SEP], marker="*", s=240, c="#1f77b4", edgecolor="k", zorder=6)   # GIG native anchor
ax.scatter([SEP],[0.4], marker="*", s=240, c="#e41a1c", edgecolor="k", zorder=6)   # DRG native anchor
ax.text(0.5, SEP, " GIG", fontsize=8, va="center"); ax.text(SEP, 0.5, " DRG", fontsize=8)
# annotate the two extreme-toward-other points
for st in ["6AM5", "6AMU"]:
    s = P[P.start == st]; e = s.loc[s.other.idxmin()]
    ax.annotate(f"extreme {st}\n{e.method}", (e.toGIG, e.toDRG), fontsize=6.5,
                xytext=(e.toGIG+0.15, e.toDRG+0.15), arrowprops=dict(arrowstyle="->", lw=.6))
ax.set_xlabel("Cα-RMSD to GIG (Å)"); ax.set_ylabel("Cα-RMSD to DRG (Å)")
ax.set_title("Native-derived structures (crystal / relaxation / MD)\ncolor = method, shape = starting structure")
ax.legend(fontsize=6, loc="upper right", ncol=1)
plt.tight_layout(); plt.savefig("/home/ubuntu/if-mhc/native_excursions.png", dpi=150)
print("\nsaved native_excursions.png")
