#!/usr/bin/env python3
"""Supplemental figure: distribution of Ca-RMSD to the OTHER (crossing-target) register, by design
class, per crystal, against the 310K native acceptance band (block-bootstrap 95% CI). Marks the
de-novo crossing designs (within the DRG band 95% CI) and the de-novo recoveries.

Data: distances.csv (one row/design: cry, group, cog=to-own, oth=to-other, seats_other, file).
Palette: Okabe-Ito (CVD-safe, publication standard)."""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CSV = "/tmp/claude-46355/nullscore/distances.csv"
OUT = os.path.join(os.path.dirname(__file__), "figS_distance_distributions.png")
df = pd.read_csv(CSV)

# --- Okabe-Ito, fixed order: de-novo / random / fixed / xover ---
OK = {"de-novo":"#0072B2", "random":"#999999", "fixed":"#E69F00", "xover":"#009E73"}
LABEL = {"de-novo":"de-novo (contact only)", "random":"random (no hotspots/template)",
         "fixed":"fixed (templated)", "xover":"crossover (N-term seeded)"}

# 310K bands + block-bootstrap 95% CI (from bootstrap_band.py). Design crosses to the OTHER register,
# so it is judged against the OTHER crystal's band: 6AM5 design -> DRG band (6AMU); 6AMU design -> GIG band.
OTHER_BAND = {"6AM5":dict(name="DRG", point=1.48, lo=1.36, hi=1.58),   # 6AMU/310K, ESS 13 (well converged)
              "6AMU":dict(name="GIG", point=1.65, lo=1.30, hi=1.68)}   # 6AM5/310K, ESS 2 (under-converged)

fig, axes = plt.subplots(2, 1, figsize=(8.2, 7.4), sharex=True)
XMAX = 22
bins = np.linspace(0, XMAX, 90)

for ax, cry in zip(axes, ("6AM5", "6AMU")):
    b = OTHER_BAND[cry]
    sub = df[df.cry == cry]
    # band + 95% CI shading
    ax.axvspan(0, b["point"], color="#56B4E9", alpha=0.16, lw=0, zorder=0)
    ax.axvspan(b["point"], b["hi"], color="#56B4E9", alpha=0.30, lw=0, zorder=0,
               label="_")  # CI extension
    ax.axvline(b["point"], color="#0072B2", lw=1.4, ls="-", zorder=2)
    ax.axvline(b["hi"], color="#0072B2", lw=1.0, ls=":", zorder=2)
    # distributions (density-normalized step histograms so groups with different n are comparable)
    for g in ("de-novo", "random", "fixed"):
        v = sub[sub.group == g].oth.values
        if len(v) < 3: continue
        ax.hist(v, bins=bins, density=True, histtype="stepfilled",
                color=OK[g], alpha=0.32, lw=0, zorder=1)
        ax.hist(v, bins=bins, density=True, histtype="step",
                color=OK[g], lw=1.8, zorder=3)
    # --- mark the individual crossing designs (within the DRG-band 95% CI) and recoveries ---
    # crossings: de-novo designs that seat the OTHER anchor and land within the band 95% CI
    cross = sub[(sub.group == "de-novo") & (sub.seats_other == 1) & (sub.oth <= b["hi"])]
    for _, r in cross.iterrows():
        ax.axvline(r.oth, color="#D55E00", lw=2.2, ymax=0.62, zorder=5)
        ax.text(r.oth, ax.get_ylim()[1]*0.64 if False else 0, "", )
    # recoveries: de-novo designs recovering their OWN register (marked at their to-OTHER value)
    rec = sub[(sub.group == "de-novo") & (sub.seats_own == 1) & (sub.cog <= OTHER_BAND["6AMU" if cry=="6AM5" else "6AM5"]["hi"])]
    # own band is this crystal's own band; recompute simply: own band point ~ (GIG 1.65 for 6AM5, DRG 1.48 for 6AMU)
    OWN = {"6AM5":1.68, "6AMU":1.58}
    rec = sub[(sub.group == "de-novo") & (sub.seats_own == 1) & (sub.cog <= OWN[cry])]
    for _, r in rec.iterrows():
        ax.axvline(r.oth, color="#CC79A7", lw=1.6, ls=(0,(4,2)), ymax=0.5, zorder=4)

    ax.set_ylabel("density")
    ax.set_xlim(0, XMAX)
    conv = "well-converged, ESS 13" if cry=="6AM5" else "under-converged, ESS 2"
    ax.set_title(f"{cry} designs  →  distance to the other register ({b['name']})    "
                 f"[band {b['point']:.2f} Å, 95% CI {b['lo']:.2f}–{b['hi']:.2f}; {conv}]",
                 fontsize=10, loc="left")

axes[1].set_xlabel("Cα-RMSD to the OTHER (crossing-target) register (Å)")

legend_elems = [
    Line2D([0],[0], color=OK["de-novo"], lw=3, label=LABEL["de-novo"]),
    Line2D([0],[0], color=OK["random"], lw=3, label=LABEL["random"]),
    Line2D([0],[0], color=OK["fixed"], lw=3, label=LABEL["fixed"]),
    Line2D([0],[0], color="#56B4E9", lw=8, alpha=0.4, label="310K acceptance band (+95% CI)"),
    Line2D([0],[0], color="#D55E00", lw=2.4, label="crossing design (within CI)"),
    Line2D([0],[0], color="#CC79A7", lw=1.8, ls=(0,(4,2)), label="recovery design (own register)"),
]
axes[0].legend(handles=legend_elems, loc="upper right", fontsize=8.5, framealpha=0.95)
fig.suptitle("Distance to the alternate register across design classes (310 K envelope)",
             fontsize=12, y=0.99)
fig.tight_layout(rect=[0,0,1,0.97])
fig.savefig(OUT, dpi=170)
print("wrote", OUT)

# report what got marked
for cry in ("6AM5","6AMU"):
    b=OTHER_BAND[cry]; sub=df[df.cry==cry]
    cross=sub[(sub.group=="de-novo")&(sub.seats_other==1)&(sub.oth<=b["hi"])]
    print(f"{cry}: {len(cross)} de-novo crossing lines (within CI) at oth = {sorted(round(x,2) for x in cross.oth)}")
