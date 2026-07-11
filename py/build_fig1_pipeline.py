#!/usr/bin/env python3
"""Figure 1: pipeline schematic. Plain boxes + arrows, white background, no axes/grid/title/bold."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(11.5, 3.2), dpi=200)
ax.set_xlim(0, 11.5); ax.set_ylim(0, 2.2)
ax.axis("off")
fig.patch.set_facecolor("white")

steps = [
    "RFdiffusion\ncontact hotspots only,\nno peptide template",
    "generated\nCα backbone",
    "register scoring:\nproximity + anchor + depth\n(MD-calibrated)",
    "ProteinMPNN\nsequence design\n(out of scope here)",
]
n = len(steps)
box_w, box_h = 2.4, 1.3
gap = (11.5 - n * box_w) / (n + 1)
xs = [gap + i * (box_w + gap) for i in range(n)]
y0 = (2.2 - box_h) / 2

for i, (x, label) in enumerate(zip(xs, steps)):
    fc = "white" if i < n - 1 else "#f2f2f2"
    box = FancyBboxPatch((x, y0), box_w, box_h,
                          boxstyle="round,pad=0.04,rounding_size=0.08",
                          linewidth=1.2, edgecolor="black", facecolor=fc)
    ax.add_patch(box)
    ax.text(x + box_w / 2, y0 + box_h / 2, label, ha="center", va="center",
            fontsize=9.5, color="black", linespacing=1.4)
    if i < n - 1:
        x_next = xs[i + 1]
        arrow = FancyArrowPatch((x + box_w + 0.03, y0 + box_h / 2),
                                 (x_next - 0.03, y0 + box_h / 2),
                                 arrowstyle="-|>", mutation_scale=14,
                                 linewidth=1.2, color="black")
        ax.add_patch(arrow)

plt.tight_layout()
out = "/home/ubuntu/if-mhc/figures/fig1_pipeline/fig1_pipeline.png"
plt.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
print("wrote", out)
