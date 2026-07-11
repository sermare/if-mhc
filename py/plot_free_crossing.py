#!/usr/bin/env python3
"""4-panel Ca-trace comparison for one de-novo design vs both native registers.

Usage:
    python py/plot_free_crossing.py <path/to/design.pdb> [out.png]

Panels: (1) both natives (GIG vs DRG), (2) cognate native + design,
(3) other native + design, (4) design alone (N->C gradient, F-pocket marked).
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import score_denovo_designs as S

def trace(ax, ca, color, label, ls="-", lw=2.0, alpha=1.0, gradient=False):
    if gradient:
        cmap = plt.get_cmap("viridis")
        for i in range(len(ca) - 1):
            ax.plot(*zip(ca[i], ca[i + 1]), color=cmap(i / (len(ca) - 2)), lw=lw)
        ax.scatter(*ca[0], color=cmap(0.0), s=70, marker="o", label=f"{label} N-term", zorder=5)
        ax.scatter(*ca[-1], color=cmap(1.0), s=70, marker="^", label=f"{label} C-term", zorder=5)
    else:
        ax.plot(ca[:, 0], ca[:, 1], ca[:, 2], color=color, ls=ls, lw=lw, alpha=alpha, label=label)
        ax.scatter(*ca[0], color=color, s=60, marker="o", alpha=alpha, zorder=5)
        ax.scatter(*ca[-1], color=color, s=60, marker="^", alpha=alpha, zorder=5)

def fpocket(ax):
    ax.scatter(*S.FPOCKET_CENTROID, color="black", s=90, marker="x", label="F-pocket", zorder=6)

def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        "outputs", "denovo_scores", os.path.basename(path).replace(".pdb", "_crossing.png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)

    pa, tcr, L = S._map_peptide(path)
    ca = np.asarray(pa)
    GIG, DRG = S.GIG, S.DRG
    to_gig = float(np.sqrt(((ca - GIG) ** 2).sum(1).mean()))
    to_drg = float(np.sqrt(((ca - DRG) ** 2).sum(1).mean()))
    meta = S.parse_meta(path)
    cognate, other = ("GIG", "DRG") if meta["pid"] == "6AM5" else ("DRG", "GIG")
    cog_ca, oth_ca = (GIG, DRG) if cognate == "GIG" else (DRG, GIG)
    to_cog, to_oth = (to_gig, to_drg) if cognate == "GIG" else (to_drg, to_gig)

    fig = plt.figure(figsize=(16, 4.2))
    fig.suptitle(f"{os.path.basename(path)}   to_cognate({cognate})={to_cog:.2f}A   "
                 f"to_other({other})={to_oth:.2f}A", fontsize=11)

    ax1 = fig.add_subplot(141, projection="3d")
    trace(ax1, GIG, "teal", "GIG (native)")
    trace(ax1, DRG, "darkorange", "DRG (native)", ls="--")
    fpocket(ax1); ax1.set_title("Native registers"); ax1.legend(fontsize=7)

    ax2 = fig.add_subplot(142, projection="3d")
    trace(ax2, cog_ca, "teal" if cognate == "GIG" else "darkorange", f"{cognate} (cognate)")
    trace(ax2, ca, "purple", "design", lw=2.5)
    fpocket(ax2); ax2.set_title(f"Cognate {cognate} + design ({to_cog:.2f}A)"); ax2.legend(fontsize=7)

    ax3 = fig.add_subplot(143, projection="3d")
    trace(ax3, oth_ca, "darkorange" if other == "DRG" else "teal", f"{other} (other)", ls="--")
    trace(ax3, ca, "purple", "design", lw=2.5)
    fpocket(ax3); ax3.set_title(f"Other {other} + design ({to_oth:.2f}A)"); ax3.legend(fontsize=7)

    ax4 = fig.add_subplot(144, projection="3d")
    trace(ax4, ca, None, "design", gradient=True)
    fpocket(ax4); ax4.set_title("Design alone (N→C)"); ax4.legend(fontsize=7)

    for ax in (ax1, ax2, ax3, ax4):
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"{out}\tto_cognate={to_cog:.3f}({cognate})\tto_other={to_oth:.3f}({other})\tfpocket_pos={S.score_occ(path).get('fpocket_pos') if hasattr(S,'score_occ') else ''}")

if __name__ == "__main__":
    main()
