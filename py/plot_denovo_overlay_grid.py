#!/usr/bin/env python3
"""Campaign x conditioning grid of 3D peptide-Ca overlays, on the common groove frame used by
score_denovo_designs.py. Rows = campaign (source dir), columns = conditioning (cond token). Each
cell overlays every design's frame-aligned peptide trace (blue=targets GIG/6AM5, orange=targets
DRG/6AMU; thin, translucent) on top of the two native crystal peptides (bold) and the rigid MHC
groove backbone (gray, REF_CA) for spatial context. Reuses _map_peptide/parse_meta/gather from
score_denovo_designs.py so the frame and parsing stay identical to the scoring table.

Usage: python py/plot_denovo_overlay_grid.py [outputs/dir ...]
Output: outputs/denovo_scores/overlay_grid.png
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import score_denovo_designs as S

OUT = f"{S.ROOT}/outputs/denovo_scores"
os.makedirs(OUT, exist_ok=True)

ROW_ORDER = ["grind", "ladder", "promising", "maxcond", "denovo30"]
COL_ORDER = ["mhc", "mhc_tcr1", "mhc_tcr2", "tcr1", "tcr2",
             "L1_nterm", "L2_nterm_t1", "L3_nterm_t2", "L4_expanded", "L5_max",
             "k18", "k24", "max"]
PID_COLOR = {"6AM5": "#3b6fd4", "6AMU": "#e07b1f"}   # design lines: GIG-target blue, DRG-target orange
PID_NATIVE = {"6AM5": ("GIG", "#0b2e6b"), "6AMU": ("DRG", "#8a3d00")}

def gather_records(dirs):
    files = S.gather(dirs)
    recs = []
    for f in files:
        meta = S.parse_meta(f)
        r = S._map_peptide(f)
        if r is None: continue
        pa, tcr, pep_len = r
        if pa is None: continue
        recs.append(dict(**meta, pa=pa, tcr=tcr))
    return recs

def main():
    dirs = sys.argv[1:] or [f"{S.ROOT}/outputs/{x}/pdb" for x in ("grind", "ladder", "promising")] + \
                            [f"{S.ROOT}/outputs/rfd_denovo30/pdb", f"{S.ROOT}/outputs/rfd_maxcond/pdb"]
    dirs = [d for d in dirs if os.path.isdir(d)]
    print(f"scanning {len(dirs)} dir(s)...")
    recs = gather_records(dirs)
    print(f"mapped {len(recs)} designs with a locatable 10-mer peptide")

    rows = [r for r in ROW_ORDER if any(rec["source"] == r for rec in recs)]
    present_cols = {rec["cond"] for rec in recs}
    cols = [c for c in COL_ORDER if c in present_cols] + sorted(present_cols - set(COL_ORDER))

    # shared axis limits across every panel, so panel-to-panel spread is visually comparable
    allpts = np.vstack([S.REF_CA] + [rec["pa"] for rec in recs] + [S.GIG, S.DRG])
    ctr = allpts.mean(0)
    rad = np.percentile(np.linalg.norm(allpts - ctr, axis=1), 98) * 1.05

    grouped = {}
    for rec in recs:
        grouped.setdefault((rec["source"], rec["cond"]), []).append(rec)

    nrow, ncol = len(rows), len(cols)
    fig = plt.figure(figsize=(3.1 * ncol, 3.1 * nrow))
    elev, azim = 18, -60

    for i, src in enumerate(rows):
        for j, cond in enumerate(cols):
            grp = grouped.get((src, cond), [])
            n = len(grp)
            ax = fig.add_subplot(nrow, ncol, i * ncol + j + 1, projection="3d")
            if n == 0:
                # never attempted: mute the panel instead of drawing an empty groove
                ax.set_axis_off()
                ax.text2D(0.5, 0.5, "not run", transform=ax.transAxes, ha="center", va="center",
                          fontsize=8, color="0.75", style="italic")
                if i == 0:
                    ax.set_title(cond, fontsize=9, pad=2, color="0.6")
                if j == 0:
                    ax.text2D(-0.08, 0.5, src, transform=ax.transAxes, ha="right", va="center",
                              fontsize=10, fontweight="bold", rotation=90, color="0.6")
                continue
            ax.set_xlim(ctr[0] - rad, ctr[0] + rad); ax.set_ylim(ctr[1] - rad, ctr[1] + rad)
            ax.set_zlim(ctr[2] - rad, ctr[2] + rad)
            ax.view_init(elev=elev, azim=azim)
            ax.set_box_aspect((1, 1, 1))
            # backdrop: rigid MHC groove backbone, same in every panel
            ax.plot(*S.REF_CA.T, color="0.75", lw=0.8, alpha=0.6, zorder=1)
            # native crystal peptides, bold
            for pid, (label, color) in PID_NATIVE.items():
                nat = S.GIG if pid == "6AM5" else S.DRG
                ax.plot(*nat.T, color=color, lw=2.6, alpha=0.95, zorder=5)
            for rec in grp:
                color = PID_COLOR[rec["pid"]]
                ls = "-" if rec["context"] == "withTCR" else "--"
                ax.plot(*rec["pa"].T, color=color, lw=0.7, alpha=0.35, ls=ls, zorder=3)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            ax.set_axis_off()
            ax.text2D(0.5, -0.04, f"n={n}", transform=ax.transAxes, ha="center", va="top", fontsize=7, color="0.4")
            if i == 0:
                ax.set_title(cond, fontsize=9, pad=2)
            if j == 0:
                ax.text2D(-0.08, 0.5, src, transform=ax.transAxes, ha="right", va="center",
                          fontsize=10, fontweight="bold", rotation=90)

    # legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=PID_NATIVE["6AM5"][1], lw=2.6, label="native GIG (6AM5)"),
        Line2D([0], [0], color=PID_NATIVE["6AMU"][1], lw=2.6, label="native DRG (6AMU)"),
        Line2D([0], [0], color=PID_COLOR["6AM5"], lw=1.5, label="design, targets 6AM5/GIG"),
        Line2D([0], [0], color=PID_COLOR["6AMU"], lw=1.5, label="design, targets 6AMU/DRG"),
        Line2D([0], [0], color="0.4", lw=1.5, ls="-", label="withTCR"),
        Line2D([0], [0], color="0.4", lw=1.5, ls="--", label="noTCR"),
        Line2D([0], [0], color="0.75", lw=1.5, label="MHC groove backbone"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("De-novo peptide designs: per-campaign x per-conditioning overlay on native groove frame",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=(0.02, 0.03, 1, 0.98))
    out_path = f"{OUT}/overlay_grid.png"
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}  ({nrow} rows x {ncol} cols, {len(recs)} designs plotted)")

if __name__ == "__main__":
    main()
