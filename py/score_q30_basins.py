#!/usr/bin/env python3
"""Basin scoring for the Q30 dual-hotspot partial-diffusion campaign (primary deliverable).

For every diffused backbone in outputs/rfdiff_q30/*_T*_*.pdb, put its peptide into the common
6AMU groove frame and measure register-preserving Cα-RMSD to the GIG and DRG basins -- reusing
core_load.{to_common, GIG, DRG, _rmsd, _detect}. Success = coming OFF the toGIG≈toDRG diagonal
INTO a defined basin (min(toGIG,toDRG) small), unlike the prior sole-conditioning runs.

Outputs:
  outputs/rfdiff_q30/basins.csv        one row per diffused backbone (seed,pid,partial_T,idx,
                                       toGIG,toDRG,min_native,basin,seated)
  outputs/rfdiff_q30/basin_map.png     toGIG-vs-toDRG scatter of diffused pts over the prior map
  stdout                               occupancy table (% seated, basin split) per partial_T
"""
import os, re, sys, glob
import numpy as np, pandas as pd
sys.path.insert(0, "/home/ubuntu/if-mhc/py")
from core_load import to_common, GIG, DRG, _rmsd, _detect, CACHE

ROOT = "/home/ubuntu/if-mhc"
OUT  = f"{ROOT}/outputs/rfdiff_q30"
FNAME = re.compile(r"^(?P<seed>.+)_T(?P<T>\d+)_(?P<idx>\d+)_split\.pdb$")


def score_one(path):
    mhc, pep = _detect(path)
    if mhc is None or pep is None or len(pep) != 10:
        return None
    pc, fit = to_common(mhc, pep)          # peptide CA in common groove frame
    if pc is None:
        return None
    return round(_rmsd(pc, GIG), 3), round(_rmsd(pc, DRG), 3), round(float(fit), 3)


def main():
    files = sorted(glob.glob(f"{OUT}/*_T*_*_split.pdb"))
    rows, skipped = [], 0
    for f in files:
        m = FNAME.match(os.path.basename(f))
        if not m:
            skipped += 1; continue
        sc = score_one(f)
        if sc is None:
            skipped += 1; continue
        toGIG, toDRG, fit = sc
        seed = m["seed"]
        pid = "6AM5" if seed.startswith("6AM5") else "6AMU"
        rows.append(dict(seed=seed, pid=pid, partial_T=int(m["T"]), idx=int(m["idx"]),
                         toGIG=toGIG, toDRG=toDRG, fit=fit,
                         min_native=round(min(toGIG, toDRG), 3),
                         basin=("GIG" if toGIG < toDRG else "DRG"),
                         seated=int(min(toGIG, toDRG) < 4.0), file=f))
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(f"{OUT}/basins.csv", index=False)
    print(f"scored {len(df)} diffused backbones ({skipped} skipped) -> {OUT}/basins.csv")
    if df.empty:
        return

    # occupancy per partial_T x seed-epitope
    print("\n=== basin occupancy (mean toGIG/toDRG, % seated, basin split) ===")
    g = (df.groupby(["pid", "partial_T"])
           .agg(n=("seated", "size"), seated_pct=("seated", "mean"),
                toGIG_med=("toGIG", "median"), toDRG_med=("toDRG", "median"),
                to_own_med=("min_native", "median")).reset_index())
    g["seated_pct"] = (g["seated_pct"] * 100).round(0)
    print(g.round(2).to_string(index=False))
    print("\nbasin landing (of seated only):")
    print(df[df.seated == 1].groupby(["pid"]).basin.value_counts().to_string())

    # scatter over the existing basin map (the prior diagonal)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        base = pd.read_parquet(f"{CACHE}/df.parquet") if os.path.exists(f"{CACHE}/df.parquet") else None
        fig, ax = plt.subplots(figsize=(6.2, 6))
        if base is not None:
            ax.scatter(base.toGIG, base.toDRG, s=8, c="0.8", label=f"prior map (n={len(base)})", zorder=1)
        cmap = {5: "#4575b4", 10: "#91bfdb", 20: "#fc8d59", 40: "#d73027"}
        for T, sub in df.groupby("partial_T"):
            for pid, mk in [("6AM5", "o"), ("6AMU", "^")]:
                s = sub[sub.pid == pid]
                if len(s):
                    ax.scatter(s.toGIG, s.toDRG, s=42, marker=mk, edgecolor="k", linewidth=.4,
                               c=cmap.get(T, "k"), label=f"{pid} T{T}", zorder=3)
        lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([0, lim], [0, lim], "k--", lw=.8, alpha=.5, zorder=2)         # the diagonal
        ax.axhline(4, color="g", lw=.5, ls=":"); ax.axvline(4, color="g", lw=.5, ls=":")  # seated<4
        ax.set_xlabel("Cα-RMSD to GIG basin (Å)"); ax.set_ylabel("Cα-RMSD to DRG basin (Å)")
        ax.set_title("Q30+F-pocket dual-hotspot: basin occupancy vs prior diagonal")
        ax.legend(fontsize=6, ncol=2, loc="upper right")
        fig.tight_layout(); fig.savefig(f"{OUT}/basin_map.png", dpi=140)
        print(f"\nfigure -> {OUT}/basin_map.png")
    except Exception as e:
        print("figure skipped:", e)


if __name__ == "__main__":
    main()
