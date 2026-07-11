#!/usr/bin/env python3
"""How CONSISTENT are de-novo designs with each other under a given conditioning — i.e. is the
model confident (tight cluster) or is there high structural entropy (scattered), independent of
whether that cluster sits on the native register? Two independent metrics, same (source, pid,
context, cond) cells as score_denovo_designs.py:

  1. self-consistency: mean pairwise Cα RMSD among a cell's OWN frame-aligned peptide traces
     (_map_peptide's (10,3) array, same committor frame). Low = every design in the cell lands in
     ~the same place (confident, whether right or wrong); high = designs scatter (uncertain).
  2. torsion consistency: circular std of backbone phi/psi per peptide position within the cell,
     from the existing peptide_phipsi.csv (rebuilt here directly from the PDBs so it's self
     contained). Low = one dominant local backbone conformation; high = many distinct conformations
     realize "the same" Cα-RMSD.

Also reports n_hot (# ppi.hotspot_res residues, from jobs/rfd_denovo30_spec.tsv where available) per
cell so entropy can be read against conditioning strength, and to_cognate median (from
score_denovo_designs.score_occ) so "confident but wrong" vs "confident and right" are distinguishable.

Usage: python py/analyze_design_consistency.py [outputs/dir ...]
Output: outputs/denovo_scores/consistency_summary.csv, outputs/denovo_scores/consistency.png
"""
import os, sys, re, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import score_denovo_designs as S

OUT = f"{S.ROOT}/outputs/denovo_scores"; os.makedirs(OUT, exist_ok=True)

def circ_std_deg(angles_deg):
    """circular std (deg) of a list of angles in degrees; nan if <2 obs."""
    a = np.deg2rad(np.asarray(angles_deg, dtype=float))
    if len(a) < 2: return np.nan
    R = np.hypot(np.cos(a).mean(), np.sin(a).mean())
    R = min(max(R, 1e-12), 1.0)
    return float(np.rad2deg(np.sqrt(-2 * np.log(R))))

def dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    x = np.dot(v, w); y = np.dot(np.cross(b1n, v), w)
    return float(np.degrees(np.arctan2(y, x)))

def backbone(path):
    """-> dict(chain -> {resnum: {'N':.., 'CA':.., 'C':..}}) for the peptide chain only, keyed by
    the same 1..10 ordinal used elsewhere (first <=12-residue non-MHC/b2m/TCR chain, or merged-chain
    fallback), so it lines up with _map_peptide's peptide selection."""
    atoms = {}
    for l in open(path):
        if l.startswith("ATOM"):
            name = l[12:16].strip()
            if name not in ("N", "CA", "C"): continue
            c = l[21]; resi = int(l[22:26])
            atoms.setdefault(c, {}).setdefault(resi, {})[name] = np.array(
                [float(l[30:38]), float(l[38:46]), float(l[46:54])])
    return atoms

def peptide_phipsi(path):
    """Recompute per-position phi/psi for the SAME peptide chain _map_peptide would use, keeping the
    logic minimal (reuse chain-selection via S._chains + simple 8-12mer heuristic)."""
    ch = S._chains(path)
    if not ch: return None
    tcr_any = any(S._has_tcr("".join(n for n, _ in rs)) for rs in ch.values())
    mhc_c = None
    for c, rs in ch.items():
        if S.MHC_MOTIF in "".join(n for n, _ in rs): mhc_c = c; break
    pep_c = None
    for c, rs in ch.items():
        if c == mhc_c: continue
        s2 = "".join(n for n, _ in rs)
        if 8 <= len(rs) <= 12 and S.B2M_MOTIF not in s2 and not S._has_tcr(s2) and S.MHC_MOTIF not in s2:
            pep_c = c; break
    if pep_c is None or len(ch[pep_c]) != 10: return None
    bb = backbone(path).get(pep_c)
    if bb is None: return None
    resnums = sorted(bb)
    if len(resnums) != 10: return None
    out = []
    for i, r in enumerate(resnums):
        phi = psi = None
        if i > 0 and all(k in bb[resnums[i - 1]] for k in ("C",)) and all(k in bb[r] for k in ("N", "CA", "C")):
            phi = dihedral(bb[resnums[i - 1]]["C"], bb[r]["N"], bb[r]["CA"], bb[r]["C"])
        if i < len(resnums) - 1 and all(k in bb[r] for k in ("N", "CA", "C")) and "N" in bb[resnums[i + 1]]:
            psi = dihedral(bb[r]["N"], bb[r]["CA"], bb[r]["C"], bb[resnums[i + 1]]["N"])
        out.append((i + 1, phi, psi))
    return out

def load_spec_nhot():
    spec = f"{S.ROOT}/jobs/rfd_denovo30_spec.tsv"
    d = {}
    if not os.path.isfile(spec): return d
    for line in open(spec):
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 6: continue
        _, pid, ctx, cond, _, hot = parts[:6]
        hot = hot.strip()
        d[(pid, ctx, cond)] = len([t for t in hot.split(",") if t.strip()]) if hot else 0
    return d

def main():
    dirs = sys.argv[1:] or [f"{S.ROOT}/outputs/{x}/pdb" for x in ("grind", "ladder", "promising")] + \
                            [f"{S.ROOT}/outputs/rfd_denovo30/pdb", f"{S.ROOT}/outputs/rfd_maxcond/pdb"]
    dirs = [d for d in dirs if os.path.isdir(d)]
    files = S.gather(dirs)
    print(f"scanning {len(files)} PDBs...")
    nhot = load_spec_nhot()

    recs = []
    for f in files:
        meta = S.parse_meta(f)
        r = S._map_peptide(f)
        if r is None: continue
        pa, tcr, pep_len = r
        if pa is None: continue
        occ = S.occupancy(pa)
        toG = float(np.sqrt(((pa - S.GIG) ** 2).sum() / 10))
        toD = float(np.sqrt(((pa - S.DRG) ** 2).sum() / 10))
        cog = toG if meta["pid"] == "6AM5" else toD
        pp = peptide_phipsi(f)
        recs.append(dict(**meta, pa=pa, to_cognate=cog, register=occ["register"],
                         n_hot=nhot.get((meta["pid"], meta["context"], meta["cond"])), phipsi=pp))
    print(f"mapped {len(recs)} designs")

    rows = []
    for (src, pid, ctx, cond), grp in pd.DataFrame(recs).groupby(["source", "pid", "context", "cond"]):
        n = len(grp)
        pas = np.stack(grp["pa"].tolist())                     # (n,10,3)
        if n >= 2:
            d = pas[:, None] - pas[None, :]                     # (n,n,10,3)
            rmsd = np.sqrt((d ** 2).sum(-1).mean(-1))            # (n,n)
            iu = np.triu_indices(n, k=1)
            self_rmsd = float(rmsd[iu].mean())
        else:
            self_rmsd = np.nan
        # torsion circular std, averaged over the 10 positions (phi+psi combined)
        pp_lists = [p for p in grp["phipsi"] if p is not None]
        tstds = []
        if len(pp_lists) >= 2:
            for pos_i in range(10):
                phis = [pp[pos_i][1] for pp in pp_lists if pp[pos_i][1] is not None]
                psis = [pp[pos_i][2] for pp in pp_lists if pp[pos_i][2] is not None]
                if len(phis) >= 2: tstds.append(circ_std_deg(phis))
                if len(psis) >= 2: tstds.append(circ_std_deg(psis))
        torsion_std = float(np.nanmean(tstds)) if tstds else np.nan
        rows.append(dict(source=src, pid=pid, context=ctx, cond=cond, n=n,
                         n_hot=grp["n_hot"].iloc[0],
                         self_rmsd=round(self_rmsd, 2) if self_rmsd == self_rmsd else None,
                         torsion_std_deg=round(torsion_std, 1) if torsion_std == torsion_std else None,
                         to_cognate_med=round(float(grp["to_cognate"].median()), 2),
                         to_cognate_std=round(float(grp["to_cognate"].std()), 2) if n >= 2 else None,
                         register_mode=grp["register"].mode().iat[0] if len(grp["register"].mode()) else None,
                         register_purity=round(float((grp["register"] == grp["register"].mode().iat[0]).mean()), 2) if n >= 2 else None))
    summ = pd.DataFrame(rows).sort_values(["source", "pid", "context", "cond"])
    summ.to_csv(f"{OUT}/consistency_summary.csv", index=False)
    print(summ.to_string(index=False))

    d2 = summ[summ.self_rmsd.notna() & summ.n_hot.notna()]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axes[0]
    for ctx, mk, cl in [("withTCR", "o", "#2a78d6"), ("noTCR", "s", "#e08a2e")]:
        sub = d2[d2.context == ctx]
        ax.scatter(sub.n_hot, sub.self_rmsd, s=30 + 4 * sub.n, marker=mk, color=cl, alpha=.75,
                   edgecolor="k", lw=.4, label=ctx)
    ax.set_xlabel("# conditioning hotspots (n_hot)"); ax.set_ylabel("self-consistency: mean pairwise Cα RMSD (Å)")
    ax.set_title("Structural entropy vs conditioning strength\n(low = model confident/tight; high = scattered)")
    ax.legend()

    ax = axes[1]
    d3 = summ[summ.self_rmsd.notna()]
    ax.scatter(d3.self_rmsd, d3.to_cognate_std, s=30, color="#5b3fa0", alpha=.8, edgecolor="k", lw=.4)
    ax.set_xlabel("self-consistency: mean pairwise Cα RMSD (Å)")
    ax.set_ylabel("std of to-cognate RMSD within cell (Å)")
    ax.set_title("Does a tight self-consistent cluster\nalso mean tight accuracy-to-native spread?")

    ax = axes[2]
    d4 = summ[summ.torsion_std_deg.notna() & summ.n_hot.notna()]
    for ctx, mk, cl in [("withTCR", "o", "#2a78d6"), ("noTCR", "s", "#e08a2e")]:
        sub = d4[d4.context == ctx]
        ax.scatter(sub.n_hot, sub.torsion_std_deg, s=30 + 4 * sub.n, marker=mk, color=cl, alpha=.75,
                   edgecolor="k", lw=.4, label=ctx)
    ax.set_xlabel("# conditioning hotspots (n_hot)"); ax.set_ylabel("mean circular std of backbone φ/ψ (deg)")
    ax.set_title("Torsional entropy vs conditioning strength")
    ax.legend()
    plt.tight_layout()
    fig.savefig(f"{OUT}/consistency.png", dpi=150)
    print(f"\nwrote {OUT}/consistency_summary.csv and {OUT}/consistency.png")

if __name__ == "__main__":
    main()
