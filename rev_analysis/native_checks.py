"""Native-frame controls (run under the mdtraj env):
 (1) anchor FLIP RATE  -- fraction of native MD frames whose F-pocket argmin != the crystal anchor.
     Frame-invariant: argmin of |pep_i - fpocket_centroid| uses peptide+F-pocket Ca from the SAME frame.
 (7) offset FALSE-POSITIVE FLOOR -- run the +-3 register-slippage optimization on native frames
     (in-register by construction). Whatever fraction of NATIVES "improves" at a nonzero offset is the
     floor; the design excess over this floor is the real slipped-register signal.
"""
import sys; sys.path.insert(0, "/global/scratch/users/sergiomar10/if-mhc/rev_analysis")
import numpy as np
import nmc  # ROOT-patched native_md_components

JOBS = {"6AM5": ("ifmhc_6AM5_md_310K", 9),    # GIG anchors P10 -> 0-indexed 9
        "6AMU": ("ifmhc_6AMU_md_310K", 8)}    # DRG anchors P9  -> 0-indexed 8

def kabsch_apply(P, Q, X):
    Pc, Qc = P.mean(0), Q.mean(0)
    V, S, Wt = np.linalg.svd((P - Pc).T @ (Q - Qc)); d = np.sign(np.linalg.det(V @ Wt))
    R = V @ np.diag([1, 1, d]) @ Wt
    return (X - Pc) @ R + Qc

for cry, (job, anchor) in JOBS.items():
    models, seg = nmc.load_md(job, nframes=1000)
    cry_model = nmc.load_crystal(cry)
    cpep = nmc.comp_ca(cry_model, "pep"); ca1a2 = nmc.comp_ca(cry_model, "a1a2")
    cpep_arr = np.array([cpep[("P", i)] for i in range(10)])
    flips = 0; margins = []; argmins = []
    off_nonzero = 0; off_impr = []
    n = 0
    for m in models:
        pep = nmc.comp_ca(m, "pep"); fp = nmc.comp_ca(m, "fpocket"); a12 = nmc.comp_ca(m, "a1a2")
        if len(pep) < 10 or len(fp) < 4:
            continue
        n += 1
        P = np.array([pep[("P", i)] for i in range(10)])
        cent = np.mean(np.array(list(fp.values())), 0)
        d = np.linalg.norm(P - cent, axis=1)
        am = int(d.argmin()); argmins.append(am + 1)
        if am != anchor:
            flips += 1
        margins.append(float(d[8] - d[9]))                 # d_P9 - d_P10
        # (7) offset floor: superpose frame a1a2 -> crystal a1a2, map peptide, slide +-3 vs crystal pep
        keys = [k for k in a12 if k in ca1a2]
        if len(keys) >= 30:
            Pg = np.array([a12[k] for k in keys]); Qg = np.array([ca1a2[k] for k in keys])
            Pt = kabsch_apply(Pg, Qg, P)
            r0 = np.sqrt(((Pt - cpep_arr) ** 2).sum() / 10)
            best = (r0, 0)
            for k in range(-3, 4):
                if k == 0: continue
                idx = np.arange(10) + k; ok = (idx >= 0) & (idx < 10)
                if ok.sum() < 6: continue
                r = np.sqrt(((Pt[np.arange(10)[ok]] - cpep_arr[idx[ok]]) ** 2).sum() / ok.sum())
                if r < best[0]: best = (r, k)
            if best[1] != 0 and best[0] < r0 - 0.5:
                off_nonzero += 1; off_impr.append(r0 - best[0])
    import collections
    print(f"\n=== {cry} native MD (n={n} frames, crystal anchor P{anchor+1}) ===")
    print(f"  (1) FLIP RATE: {flips}/{n} = {100*flips/n:.2f}% of native frames pick the WRONG anchor")
    print(f"      argmin distribution: {dict(sorted(collections.Counter(argmins).items()))}")
    print(f"      margin(P9-P10) median={np.median(margins):+.2f}A  (|.|<0.5 in {100*np.mean(np.abs(margins)<0.5):.1f}%)")
    print(f"  (7) OFFSET FLOOR: {off_nonzero}/{n} = {100*off_nonzero/n:.1f}% of IN-REGISTER natives 'improve' at "
          f"nonzero offset  (this is the false-positive floor; design 81.6% must be read as excess over this)")
print("\nNATIVE_DONE")
