"""Five-gate scoring framework (reviewer spec). Every arm passes ALL FIVE or the readout lies:
  1 DIRECTION      -- forward/reverse threading (argmin), reported first
  2 PLACEMENT      -- groove gate, strict (RMSD<8) AND permissive (physical centroid), both ratios
  3 PHASE          -- offset-optimized RMSD slippage vs a SAME-PIPELINE floor (fix8, not MD frames)
  4 REGISTER COORD -- d(GIG)-d(DRG), continuous, the PRIMARY endpoint (never the binary)
  5 EFFECT SIZE    -- AUC / Cliff's delta next to every p (auc_cliff helper)
Usage: score_cell(glob) -> dict of per-design arrays; report(name, cell); compare(nameA,a, nameB,b).
"""
import sys, glob, os; sys.path.insert(0, "/global/scratch/users/sergiomar10/if-mhc/rev_analysis")
import numpy as np, score_sd as S
from scipy import stats
GIG, DRG = S.GIG, S.DRG
E2E_LO, E2E_HI, GROOVE = 17.0, 30.0, 8.0
GC = (GIG.mean(0) + DRG.mean(0)) / 2
PHYS_R = float(np.linalg.norm(np.array([GIG.mean(0), DRG.mean(0)]) - GC, axis=1).max()) + 6.0

def score_cell(pat):
    """pat: glob of design PDBs. Returns per-design dict of the raw quantities all 5 gates need."""
    rows = []
    for f in glob.glob(pat):
        if "traj" in f or "_split" in f: continue
        r = S._map_peptide(f)
        if r is None: continue
        pa, _, pl = r
        if pa is None or len(pa) != 10: continue
        b = os.path.basename(f); cry = "6AMU" if b.startswith("6AMU") or "6AMU" in b else "6AM5"
        toG = float(np.sqrt(((pa - GIG) ** 2).sum() / 10)); toD = float(np.sqrt(((pa - DRG) ** 2).sum() / 10))
        sc = S.occupancy(pa); e2e = float(np.linalg.norm(pa[0] - pa[9]))
        # phase: best non-zero index offset vs OWN reference
        ref = GIG if cry == "6AM5" else DRG
        r0 = np.sqrt(((pa - ref) ** 2).sum() / 10); best = (r0, 0)
        for k in range(-3, 4):
            if k == 0: continue
            idx = np.arange(10) + k; ok = (idx >= 0) & (idx < 10)
            if ok.sum() < 6: continue
            rr = np.sqrt(((pa[np.arange(10)[ok]] - ref[idx[ok]]) ** 2).sum() / ok.sum())
            if rr < best[0]: best = (rr, k)
        rows.append(dict(cry=cry, toGIG=toG, toDRG=toD, fpos=sc["fpocket_pos"], thread=sc["threading"],
                         e2e=e2e, cdist=float(np.linalg.norm(pa.mean(0) - GC)),
                         slip=(best[1] != 0 and best[0] < r0 - 0.5)))
    return rows

def report(name, rows):
    if not rows: print(f"  {name}: n=0"); return None
    import numpy as np
    n = len(rows); fwd = np.mean([r["thread"] == "forward" for r in rows])
    mind = np.array([min(r["toGIG"], r["toDRG"]) for r in rows])
    grv = np.mean(mind < GROOVE); phys = np.mean([r["cdist"] < PHYS_R for r in rows])
    ext = np.array([E2E_LO <= r["e2e"] <= E2E_HI for r in rows])
    fwdm = np.array([r["thread"] == "forward" for r in rows])
    regdef = ext & (mind < GROOVE) & fwdm
    rc = np.array([r["toGIG"] - r["toDRG"] for r in rows])   # >0 = DRG-like
    slip = np.mean([r["slip"] for r in rows if (E2E_LO <= r["e2e"] <= E2E_HI and min(r["toGIG"],r["toDRG"])<GROOVE and r["thread"]=="forward")] or [0])
    toown = np.array([r["toGIG"] if r["cry"]=="6AM5" else r["toDRG"] for r in rows])
    print(f"  {name:<16} n={n:4d} | 1.fwd={100*fwd:4.0f}% | 2.groove RMSD={100*grv:4.0f}% phys={100*phys:4.0f}% | "
          f"3.slip(regdef)={100*slip:4.0f}% | 4.regcoord med={np.median(rc[regdef]) if regdef.any() else float('nan'):+.2f} "
          f"| to-own med(fwd)={np.median(toown[fwdm]) if fwdm.any() else float('nan'):.2f} best={toown.min():.2f}")
    return dict(rc=rc[regdef], toown_fwd=toown[fwdm])

def auc_cliff(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2 or len(b) < 2: return (float('nan'),)*3
    U, p = stats.mannwhitneyu(a, b, alternative="two-sided"); auc = U/(len(a)*len(b))
    return auc, 2*auc-1, p

if __name__ == "__main__":
    ROOT = "/global/scratch/users/sergiomar10/if-mhc"
    print("5-GATE READOUT -- template-identity ladder (which peptide residues templated) + fix references")
    print(f"  [physical radius {PHYS_R:.1f}A; register-defined = extended&groove&forward; phase floor = fix8]\n")
    cells = [("ti_cterm1(P10)", f"{ROOT}/outputs/exp_tmplid/pdb/*ti_cterm1*.pdb"),
             ("ti_cterm2(P9-10)",f"{ROOT}/outputs/exp_tmplid/pdb/*ti_cterm2*.pdb"),
             ("ti_nterm1(P1)",  f"{ROOT}/outputs/exp_tmplid/pdb/*ti_nterm1*.pdb"),
             ("ti_nterm2(P1-2)", f"{ROOT}/outputs/exp_tmplid/pdb/*ti_nterm2*.pdb"),
             ("ti_mid2(P5-6)",   f"{ROOT}/outputs/exp_tmplid/pdb/*ti_mid2*.pdb"),
             ("fix8(ref)",       f"{ROOT}/outputs/allcond150/pdb/*fix8*.pdb"),
             ("fix2(ref)",       f"{ROOT}/outputs/allcond150/pdb/*fix2*.pdb")]
    store = {}
    for name, pat in cells:
        store[name] = report(name, score_cell(pat))
    print("\n  EFFECT SIZE (gate 5): to-own(fwd) vs fix8 reference -- does any tmplid arm reach fix8?")
    ref = store.get("fix8(ref)")
    for name in ("ti_cterm2(P9-10)", "ti_mid2(P5-6)", "ti_nterm2(P1-2)"):
        s = store.get(name)
        if s is not None and ref is not None:
            auc, cd, p = auc_cliff(s["toown_fwd"], ref["toown_fwd"])
            print(f"    {name:<16} vs fix8: AUC={auc:.3f} Cliff={cd:+.3f} p={p:.1e}  (AUC>0.5 => WORSE than fix8)")
    print("GATE5_DONE")
