"""FROZEN analysis pool + inclusion rules -- the single source of truth for every
reviewer-driven re-analysis. Import this; never redefine a pool inline.

Decisions (pinned 2026-07-24, do not drift):
  POOL          = the full converged de-novo corpus = every allcond150 cell that is
                  NOT a templating (fix*) cell and NOT the null (null0) cell.
                  N is whatever the completed campaign holds (report it, don't hardcode).
                  This is LARGER than the paper's frozen 8,722 snapshot; new analyses use
                  the full corpus and say so. (paper text should be reconciled to match.)
  REGISTER_DEFINED (the only population where register is a meaningful quantity):
                  extended  AND  groove-placed  AND  forward-threaded.
  Hit (DRG register, the only well-converged band, ESS~13):
                  recovery = 6AMU & fpocket_pos==9 & toDRG<=DRG_HI
                  crossing = 6AM5 & fpocket_pos==9 & toDRG<=DRG_HI
                  DRG_HI = 1.58 (95%-CI upper of the 310K DRG band).  Point band = 1.48.
"""
import os, numpy as np, pandas as pd
RA = "/global/scratch/users/sergiomar10/if-mhc/rev_analysis"
ROOT = "/global/scratch/users/sergiomar10/if-mhc"

DRG_HI, DRG_POINT = 1.58, 1.48
GROOVE_CUT = 8.0            # min(toGIG,toDRG) < 8.0 A  -> "in the groove"
E2E_LO, E2E_HI = 17.0, 30.0  # native end-to-end 22.6 (GIG) / 23.8 (DRG); window +-~6A

def contact_counts():
    """cond -> dict(n, mhc, tcr) from the LIVE allcond150 spec (both crystals identical count)."""
    d = {}
    for ln in open(f"{ROOT}/jobs/allcond150_spec.tsv"):
        p = ln.rstrip("\n").split("\t")
        if len(p) >= 6:
            hs = [h for h in p[5].split(",") if h.strip()]
            d[p[1]] = dict(n=len(hs), mhc=sum(h[0] == "A" for h in hs), tcr=sum(h[0] in "DE" for h in hs))
    return d

def load(aug=True):
    """Return the cache DataFrame with pool flags added. aug -> use the analyze_all augmented cache
    if present (has margin/reg_coord/extended/groove), else the base cache."""
    p = f"{RA}/design_cache_aug.pkl" if (aug and os.path.exists(f"{RA}/design_cache_aug.pkl")) else f"{RA}/design_cache.pkl"
    DF = pd.read_pickle(p).reset_index(drop=True)
    if "extended" not in DF:
        DF["extended"] = DF.e2e.between(E2E_LO, E2E_HI)
    if "groove" not in DF:
        DF["groove"] = DF[["toGIG", "toDRG"]].min(1) < GROOVE_CUT
    DF["forward"] = DF.thread == "forward"
    DF["register_defined"] = DF.extended & DF.groove & DF.forward
    NCON = contact_counts()
    DF["ncon"] = DF.cond.map(lambda c: NCON.get(c, dict(n=np.nan))["n"])
    # hit flags (DRG band, within 95% CI)
    seat9 = DF.fpos == 9
    DF["recovery"] = (DF.cry == "6AMU") & seat9 & (DF.toDRG <= DRG_HI)
    DF["crossing"] = (DF.cry == "6AM5") & seat9 & (DF.toDRG <= DRG_HI)
    DF["hit"] = DF.recovery | DF.crossing
    DF["hit_point"] = (seat9 & (DF.toDRG <= DRG_POINT) &
                       ((DF.cry == "6AMU") | (DF.cry == "6AM5")))
    return DF

def denovo(DF):     return DF[DF.g == "denovo"].copy()
def regdef(DF):     return DF[(DF.g == "denovo") & DF.register_defined].copy()

def coords():
    Z = np.load(f"{RA}/design_coords.npz")
    return Z["coords"], Z["GIG"], Z["DRG"]

if __name__ == "__main__":
    DF = load()
    dn = denovo(DF); rd = regdef(DF)
    print("FROZEN POOL")
    print(f"  full corpus rows      : {len(DF)}")
    print(f"  de-novo POOL          : {len(dn)}")
    print(f"    extended            : {dn.extended.sum()}")
    print(f"    groove-placed       : {dn.groove.sum()}")
    print(f"    forward-threaded    : {dn.forward.sum()}")
    print(f"    REGISTER_DEFINED    : {len(rd)}  <- inclusion rule for all register analyses")
    print(f"  de-novo hits (DRG CI) : recovery {int(dn.recovery.sum())}, crossing {int(dn.crossing.sum())}, "
          f"total {int(dn.hit.sum())}")
    print(f"  contact-count tiers (de-novo cells):")
    nc = dn.groupby('cond').agg(N=('hit','size'), ncon=('ncon','first'), hits=('hit','sum')).sort_values('ncon')
    print(nc.to_string())
