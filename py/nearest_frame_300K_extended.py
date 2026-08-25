#!/usr/bin/env python3
"""For the 5 known crossing candidates (and the best own-register de-novo design),
find the SINGLE closest individual frame in the 300K_ext native trajectory of the
target register -- not just the mean-band comparison. Thermal snapshots wander
further from the mean than the mean-based band alone captures, so a design can be
closer to one real instantaneous conformation than to the register's average.

Mirrors notebook 01's 4c analysis (which did this against 370K); this is the same
comparison against the better-converged 300K_ext run.

Env: esmfold2 (mdtraj 1.11, biopython 1.87).
"""
import sys
sys.path.insert(0, "/home/ubuntu/if-mhc/py")
import numpy as np
import pandas as pd
import native_md_components as NM
import score_denovo_designs as S

ROOT = "/home/ubuntu/if-mhc"
NF = 1000

CROSS5 = [
    "outputs/rfd_maxcond/pdb/6AM5_k18_44.pdb",
    "outputs/promising/pdb/6AM5_L4_expanded_w1_4992623_0.pdb",
    "outputs/rfd_maxcond/pdb/6AM5_k18_12.pdb",
    "outputs/rfd_maxcond/pdb/6AM5_k18_109.pdb",
    "outputs/promising/pdb/6AMU_L1_nterm_wA_0082638_0.pdb",
]

MD_JOBS_300 = [("6AM5", "ifmhc_6AM5_md_300K_ext"), ("6AMU", "ifmhc_6AMU_md_300K_ext")]


def frame_pa(model):
    ms_start, ms = model.chain_seqs["mhc"]
    mca = np.array([model.ca[ms_start + i] for i in range(len(ms))])
    m = ms.find(S.MHC_MOTIF)
    loc = list(range(m, min(m + 179, len(ms))))
    k = S._offset("".join(ms[i] for i in loc), S.REFSEQ)
    idx = [j for j, i in enumerate(loc) if 0 <= j + k < len(S.REF_CA)]
    if len(idx) < 50:
        return None
    R, t = S._robust(mca[[loc[j] for j in idx]], S.REF_CA[[j + k for j in idx]])
    ps, _ = model.chain_seqs["pep"]
    return np.array([model.ca[ps + i] for i in range(10)]) @ R + t


PA_300 = {}
for pid, job in MD_JOBS_300:
    models, _ = NM.load_md(job, NF)
    pa_list = [p for p in (frame_pa(m) for m in models) if p is not None]
    PA_300[pid] = np.array(pa_list)
    print(f"{pid}/300K_ext: {len(pa_list)} frames loaded")

# bands from the previous run (mean +/- 3SD on the cognate axis) for reference
BANDS_300 = {}
for pid in ["6AM5", "6AMU"]:
    ref = S.GIG if pid == "6AM5" else S.DRG
    d = np.sqrt(((PA_300[pid] - ref) ** 2).sum(axis=1) / 10)
    BANDS_300[pid] = dict(mean=float(d.mean()), std=float(d.std()), hi=float(d.mean() + 3 * d.std()))

print()
print("=" * 110)
print(f"{'design':45s} {'to_other(mean)':>14s} {'300K band hi':>12s} {'closest 300K frame':>19s} {'delta vs band':>14s}")
print("=" * 110)

rows = []
for f in CROSS5:
    r = S.score_occ(f)
    if r is None or r.get("toGIG") is None:
        print(f"{f}: could not score")
        continue
    my_pid = "6AM5" if "6AM5" in f else "6AMU"
    other_pid = "6AMU" if my_pid == "6AM5" else "6AM5"
    pa_design, _, _ = S._map_peptide(f)
    other_ref = S.GIG if other_pid == "6AM5" else S.DRG
    to_other_mean = float(np.sqrt(((pa_design - other_ref) ** 2).sum() / 10))
    # distance to every individual frame of the OTHER register's 300K trajectory
    dists = np.sqrt(((PA_300[other_pid] - pa_design[None, :, :]) ** 2).sum(axis=2).mean(axis=1))
    best_i = int(dists.argmin())
    best_d = float(dists[best_i])
    band_hi = BANDS_300[other_pid]["hi"]
    print(f"{f.split('/')[-1]:45s} {to_other_mean:14.2f} {band_hi:12.2f} "
          f"frame#{best_i:<5d}={best_d:6.2f}A {best_d - band_hi:+14.2f}")
    rows.append(dict(file=f, my_pid=my_pid, other_pid=other_pid, to_other_mean=to_other_mean,
                      band_hi_300=band_hi, closest_frame_idx=best_i, closest_frame_dist=best_d,
                      inside_band=best_d <= band_hi))

OUT = pd.DataFrame(rows)
OUT.to_csv(f"{ROOT}/outputs/native_md_rmsd/cross5_nearest_300Kext_frame.csv", index=False)
print(f"\nwrote outputs/native_md_rmsd/cross5_nearest_300Kext_frame.csv")

print()
print("=" * 110)
print("For comparison: closest 300K_ext frame to EACH design's OWN register (not crossing)")
print("=" * 110)
for f in CROSS5:
    my_pid = "6AM5" if "6AM5" in f else "6AMU"
    pa_design, _, _ = S._map_peptide(f)
    if pa_design is None:
        continue
    dists = np.sqrt(((PA_300[my_pid] - pa_design[None, :, :]) ** 2).sum(axis=2).mean(axis=1))
    best_i = int(dists.argmin()); best_d = float(dists[best_i])
    own_ref = S.GIG if my_pid == "6AM5" else S.DRG
    to_own_mean = float(np.sqrt(((pa_design - own_ref) ** 2).sum() / 10))
    print(f"{f.split('/')[-1]:45s} own={my_pid}  to_own_mean={to_own_mean:.2f}  "
          f"closest_own_frame=#{best_i} d={best_d:.2f}A  band_hi={BANDS_300[my_pid]['hi']:.2f}")
