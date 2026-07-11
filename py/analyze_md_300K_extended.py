#!/usr/bin/env python3
"""Calibration analysis for the extended 300K/50ns native MD run (6AM5+6AMU),
mirroring the methodology already applied to the extended 370K run in
notebooks/01_md_calibrated_baseline.ipynb: per-frame (toGIG, toDRG, F-pocket
occupancy) scoring via native_md_components.load_md, then convergence
diagnostics (autocorrelation time, effective sample size, net drift) and
calibrated mean+-3SD acceptance bands.

Env: esmfold2 (mdtraj 1.11, biopython 1.87).
"""
import sys, os
sys.path.insert(0, "/home/ubuntu/if-mhc/py")
import numpy as np
import pandas as pd
import native_md_components as NM
import score_denovo_designs as S

ROOT = "/home/ubuntu/if-mhc"
NF = 1000

MD_JOBS = [
    ("6AM5", "300K_ext", "ifmhc_6AM5_md_300K_ext"),
    ("6AMU", "300K_ext", "ifmhc_6AMU_md_300K_ext"),
    ("6AM5", "370K", "lbv69"),
    ("6AMU", "370K", "p1yv2"),
]


def score_frame(model):
    ms_start, ms = model.chain_seqs["mhc"]
    mca = np.array([model.ca[ms_start + i] for i in range(len(ms))])
    m = ms.find(S.MHC_MOTIF)
    loc = list(range(m, min(m + 179, len(ms))))
    k = S._offset("".join(ms[i] for i in loc), S.REFSEQ)
    idx = [j for j, i in enumerate(loc) if 0 <= j + k < len(S.REF_CA)]
    if len(idx) < 50:
        return None
    R, t = S._robust(mca[[loc[j] for j in idx]], S.REF_CA[[j + k for j in idx]])
    ps, pseq = model.chain_seqs["pep"]
    pa = np.array([model.ca[ps + i] for i in range(10)]) @ R + t
    o = S.occupancy(pa)
    return dict(toGIG=float(np.sqrt(((pa - S.GIG) ** 2).sum() / 10)),
                toDRG=float(np.sqrt(((pa - S.DRG) ** 2).sum() / 10)),
                fpocket_pos=o["fpocket_pos"], fpocket_dist=o["fpocket_dist"],
                bpocket_dist=o["bpocket_dist"], register=o["register"])


def autocorr_time(x, max_lag_frac=0.5):
    """Integrated autocorrelation time via windowed ACF sum, cut off when ACF
    first drops below 0.05 (or at max_lag_frac*N, whichever is sooner)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    xc = x - x.mean()
    var = (xc ** 2).sum()
    if var == 0:
        return 0.0, 1.0
    max_lag = int(n * max_lag_frac)
    tau = 1.0
    for lag in range(1, max_lag):
        acf = (xc[:-lag] * xc[lag:]).sum() / var
        if acf < 0.05:
            break
        tau += 2 * acf
    return tau, tau  # integrated autocorrelation time in units of "frames"


rows = []
PA = {}
for pid, T, job in MD_JOBS:
    models, seg = NM.load_md(job, NF)
    PA[(pid, T)] = models
    scored = [score_frame(m) for m in models]
    scored = [s for s in scored if s is not None]
    for i, s in enumerate(scored):
        s["pid"] = s.get("pid", pid); s["T"] = T; s["frame"] = i
        rows.append({**s, "pid": pid, "T": T, "frame": i})

MDF = pd.DataFrame(rows)
out_dir = f"{ROOT}/outputs/native_md_rmsd"
os.makedirs(out_dir, exist_ok=True)
MDF.to_csv(f"{out_dir}/md_300K_extended_per_frame.csv", index=False)

print("=" * 100)
print("CONVERGENCE DIAGNOSTICS -- extended 300K (50ns) vs extended 370K (50ns)")
print("=" * 100)

# assume evenly spaced frames spanning 50ns production (matches settings.json productionTime=50)
PROD_NS = 50.0

for pid, T, job in MD_JOBS:
    g = MDF[(MDF.pid == pid) & (MDF["T"] == T)].sort_values("frame")
    cog = "toGIG" if pid == "6AM5" else "toDRG"
    x = g[cog].values
    n = len(x)
    dt_ns = PROD_NS / n
    tau_frames, _ = autocorr_time(x)
    tau_ns = tau_frames * dt_ns
    ess = n / (2 * tau_frames) if tau_frames > 0 else n
    drift = x[-1] - x[0]
    # net drift over a running mean of the first/last 10% to reduce single-frame noise
    k = max(1, n // 10)
    drift_smoothed = x[-k:].mean() - x[:k].mean()
    print(f"[{pid}/{T}] n_frames={n}  {cog}: mean={x.mean():.2f}+-{x.std():.2f}  "
          f"tau~{tau_ns:.2f}ns  ESS~{ess:.1f}/{n}  "
          f"net_drift(first->last frame)={drift:+.2f}A  drift(smoothed 10%)={drift_smoothed:+.2f}A")

print()
print("=" * 100)
print("CALIBRATED BANDS (mean + 3SD) -- 300K_ext vs 370K, side by side")
print("=" * 100)


def make_band(pid, T, ksd=3.0):
    c = MDF[(MDF.pid == pid) & (MDF["T"] == T)]
    cog = "toGIG" if pid == "6AM5" else "toDRG"
    want = 10 if pid == "6AM5" else 9
    return dict(pid=pid, T=T, cog=cog, want=want,
                rmsd_hi=float(c[cog].mean() + ksd * c[cog].std()),
                fdepth_lo=float(c.fpocket_dist.mean() - ksd * c.fpocket_dist.std()),
                fdepth_hi=float(c.fpocket_dist.mean() + ksd * c.fpocket_dist.std()))


BANDS = {}
for pid in ["6AM5", "6AMU"]:
    for T in ["300K_ext", "370K"]:
        b = make_band(pid, T)
        BANDS[(pid, T)] = b
        print(f"  {pid}/{T}: {b['cog']} <= {b['rmsd_hi']:.2f} A  AND  F-pocket p{b['want']} within "
              f"[{b['fdepth_lo']:.2f}, {b['fdepth_hi']:.2f}] A")

print()
print("=" * 100)
print("CROSS-CLOUD SEPARATION CHECK @ 300K_ext (are GIG/DRG clouds still cleanly resolved?)")
print("=" * 100)
g = MDF[(MDF.pid == "6AM5") & (MDF["T"] == "300K_ext")]
d = MDF[(MDF.pid == "6AMU") & (MDF["T"] == "300K_ext")]
gig_toDRG_min = g.toDRG.min(); drg_toGIG_min = d.toGIG.min()
gig_toGIG_hi = g.toGIG.quantile(.975); drg_toDRG_hi = d.toDRG.quantile(.975)
print(f"GIG basin: toGIG {g.toGIG.mean():.2f}+-{g.toGIG.std():.2f} (97.5%={gig_toGIG_hi:.2f}) | its toDRG floor={gig_toDRG_min:.2f}")
print(f"DRG basin: toDRG {d.toDRG.mean():.2f}+-{d.toDRG.std():.2f} (97.5%={drg_toDRG_hi:.2f}) | its toGIG floor={drg_toGIG_min:.2f}")
resolved = not (gig_toDRG_min < drg_toDRG_hi or drg_toGIG_min < gig_toGIG_hi)
print(f"-> registers {'CLEANLY RESOLVED' if resolved else 'OVERLAP'}")

print()
print("=" * 100)
print("RE-SCORING THE DE-NOVO CORPUS UNDER THE 300K_ext BAND")
print("=" * 100)
per_design = pd.read_csv(f"{ROOT}/outputs/denovo_scores/per_design.csv")
DN = per_design[per_design.pep_len == 10].copy()


def in_basin(toGIG, toDRG, fpos, fdist, band):
    cogval = toGIG if band["cog"] == "toGIG" else toDRG
    return bool(cogval <= band["rmsd_hi"] and fpos == band["want"]
                and band["fdepth_lo"] <= fdist <= band["fdepth_hi"])


def classify(row, T):
    if in_basin(row.toGIG, row.toDRG, row.fpocket_pos, row.fpocket_dist, BANDS[("6AM5", T)]):
        return "GIG"
    if in_basin(row.toGIG, row.toDRG, row.fpocket_pos, row.fpocket_dist, BANDS[("6AMU", T)]):
        return "DRG"
    return "neither"


DN["class_300Kext"] = [classify(r, "300K_ext") for r in DN.itertuples()]
DN["class_370K"] = [classify(r, "370K") for r in DN.itertuples()]
n_300 = int((DN.class_300Kext != "neither").sum())
n_370 = int((DN.class_370K != "neither").sum())
print(f"de-novo designs scored: {len(DN)}")
print(f"  inside a native cloud @300K_ext (50ns): {n_300}")
print(f"  inside a native cloud @370K (50ns)    : {n_370}")
if n_300:
    print(DN[DN.class_300Kext != "neither"][["source", "pid", "toGIG", "toDRG", "fpocket_pos", "fpocket_dist", "class_300Kext"]].to_string())

DN.to_csv(f"{out_dir}/denovo_classified_300Kext_vs_370K.csv", index=False)
print(f"\nwrote {out_dir}/md_300K_extended_per_frame.csv")
print(f"wrote {out_dir}/denovo_classified_300Kext_vs_370K.csv")
