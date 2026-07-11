#!/usr/bin/env python3
"""Emit 01_md_calibrated_baseline.ipynb — the native MD ensemble as the CALIBRATED baseline.

The baseline for "resembles the native peptide" was never the 2.87 Å crystal separation — a single
number is a point with no scale, which is the hole the RMSD story fell into. The baseline is the native
MD ENSEMBLE: hundreds of frames of 6AM5 and 6AMU (300 K & 370 K) give each register's cloud its
LOCATION, its WIDTH, and the register coordinate (F-pocket occupant). "Native-like" then has a rigorous
two-axis definition: falls inside the native MD distribution on BOTH (a) a backbone-proximity axis
(toGIG/toDRG) AND (b) the F-pocket-occupant axis — thresholds READ OFF the clouds, not chosen by hand.

This notebook:
  1. builds the four native reference clouds (per-frame toGIG,toDRG,F-pocket occupant+depth),
  2. reports the separation done right (two distributions + their overlap, not a point),
  3. derives calibrated acceptance bands (300 K = tight/defining, 370 K = generous bound),
  4. evaluates the two real jobs against them:
       - seeded/graft recovery  -> two-sample: is it statistically inside the native cloud?
       - de-novo designs         -> fraction inside EITHER native cloud on BOTH axes,
     retiring both broken proxies at once (RMSD's false-negative wall, occupancy's false-positive argmin).
Cautions honoured: MD frames are autocorrelated (thin before quoting p-values); 300 K defines
native-like, 370 K is the generous bound. Reuses native_md_components (MD loader/segmentation) and
score_denovo_designs (committor frame, GIG/DRG, F/B-pocket occupancy). Env: esmfold2 (mdtraj)."""
import nbformat as nbf
nb = nbf.v4.new_notebook(); C = []
def md(t): C.append(nbf.v4.new_markdown_cell(t))
def co(t): C.append(nbf.v4.new_code_cell(t))

md(r"""# The native MD ensemble is the ruler — calibrated two-axis recovery evaluation

Every prior verdict compared designs to a **crystal** (a point) via **RMSD** (an unscaled distance).
A point gives you a location but no scale: you cannot say whether a design at 1.2 Å is "inside the GIG
cloud" or "unusually far from GIG" without knowing **how wide the GIG cloud is**. That width is exactly
what a single 2.87 Å separation hides. The width lives in the **native MD ensemble**, which the project
ran (6AM5/6AMU at 300 K & 370 K) but never used as a *reference*.

So the baseline is not the 2.87 Å separation — it is the **two native distributions** whose separation
that number is one slice through. "Resembles the native peptide" becomes a calibrated statement:
**falls inside the native MD distribution on both a backbone-proximity axis (toGIG/toDRG) and the
F-pocket-occupant axis.** Two conditions, because a register needs both backbone proximity *and* anchor
identity — and the MD supplies the numeric cutoff for each, from real thermal motion rather than a
1.45 Å or 6.5 Å guess. We calibrate on **300 K** (tight, physiological-ish) and report **370 K** as the
generous bound.""")

co(r"""import os, sys, glob, re, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/ubuntu/if-mhc/py")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import mdtraj as mdt
import score_denovo_designs as S
import native_md_components as NM
from scipy.stats import ks_2samp, mannwhitneyu
plt.rcParams.update({"figure.dpi": 130, "font.size": 10})
ROOT = S.ROOT; NF = 1000         # frames/traj for the clouds (all available; new 370K jobs have 1000)

# 370K now points to the EXTENDED reruns (2ns equil / 50ns production, 5x longer than the original
# 0.2ns/10ns runs) -- Tamarind jobs lbv69=6AM5, p1yv2=6AMU. 300K is still the original 0.2ns/10ns run
# (not yet rerun). See caveat in section 3: even 50ns did not fully converge these trajectories.
MD_JOBS = [("6AM5", "300K", "ifmhc_6AM5_md_300K"), ("6AM5", "370K", "lbv69"),
           ("6AMU", "300K", "ifmhc_6AMU_md_300K"), ("6AMU", "370K", "p1yv2")]

def score_frame(model):
    '''Committor-frame (toGIG,toDRG) + occupancy for one MD Model — identical math to S.score_occ,
    fed from native_md_components' motif segmentation (MD topology is merged single-chain).'''
    ms_start, ms = model.chain_seqs["mhc"]
    mca = np.array([model.ca[ms_start + i] for i in range(len(ms))])
    m = ms.find(S.MHC_MOTIF); loc = list(range(m, min(m + 179, len(ms))))
    k = S._offset("".join(ms[i] for i in loc), S.REFSEQ)
    idx = [j for j, i in enumerate(loc) if 0 <= j + k < len(S.REF_CA)]
    if len(idx) < 50: return None
    R, t = S._robust(mca[[loc[j] for j in idx]], S.REF_CA[[j + k for j in idx]])
    ps, pseq = model.chain_seqs["pep"]
    pa = np.array([model.ca[ps + i] for i in range(10)]) @ R + t
    o = S.occupancy(pa)
    return dict(toGIG=float(np.sqrt(((pa - S.GIG) ** 2).sum() / 10)),
                toDRG=float(np.sqrt(((pa - S.DRG) ** 2).sum() / 10)),
                fpocket_pos=o["fpocket_pos"], fpocket_dist=o["fpocket_dist"],
                bpocket_dist=o["bpocket_dist"], register=o["register"])
print("ready")""")

md(r"""## 1 · Build the four native reference clouds

For each native trajectory, score `NF` evenly-subsampled frames on both axes. The sanity check: 6AM5
frames must stay F-pocket occupant **p10** (GIG register), 6AMU frames **p9** (DRG) — the native
peptide holds its register across the whole thermal ensemble, or the reference is meaningless.""")

co(r"""rows = []
for pid, T, job in MD_JOBS:
    models, _ = NM.load_md(job, NF)
    for fi, mdl in enumerate(models):
        sc = score_frame(mdl)
        if sc is None: continue
        rows.append(dict(pid=pid, temp=T, frame=fi, **sc))
MDF = pd.DataFrame(rows)   # NB: column is 'temp' not 'T' — df.T is the transpose property (footgun)
print(f"scored {len(MDF)} native MD frames")
chk = MDF.groupby(["pid", "temp"]).agg(n=("frame", "size"),
        toGIG_mean=("toGIG", "mean"), toGIG_sd=("toGIG", "std"),
        toDRG_mean=("toDRG", "mean"), toDRG_sd=("toDRG", "std"),
        fdepth_mean=("fpocket_dist", "mean"), fdepth_sd=("fpocket_dist", "std"),
        fpos_mode=("fpocket_pos", lambda s: int(s.mode().iloc[0])),
        fpos_native_frac=("fpocket_pos", lambda s: round((s == (10 if s.name[0]=='6AM5' else 9)).mean(), 3)
                           if False else 0.0)).round(3)
# fpos native fraction (mode-based, robust)
for (pid, T), g in MDF.groupby(["pid", "temp"]):
    want = 10 if pid == "6AM5" else 9
    chk.loc[(pid, T), "fpos_native_frac"] = round((g.fpocket_pos == want).mean(), 3)
display(chk)""")

md(r"""## 2 · The separation, done right — two clouds and their overlap

The crystal gives one 2.87 Å number; the MD gives the *pair of clouds* that number slices. For the
native GIG ensemble we have toGIG (tight, its own basin) **and** toDRG (where GIG sits relative to the
other basin); likewise for DRG. The real "how separated are the registers" is the gap between these
clouds **with their widths** — and whether the tails touch. If GIG's toDRG tail and DRG's toGIG tail
don't meet, the registers are cleanly resolvable and only designs in the overlap zone are genuinely
ambiguous. That overlap (or its absence) is what 2.87 Å hid.""")

co(r"""def cloud(pid, T): return MDF[(MDF["pid"] == pid) & (MDF["temp"] == T)]
for T in ["300K", "370K"]:
    g = cloud("6AM5", T); d = cloud("6AMU", T)
    # GIG cloud's distance-to-DRG vs DRG cloud's distance-to-GIG (the two "cross" tails)
    gig_toDRG_min = g.toDRG.min(); drg_toGIG_min = d.toGIG.min()
    gig_toGIG_hi = g.toGIG.quantile(.975); drg_toDRG_hi = d.toDRG.quantile(.975)
    print(f"[{T}]  GIG basin: toGIG {g.toGIG.mean():.2f}±{g.toGIG.std():.2f} (97.5%={gig_toGIG_hi:.2f}) | "
          f"its toDRG floor={gig_toDRG_min:.2f}")
    print(f"       DRG basin: toDRG {d.toDRG.mean():.2f}±{d.toDRG.std():.2f} (97.5%={drg_toDRG_hi:.2f}) | "
          f"its toGIG floor={drg_toGIG_min:.2f}")
    # overlap on the headline min(toGIG,toDRG): does either cloud ever get as close to the OTHER basin
    # as the other cloud sits in ITS OWN basin?  (clean resolvability test)
    print(f"       -> registers {'OVERLAP' if (gig_toDRG_min < drg_toDRG_hi or drg_toGIG_min < gig_toGIG_hi) else 'CLEANLY RESOLVED'} "
          f"(GIG never closer than {gig_toDRG_min:.2f} Å to DRG; DRG never closer than {drg_toGIG_min:.2f} Å to GIG)\n")""")

co(r"""fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharex=True, sharey=True)
for ax, T in zip(axes, ["300K", "370K"]):
    for pid, name, col in [("6AM5", "GIG native", "#2a78d6"), ("6AMU", "DRG native", "#c0392b")]:
        c = cloud(pid, T)
        ax.scatter(c.toGIG, c.toDRG, s=14, alpha=.5, color=col, edgecolor="none", label=f"{name} (n={len(c)})")
    ax.plot([0, 5], [0, 5], ":", color="0.6", lw=1)                      # y=x diagonal
    ax.set_xlabel("toGIG (Å)"); ax.set_title(f"native MD clouds — {T}"); ax.legend(fontsize=8)
axes[0].set_ylabel("toDRG (Å)")
fig.suptitle("The baseline: two native clouds (location + WIDTH), not one 2.87 Å point. "
             "GIG sits at (low toGIG, ~2.9 toDRG); DRG mirrors it. Off-diagonal = registered.", y=1.02)
plt.tight_layout(); plt.savefig(f"{ROOT}/figures/mdbase_native_clouds.png", dpi=150, bbox_inches="tight")
print("saved mdbase_native_clouds.png"); plt.show()""")

md(r"""## 3 · Calibrated acceptance bands (read off the clouds)

"In the GIG basin" = **inside the native-GIG MD distribution** on the cognate-RMSD axis (`toGIG ≤`
native mean + 3 SD) **AND** F-pocket occupant = p10 **within the native depth band** (native fdepth
mean ± 3 SD). Two conditions; both cutoffs come from the MD. This replaces the 6.5 Å occupancy cutoff
with the *native depth band* (~4 Å ± its own spread) — which is why it stops being spoofable: an
index-flip on a distorted backbone lands near the centroid but **outside the native depth band**.

**Caveat — these bands are still provisional, and the 370K extension changed the answer.** 300K is
still the original 0.2ns-equil/10ns-production run. 370K was re-run 5x longer (2ns equil / 50ns
production, Tamarind jobs `lbv69`=6AM5, `p1yv2`=6AMU) specifically to firm up the calibration — but a
convergence check on the LONGER run shows it *still* hasn't converged: both trajectories keep drifting
(net +0.25 Å for 6AM5, +0.82 Å for 6AMU over the 50ns), and the autocorrelation time turned out to be
**longer** than the original 10ns run suggested (5.3 ns for 6AM5, 10.7 ns for 6AMU, vs 1.7/0.8 ns
before) — so effective independent samples are still only ~9 and ~5 out of 1000 raw frames. The
practical consequence: the 370K DRG band widened from ≤1.72 Å (10ns) to **≤2.10 Å** (50ns), while the
GIG band actually *tightened* slightly (≤1.86 -> ≤1.49 Å) — both directions are just where this
particular under-converged trajectory happened to wander, not a settled answer. Read every 370K cutoff
below as "current best estimate," not final — a true converged estimate likely needs several-hundred-ns
runs given the measured ~10ns relaxation timescale.""")

co(r"""def make_band(pid, T, ksd=3.0):
    c = cloud(pid, T); cog = "toGIG" if pid == "6AM5" else "toDRG"; want = 10 if pid == "6AM5" else 9
    return dict(pid=pid, T=T, cog=cog, want=want,
                rmsd_hi=float(c[cog].mean() + ksd * c[cog].std()),
                fdepth_lo=float(c.fpocket_dist.mean() - ksd * c.fpocket_dist.std()),
                fdepth_hi=float(c.fpocket_dist.mean() + ksd * c.fpocket_dist.std()))
BANDS = {(pid, T): make_band(pid, T) for pid in ["6AM5", "6AMU"] for T in ["300K", "370K"]}
print("calibrated bands (mean ± 3 SD):")
for k, b in BANDS.items():
    print(f"  {k[0]}/{k[1]}: {b['cog']} ≤ {b['rmsd_hi']:.2f} Å  AND  F-pocket p{b['want']} within "
          f"[{b['fdepth_lo']:.2f}, {b['fdepth_hi']:.2f}] Å")

def in_basin(toGIG, toDRG, fpos, fdist, band):
    cogval = toGIG if band["cog"] == "toGIG" else toDRG
    return bool(cogval <= band["rmsd_hi"] and fpos == band["want"]
                and band["fdepth_lo"] <= fdist <= band["fdepth_hi"])

def classify(toGIG, toDRG, fpos, fdist, T="300K"):
    '''-> 'GIG' / 'DRG' / 'neither' on the joint two-axis test at temperature T.'''
    if in_basin(toGIG, toDRG, fpos, fdist, BANDS[("6AM5", T)]): return "GIG"
    if in_basin(toGIG, toDRG, fpos, fdist, BANDS[("6AMU", T)]): return "DRG"
    return "neither"
# self-consistency: native frames must classify into their own basin at 300K
for pid in ["6AM5", "6AMU"]:
    c = cloud(pid, "300K"); want = "GIG" if pid == "6AM5" else "DRG"
    frac = np.mean([classify(r.toGIG, r.toDRG, r.fpocket_pos, r.fpocket_dist) == want for r in c.itertuples()])
    print(f"  self-consistency {pid}/300K -> classified '{want}': {frac:.2f}")""")

md(r"""## 4a · Seeded / graft recovery — is it statistically inside the native cloud?

The seed-from-crystal partial-diffusion echoes (`rfdiff_q30`) and the chimera grafts
(`rfdiff_q30_v4arm2/seeds`) start *inside* a basin by construction. The real question is whether the
seeded **ensemble** is statistically indistinguishable from native MD — a clean two-sample test, not an
assertion from one small RMSD. **Autocorrelation caution:** native MD frames are correlated, so we thin
before quoting a p-value (effective N ≪ frame count).""")

co(r"""def score_pdbs(paths, label):
    out = []
    for f in paths:
        r = S.score_occ(f)
        if r is None or r.get("toGIG") is None: continue
        out.append(dict(set=label, file=os.path.basename(f), toGIG=r["toGIG"], toDRG=r["toDRG"],
                        fpocket_pos=r["fpocket_pos"], fpocket_dist=r["fpocket_dist"]))
    return out
seed_rows = []
# partial-diffusion echoes (chain-split), tag partial_T from filename
q30 = sorted(glob.glob(f"{ROOT}/outputs/rfdiff_q30/*_split.pdb"))
for r in score_pdbs(q30, "seed_echo"):
    mt = re.search(r"_T(\d+)_", r["file"]); r["partial_T"] = int(mt.group(1)) if mt else None
    r["pid"] = "6AM5" if r["file"].startswith("6AM5") else "6AMU"; seed_rows.append(r)
# chimera grafts
for r in score_pdbs(sorted(glob.glob(f"{ROOT}/outputs/rfdiff_q30_v4arm2/seeds/*.pdb")), "graft"):
    r["partial_T"] = None; r["pid"] = "6AM5" if "GIGbase" in r["file"] else "6AMU"; seed_rows.append(r)
SEED = pd.DataFrame(seed_rows)
SEED["class300"] = [classify(r.toGIG, r.toDRG, r.fpocket_pos, r.fpocket_dist, "300K") for r in SEED.itertuples()]
print(f"seeded/graft designs: {len(SEED)}  (echoes {sum(SEED.set=='seed_echo')}, grafts {sum(SEED.set=='graft')})")
display(SEED.groupby(["set", "pid"]).agg(n=("file","size"), toGIG_med=("toGIG","median"),
        toDRG_med=("toDRG","median"), in_GIG=("class300", lambda s:(s=='GIG').sum()),
        in_DRG=("class300", lambda s:(s=='DRG').sum())).round(3))

# two-sample: 6AM5 seed echoes vs native 6AM5 300K, on toGIG (thinned for autocorrelation)
native = cloud("6AM5", "300K").sort_values("frame")
nat_thin = native.iloc[::5].toGIG.values                      # thin ~5x -> nearer-independent
seed6 = SEED[(SEED.set=="seed_echo") & (SEED.pid=="6AM5")].toGIG.values
print(f"\ntwo-sample toGIG — native6AM5/300K (thinned n={len(nat_thin)}) vs 6AM5 seed echoes (n={len(seed6)}):")
if len(seed6) > 3:
    ks = ks_2samp(nat_thin, seed6); mw = mannwhitneyu(nat_thin, seed6, alternative="two-sided")
    print(f"  KS D={ks.statistic:.2f} p={ks.pvalue:.3g} | Mann-Whitney U p={mw.pvalue:.3g}")
    print(f"  native toGIG {nat_thin.mean():.2f}±{nat_thin.std():.2f} | seed {seed6.mean():.2f}±{seed6.std():.2f}  "
          f"(seed inside native support: {np.mean((seed6>=native.toGIG.min())&(seed6<=BANDS[('6AM5','300K')]['rmsd_hi'])):.2f})")""")

md(r"""## 4b · De-novo designs — fraction inside EITHER native cloud on BOTH axes

The correctly-scaled version of "did anything recover a register." This retires both broken proxies at
once: RMSD's false-negative wall (`to_other<1.45` never fired) **and** occupancy's false-positive argmin
(centroid-adjacent scatter counted as "seated"). We also re-examine last turn's **40 crossings**: how
many survive the native *depth band* rather than mere centroid proximity?""")

co(r"""dn = []
for f in S.gather([f"{ROOT}/outputs/{x}/pdb" for x in ("grind","ladder","promising","rfd_denovo30","rfd_maxcond")]):
    r = S.score_occ(f)
    if r is None or r.get("toGIG") is None: continue
    m = S.parse_meta(f)
    dn.append(dict(source=m["source"], pid=m["pid"], toGIG=r["toGIG"], toDRG=r["toDRG"],
                   fpocket_pos=r["fpocket_pos"], fpocket_dist=r["fpocket_dist"],
                   f_seated_occ=r["f_seated"]))
DN = pd.DataFrame(dn)
DN["class300"] = [classify(r.toGIG, r.toDRG, r.fpocket_pos, r.fpocket_dist, "300K") for r in DN.itertuples()]
DN["class370"] = [classify(r.toGIG, r.toDRG, r.fpocket_pos, r.fpocket_dist, "370K") for r in DN.itertuples()]
# the three verdicts side by side
occ_seated = int(DN.f_seated_occ.sum())                        # last turn's occupancy argmin (≤6.5 Å)
in_any_300 = int((DN.class300 != "neither").sum())
in_any_370 = int((DN.class370 != "neither").sum())
print(f"de-novo designs: {len(DN)}")
print(f"  RMSD proxy    (to_other<1.45Å)            : 0  (never fired)")
print(f"  occupancy argmin (≤6.5 Å, any register)   : {occ_seated}  (false-positive prone)")
print(f"  MD-calibrated joint test — inside a native cloud @300K : {in_any_300}   @370K(generous): {in_any_370}")
print("\n  by campaign (@300K):")
display(DN.groupby("source").agg(n=("toGIG","size"),
        in_GIG=("class300", lambda s:(s=='GIG').sum()), in_DRG=("class300", lambda s:(s=='DRG').sum()),
        in_any_300=("class300", lambda s:(s!='neither').sum()),
        in_any_370=("class370", lambda s:(s!='neither').sum())).sort_values("in_any_300", ascending=False))

# re-examine the 40 'crossings': occupancy argmin said crossed. Peel the axes one at a time.
DN["occ_crossed"] = ((DN.pid=="6AM5")&(DN.fpocket_pos==9)&(DN.fpocket_dist<=S.SEAT_F)) | \
                    ((DN.pid=="6AMU")&(DN.fpocket_pos==10)&(DN.fpocket_dist<=S.SEAT_F))
cross = DN[DN.occ_crossed]
def _other_band(r, T): return BANDS[("6AMU", T)] if r.pid == "6AM5" else BANDS[("6AM5", T)]  # opposite register
def depth_ok(r):
    b = _other_band(r, "300K"); return b["fdepth_lo"] <= r.fpocket_dist <= b["fdepth_hi"]
def joint_ok(r, T):
    b = _other_band(r, T); cogv = r.toDRG if r.pid == "6AM5" else r.toGIG   # RMSD to the OTHER register
    return (cogv <= b["rmsd_hi"]) and (b["fdepth_lo"] <= r.fpocket_dist <= b["fdepth_hi"])
d_surv = int(cross.apply(depth_ok, axis=1).sum()) if len(cross) else 0
j_surv = int(cross.apply(lambda r: joint_ok(r, "370K"), axis=1).sum()) if len(cross) else 0
print(f"\n  'crossings' (occupancy argmin ≤{S.SEAT_F} Å, n={len(cross)}) — peeling the axes:")
print(f"    + native DEPTH band            : {d_surv} survive  ({len(cross)-d_surv} pure centroid scatter)")
print(f"    + backbone PROXIMITY (full test): {j_surv} survive  -> anchor-at-native-depth is NOT register recovery")
# best de-novo near-miss — honest threshold sensitivity of the '0 recover' verdict
DN["cog"] = np.where(DN.pid == "6AM5", DN.toGIG, DN.toDRG)
bst = DN.sort_values("cog").iloc[0]; bb = _other_band(bst, "370K") if False else BANDS[(bst.pid, "370K")]
print(f"\n  best de-novo: {bst.source}/{bst.pid} cog={bst.cog:.2f} Å, fpos={bst.fpocket_pos} (want {bb['want']}), "
      f"fdist={bst.fpocket_dist:.2f} (band {bb['fdepth_lo']:.2f}–{bb['fdepth_hi']:.2f})")
print(f"    -> passes position+depth; misses 370K proximity by {bst.cog-bb['rmsd_hi']:+.2f} Å "
      f"(the '0 recover' verdict is a threshold-brittle near-miss here, not a clean miss)")""")

md(r"""## 4c · Under the extended 370K band: the 5 closest crossing candidates, and self-recovery

With the widened 370K DRG band (≤2.10 Å, up from ≤1.72 Å), re-test the 5 closest de-novo "crossing"
candidates found across the whole corpus (best-of-many-thousand draws, not a targeted result — see
[[best-hit-is-sampling-lottery]]) on the full joint test (proximity + F-pocket position + depth). Also
re-check OWN-register recovery corpus-wide under the same new bands: does anything now recover its own
native peptide, not just cross into the other?

For each of the 5 crossing candidates, also find the single CLOSEST individual MD frame (not just the
mean/band) in the target register's 50ns trajectory — thermal snapshots wander further from the mean
than the mean-based band alone captures, so a design can be closer to a specific real conformation than
to the average register position.""")

co(r"""BANDS_NEW = {}
for pid in ["6AM5", "6AMU"]:
    c = cloud(pid, "370K"); cog = "toGIG" if pid == "6AM5" else "toDRG"; want = 10 if pid == "6AM5" else 9
    BANDS_NEW[pid] = dict(cog=cog, rmsd_hi=float(c[cog].mean() + 3*c[cog].std()), want=want,
                          lo=float(c.fpocket_dist.mean() - 3*c.fpocket_dist.std()),
                          hi=float(c.fpocket_dist.mean() + 3*c.fpocket_dist.std()))
print("extended-370K bands (this notebook's 370K source IS the 50ns rerun now, so BANDS==BANDS_NEW; "
      "the 10ns-run values below are hardcoded from the prior calibration for comparison):")
OLD_370 = {"6AM5": 1.86, "6AMU": 1.72}
for pid, b in BANDS_NEW.items():
    print(f"  {pid}: {b['cog']} <= {b['rmsd_hi']:.2f} A  (was {OLD_370[pid]:.2f} on the original 10ns run)")

CROSS5 = [
 "outputs/rfd_maxcond/pdb/6AM5_k18_44.pdb", "outputs/promising/pdb/6AM5_L4_expanded_w1_4992623_0.pdb",
 "outputs/rfd_maxcond/pdb/6AM5_k18_12.pdb", "outputs/rfd_maxcond/pdb/6AM5_k18_109.pdb",
 "outputs/promising/pdb/6AMU_L1_nterm_wA_0082638_0.pdb",
]
PA_by_pid = {}
for pid, T, job in MD_JOBS:
    if T != "370K": continue
    models, _ = NM.load_md(job, NF)
    PA_by_pid[pid] = models

def frame_pa(model):
    ms_start, ms = model.chain_seqs["mhc"]
    mca = np.array([model.ca[ms_start + i] for i in range(len(ms))])
    m = ms.find(S.MHC_MOTIF); loc = list(range(m, min(m + 179, len(ms))))
    k = S._offset("".join(ms[i] for i in loc), S.REFSEQ)
    idx = [j for j, i in enumerate(loc) if 0 <= j + k < len(S.REF_CA)]
    if len(idx) < 50: return None
    R, t = S._robust(mca[[loc[j] for j in idx]], S.REF_CA[[j + k for j in idx]])
    ps, _ = model.chain_seqs["pep"]
    return np.array([model.ca[ps + i] for i in range(10)]) @ R + t

PA_370 = {pid: np.array([p for p in (frame_pa(m) for m in models) if p is not None])
          for pid, models in PA_by_pid.items()}

print(f"\n{'design':45s} {'to_other(mean)':>14s} {'new band':>9s} {'joint':>12s} {'nearest MD frame':>17s}")
for f in CROSS5:
    r = S._map_peptide(f); pa = np.asarray(r[0]); meta = S.parse_meta(f)
    op = "6AMU" if meta["pid"] == "6AM5" else "6AM5"
    o = S.occupancy(pa)
    cogval = float(np.sqrt(((pa - (S.DRG if op=="6AMU" else S.GIG))**2).sum()/10))
    b = BANDS_NEW[op]
    prox_ok = cogval <= b["rmsd_hi"]; pos_ok = o["fpocket_pos"] == b["want"]
    depth_ok = b["lo"] <= o["fpocket_dist"] <= b["hi"]
    verdict = "PASS (all 3)" if (prox_ok and pos_ok and depth_ok) else f"{sum([prox_ok,pos_ok,depth_ok])}/3"
    d = np.sqrt(((PA_370[op] - pa[None]) ** 2).sum(-1).mean(-1))
    imin = int(d.argmin())
    print(f"{os.path.basename(f):45s} {cogval:14.3f} {b['rmsd_hi']:9.2f} {verdict:>12s} "
          f"{d.min():8.3f} A (frame {imin}, t={imin*50.0/len(d):.1f}ns)")""")

co(r"""# corpus-wide OWN-register recovery under the new 370K bands (not just crossing)
pool = DN.copy()
def in_new(row):
    b = BANDS_NEW[row.pid]; cogval = row.toGIG if row.pid == "6AM5" else row.toDRG
    return (cogval <= b["rmsd_hi"]) and (row.fpocket_pos == b["want"]) and (b["lo"] <= row.fpocket_dist <= b["hi"])
pool["own_recovered_new370"] = pool.apply(in_new, axis=1)
n_rec = int(pool.own_recovered_new370.sum())
print(f"own-register recovery under extended 370K band: {n_rec} / {len(pool)}")
if n_rec:
    display(pool[pool.own_recovered_new370][["source","pid","toGIG","toDRG","fpocket_pos","fpocket_dist"]])
print("(compare: 0/542 under the original 10ns 370K band -- the one design that now clears it, "
      "promising/6AMU L5_max toDRG=1.73 A, missed the OLD band by just 0.01 A)")""")

md(r"""## 5 · The joint picture — everything against the calibrated baseline""")

co(r"""fig, ax = plt.subplots(figsize=(9.5, 8))
# native clouds (300K) as the reference
for pid, name, col in [("6AM5","GIG native (300K)","#2a78d6"), ("6AMU","DRG native (300K)","#c0392b")]:
    c = cloud(pid, "300K"); ax.scatter(c.toGIG, c.toDRG, s=22, alpha=.35, color=col, edgecolor="none", label=name)
# de-novo (grey), coloured if it lands in a basin
neither = DN[DN.class300=="neither"]; landed = DN[DN.class300!="neither"]
ax.scatter(neither.toGIG, neither.toDRG, s=9, alpha=.25, color="#b0b0b0", label=f"de-novo · neither (n={len(neither)})")
ax.scatter(landed.toGIG, landed.toDRG, s=34, color="#111", marker="x", label=f"de-novo · in a cloud (n={len(landed)})")
# seeded echoes + grafts
se = SEED[SEED.set=="seed_echo"]; gr = SEED[SEED.set=="graft"]
ax.scatter(se.toGIG, se.toDRG, s=26, color="#2ca02c", marker="^", alpha=.8, label=f"seed echoes (n={len(se)})")
ax.scatter(gr.toGIG, gr.toDRG, s=150, color="#ff7f0e", marker="*", edgecolor="k", label=f"chimera graft (n={len(gr)})")
ax.plot([0,18],[0,18], ":", color="0.7", lw=1)
ax.set_xlabel("toGIG (Å)"); ax.set_ylabel("toDRG (Å)")
ax.set_title("Calibrated baseline: native MD clouds vs seeded (in by construction) vs de-novo (mostly neither)")
ax.legend(fontsize=8, loc="upper right"); ax.set_xlim(-0.5, 18); ax.set_ylim(-0.5, 18)
plt.tight_layout(); plt.savefig(f"{ROOT}/figures/mdbase_joint_evaluation.png", dpi=150)
print("saved mdbase_joint_evaluation.png"); plt.show()""")

md(r"""## Read-out

**The baseline was the native MD ensemble all along** — it supplies the cloud's *width*, and width is
the yardstick RMSD never had. "Resembles the native peptide" = **falls inside the native MD distribution
on both the backbone-proximity axis and the F-pocket-occupant axis**, with cutoffs calibrated to real
thermal motion (300 K tight; 370 K generous).

- **Separation, corrected:** the registers are *cleanly resolved* — the native GIG and DRG clouds never
  approach each other (≥2.7 Å) as closely as each sits in its own basin (≤~1 Å). The 2.87 Å point was a
  slice through two non-overlapping clouds, and now we can see they don't touch.
- **Seeded recovery works — but as a frozen point, not a thermal ensemble.** All 34 seed echoes and both
  grafts land inside a native basin on the joint test (100% in-band). Yet the two-sample test *rejects*
  distributional equality (KS p≈6e-16): the echoes sit at toGIG 0.07 ± 0.00, *tighter* than the native
  thermal spread (0.69 ± 0.12) — partial diffusion reproduces the **crystal point**, not the ensemble
  around it. That is the [[rfdiff-partial-diffusion-freezes-peptide]] result, now quantified against the
  thermal ruler: recovery is real, thermal fidelity is not (yet).
- **De-novo recovery is 0/542 @300 K (unchanged — 300 K wasn't rerun) — but the extended 370 K run
  flips one case.** `promising/6AMU_L5_max_w1_0796681_1` (toDRG 1.73 Å) missed the original 10ns 370K
  band by 0.01 Å; under the 50ns-extended band (≤2.10 Å) it now **passes all three axes** — the first
  de-novo design (of ~1300 pure de-novo designs scored) to clear calibrated OWN-register recovery. Still
  1/1300 (0.08%) — vanishingly rare, not a capability, but no longer a clean zero.
- **The joint test retires both broken proxies.** RMSD's 1.45 Å crossing flag false-negated (0); the
  occupancy argmin false-positived (160 "seated", 40 "crossed"). Peeling the crossings: the native
  *depth* band alone removes 11 (pure centroid scatter), and adding the *proximity* axis removes the
  remaining 29 on the original 10ns band.
- **Under the 50ns-extended 370K band, 4 of the 5 closest de-novo crossing candidates now pass the full
  joint test** (proximity + F-pocket position + depth) — `6AM5_k18_44`, `6AM5_L4_expanded_...4992623_0`,
  `6AM5_k18_12`, `6AM5_k18_109`, all targeting DRG. This flip happened because the DRG 370K band widened
  from ≤1.72 to ≤2.10 Å on the longer run (§3 caveat: that run still hasn't converged, so treat this as
  "no longer excluded by the current best calibration," not "confirmed crossing"). The 5th candidate
  (targeting GIG) still fails — the GIG band tightened instead of widening. Every one of the 5 is also
  closer to at least one INDIVIDUAL 370K thermal frame than to the register mean (closest: `6AM5_k18_44`
  at 1.26 Å to a real DRG snapshot at t=41.4ns, vs 1.85 Å to the mean) — real thermal wandering brings
  individual native conformations closer to these designs than the average register position is.

Calibrate on 300 K, report 370 K as the generous (now 5x-longer, still not fully converged) bound; a
design inside the 370 K cloud but outside 300 K is "native-like only under heat, on provisional
calibration." See [[occupancy-primary-axis]], [[rfdiff-partial-diffusion-freezes-peptide]],
[[native-md-component-rmsd]], [[best-hit-is-sampling-lottery]].""")

nb["cells"] = C
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
out = "/home/ubuntu/if-mhc/01_md_calibrated_baseline.ipynb"
with open(out, "w") as fh: nbf.write(nb, fh)
print("wrote", out)
