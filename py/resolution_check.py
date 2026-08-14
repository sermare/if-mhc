#!/usr/bin/env python3
"""Does better crystal resolution give better native-peptide recovery?

Tested twice: on the panel's own 20 crystals (numbers taken from the printed
output of notebooks/panel/03_recovery_presentation.ipynb section 9) and on the
28 new SKEMPI complexes. Resolution is in Angstroms, so BETTER resolution is a
SMALLER number -- "better resolution gives better recovery" predicts a NEGATIVE
correlation with recovery.
"""
import glob, re
import numpy as np
import pandas as pd
from scipy import stats

ROOT = "/global/scratch/users/sergiomar10/if-mhc"

# ---- panel's 20 crystals, as printed by notebook 03 section 9 -------------
PANEL = """2P5W 2.20 174 0.329680
1QSF 2.80 72 0.594787
1QRN 2.80 137 0.418793
2BNR 1.90 123 0.387897
2GJ6 2.56 140 0.457824
2F53 2.10 129 0.381230
2F54 2.70 447 0.273420
3QDG 2.69 343 0.498626
3QEQ 2.59 417 0.300938
3QFJ 2.29 92 0.571253
3GSN 2.80 86 0.329182
1OGA 1.40 134 0.613975
3UTS 2.71 284 0.413224
5C0A 2.46 148 0.431548
5C0B 2.03 82 0.533568
5HHO 2.95 195 0.455282
5EU6 2.02 52 0.550910
2VLR 2.30 186 0.598080
4MJI 2.99 226 0.197542
5NME 2.94 592 0.308216"""
P = pd.DataFrame([l.split() for l in PANEL.strip().split("\n")],
                 columns=["pdb", "res", "uniq", "rec"])
for c in ["res", "uniq", "rec"]:
    P[c] = P[c].astype(float)


def report(name, res, rec, uniq=None, n_note=""):
    r, p = stats.pearsonr(res, rec)
    rho, prho = stats.spearmanr(res, rec)
    better = "BETTER resolution -> BETTER recovery" if r < 0 else "BETTER resolution -> WORSE recovery"
    sig = "significant" if p < 0.05 else "NOT significant"
    print(f"\n  {name} (n={len(res)}) {n_note}")
    print(f"    resolution vs recovery : Pearson r={r:+.3f} (p={p:.3g}), "
          f"Spearman rho={rho:+.3f} (p={prho:.3g})")
    print(f"      -> {better}, {sig}")
    if uniq is not None:
        r2, p2 = stats.pearsonr(res, uniq)
        print(f"    resolution vs #unique  : Pearson r={r2:+.3f} (p={p2:.3g})")
    return r, p


print("=" * 82)
print("A. THE PANEL'S OWN 20 CRYSTALS (numbers from notebook 03 section 9)")
print("=" * 82)
report("panel, all 20", P.res, P.rec, P.uniq)

# ---- resolution for the 28 SKEMPI complexes ------------------------------
rows = []
for f in sorted(glob.glob(f"{ROOT}/inputs/skempi/pdb_raw/*.pdb")):
    pdb = f.split("/")[-1][:-4]
    res = np.nan
    for line in open(f):
        if line.startswith("REMARK   2 RESOLUTION"):
            m = re.search(r"(\d+\.\d+)\s*ANGSTROM", line)
            if m:
                res = float(m.group(1))
            break
        if line.startswith("ATOM"):
            break
    rows.append({"pdb": pdb, "resolution_A": res})
RES = pd.DataFrame(rows)

summ = pd.read_csv(f"{ROOT}/designs/skempi/t01/summary.csv")
man = pd.read_csv(f"{ROOT}/inputs/skempi/manifest.csv")[["complex", "pdb", "pep_len"]]
cls = pd.read_csv(f"{ROOT}/outputs/skempi_if/mhc_class.csv")[["complex", "cls"]]

full = summ[summ.arm == "full"].groupby("complex").agg(
    rec=("mean_recovery", "mean"), uniq=("n_unique", "mean")).reset_index()
full = full.merge(man, on="complex").merge(RES, on="pdb").merge(cls, on="complex").dropna()

print("\n" + "=" * 82)
print("B. THE 28 NEW SKEMPI COMPLEXES")
print("=" * 82)
print(f"\n  resolution range {full.resolution_A.min():.2f}-{full.resolution_A.max():.2f} A")
report("SKEMPI, all", full.resolution_A, full.rec, full.uniq)
ci = full[full.cls == "I"]
report("SKEMPI, class I only", ci.resolution_A, ci.rec, ci.uniq)
c9 = full[(full.cls == "I") & (full.pep_len == 9)]
report("SKEMPI, class I 9-mers", c9.resolution_A, c9.rec, c9.uniq,
       "(tightest like-for-like with the panel)")

print("\n" + "=" * 82)
print("C. POOLED: panel + SKEMPI class-I 9/10mers (same regime)")
print("=" * 82)
c910 = full[(full.cls == "I") & (full.pep_len.isin([9, 10]))]
pool_res = np.concatenate([P.res.values, c910.resolution_A.values])
pool_rec = np.concatenate([P.rec.values, c910.rec.values])
report("pooled", pool_res, pool_rec)

full[["complex", "pdb", "cls", "pep_len", "resolution_A", "rec", "uniq"]].sort_values(
    "resolution_A").to_csv(f"{ROOT}/outputs/skempi_if/resolution_vs_recovery.csv", index=False)
print("\n  best/worst resolution in the new set:")
s = full.sort_values("resolution_A")
for _, r in pd.concat([s.head(3), s.tail(3)]).iterrows():
    print(f"    {r['complex']:14s} {r.resolution_A:.2f} A  recovery={r.rec:.3f}  class {r.cls}")
