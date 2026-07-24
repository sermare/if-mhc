#!/usr/bin/env python3
"""Generate spec TSVs for the reviewer-driven experiment campaign.
Columns: crystal  cond  style  target  contig  hotspots  extra  inpdb
  - contig: full RFdiffusion contig (receptor prefix + peptide segment)
  - hotspots: comma list for ppi.hotspot_res (may be empty)
  - extra: extra run_inference.py overrides, ALWAYS includes diffuser.T=<n>
  - inpdb: optional input pdb override (empty -> {x}_trim.pdb)
All reuse jobs/exp_worker.sh. One TSV per experiment family.
"""
import os
JOBS = "/global/scratch/users/sergiomar10/if-mhc/jobs"

# receptor contig prefix per crystal (peptide segment appended per-experiment)
RECP = {"6AM5": "A1-180/0 B1-100/0 D1-115/0 E1-120/0",
        "6AMU": "A2-180/0 B0-99/0 D2-115/0 E4-120/0"}

# contact-ranked hotspot lists (from the xover 'own' specs; MHC=A, b2m=B, TCRa=D, TCRb=E)
RANK = {
 "6AM5": "A159,A7,A99,A63,A147,A66,A77,A70,A155,D30,E97,A143,A146,A167,A73,A156,A67,A80,A97,A84,E30,E96,A116,A152,A171,D50,E98,A5,A9,A59,A81,A45,A76,A114,A123,D31,E95".split(","),
 "6AMU": "E100,A159,A7,D30,A66,A77,A147,A167,A156,A63,A146,A155,E99,A99,A116,D93,A152,A97,A70,A9,A59,A171,E102,A73,A80,A81,A123,A143,D28,D91,D94,A5,A67,A114,E101,A150,E98,A45,A84,A163,D92,D95".split(","),
}
# scrambled: receptor residues >15A from the peptide (surface-away), matched count
FAR = {
 "6AM5": "D115,D113,D114,D112,E116,B76,E119,E120,E117,B75,D111,E118,E115,E114,B77".split(","),
 "6AMU": "D115,D113,D114,D112,D111,B75,E119,B74,E120,D110,B76,B73,E117,E118,B42".split(","),
}
T30 = "diffuser.T=30"

def rich(x):            # full rich hotspot list (the 'max' condition)
    return ",".join(RANK[x])
def topk(x, k):         # first k of the ranked list
    return ",".join(RANK[x][:k])
def mhc_only(x, k):     # first k chain-A (MHC) residues
    return ",".join([r for r in RANK[x] if r[0] == "A"][:k])
def tcr_only(x, k):     # first k chain-D/E (TCR) residues
    return ",".join([r for r in RANK[x] if r[0] in "DE"][:k])

def write(name, rows):
    p = os.path.join(JOBS, name)
    with open(p, "w") as f:
        for r in rows:
            f.write("\t".join(str(c) for c in r) + "\n")
    print(f"wrote {p}  ({len(rows)} cells, {sum(int(r[3]) for r in rows)} designs)")

# ---------- E1 template-identity ladder (which residues are templated) ----------
# peptide segment contigs (chain C = 10-mer); hotspots held constant = rich
TI = {
 "ti_cterm2": "8-8/C9-10",     # C-terminal anchor P9-P10 templated
 "ti_cterm1": "9-9/C10-10",    # PΩ only
 "ti_nterm2": "C1-2/8-8",      # N-terminus P1-P2 templated
 "ti_nterm1": "C1-1/9-9",      # P1 only
 "ti_mid2":   "4-4/C5-6/4-4",  # middle P5-P6 templated
}
rows = []
for x in ("6AM5", "6AMU"):
    for cond, seg in TI.items():
        rows.append([x, cond, "tmplid", 150, f"{RECP[x]} {seg}", rich(x), T30, ""])
write("exp_tmplid_spec.tsv", rows)

# ---------- E2 hotspot-count sweep (in-distribution 1..15) ----------
rows = []
for x in ("6AM5", "6AMU"):
    for k in (1, 3, 5, 7, 9, 11, 13, 15):
        rows.append([x, f"hs{k:02d}", "hsweep", 120, f"{RECP[x]} 10-10", topk(x, k), T30, ""])
write("exp_hsweep_spec.tsv", rows)

# ---------- E3 scrambled-hotspot control (same count, wrong info) ----------
rows = []
for x in ("6AM5", "6AMU"):
    rows.append([x, "scr15", "scramble", 150, f"{RECP[x]} 10-10", ",".join(FAR[x]),      T30, ""])
    rows.append([x, "scr08", "scramble", 150, f"{RECP[x]} 10-10", ",".join(FAR[x][:8]),  T30, ""])
write("exp_scramble_spec.tsv", rows)

# ---------- E4 region ablation (fixed count 12, leave-one-region-out) ----------
rows = []
for x in ("6AM5", "6AMU"):
    rows.append([x, "ra_full12",  "region", 150, f"{RECP[x]} 10-10", topk(x, 12),      T30, ""])
    rows.append([x, "ra_mhc12",   "region", 150, f"{RECP[x]} 10-10", mhc_only(x, 12),  T30, ""])
    rows.append([x, "ra_tcronly", "region", 150, f"{RECP[x]} 10-10", tcr_only(x, 12),  T30, ""])
write("exp_region_spec.tsv", rows)

# ---------- E5 noise-off / T=50 ablation (best condition = rich 'max') ----------
rows = []
NOISE0 = "denoiser.noise_scale_ca=0 denoiser.noise_scale_frame=0"
for x in ("6AM5", "6AMU"):
    rows.append([x, "n0_T50", "noise", 150, f"{RECP[x]} 10-10", rich(x), f"diffuser.T=50 {NOISE0}", ""])
    rows.append([x, "n0_T30", "noise", 150, f"{RECP[x]} 10-10", rich(x), f"diffuser.T=30 {NOISE0}", ""])
    rows.append([x, "T50",    "noise", 150, f"{RECP[x]} 10-10", rich(x), "diffuser.T=50",           ""])
write("exp_noise_spec.tsv", rows)

# ---------- E6 partial diffusion (barrier crossing: seed native, denoise under own/cross) ----------
# contig keeps the full native peptide as motif (C1-10); partial_T re-noises it.
# own  = seed's own hotspots (control, should stay); cross = OTHER register's hotspots (redirect).
OTHER = {"6AM5": "6AMU", "6AMU": "6AM5"}
rows = []
for x in ("6AM5", "6AMU"):
    for pt in (5, 10, 15, 20):
        rows.append([x, f"pd{pt:02d}_own",   "partial", 100, f"{RECP[x]} C1-10", rich(x),
                     f"diffuser.T=50 diffuser.partial_T={pt}", ""])
        rows.append([x, f"pd{pt:02d}_cross", "partial", 100, f"{RECP[x]} C1-10", rich(OTHER[x]),
                     f"diffuser.T=50 diffuser.partial_T={pt}", ""])
write("exp_partial_spec.tsv", rows)
print("DONE")
