#!/usr/bin/env python3
"""Generate jobs/allcond150_spec.tsv : the master top-up spec for the "even out to
150/crystal" campaign. One row per (crystal, condition) cell.

Columns (tab-separated):
  crystal  cond  style  target  contig  hotspots

`target` = number of NEW designs to generate on Savio = max(0, 150 - already_generated),
where `already_generated` is the count recorded in outputs/README.md (the pre-existing
corpus, which lives in paper_designs_by_condition.zip -- not loose on this filesystem).
This makes the campaign a top-up to a clean 150 6AM5 + 150 6AMU = 300 total per condition.

Contigs/hotspots are copied verbatim from the validated per-campaign spec files:
  jobs/all_conditionings.tsv (ladder + maxcond), jobs/maxcond_spec.tsv (k14),
  jobs/recover_spec.tsv (fix ladder), inputs/focus_6am/ablation_spec.tsv (mhc/tcr subsets).
All runs: de-novo 10-mer peptide, full T=30, Complex_base_ckpt, MHC/B2M/TCR held as context.
"""
TARGET = 150

# base structural context contig per crystal (chains A=MHC, B=B2M, D/E=TCR)
BASE = {
    "6AM5": "A1-180/0 B1-100/0 D1-115/0 E1-120/0",
    "6AMU": "A2-180/0 B0-99/0 D2-115/0 E4-120/0",
}
DENOVO = "10-10"  # fully de-novo 10-residue peptide, nothing templated

# max-contact hotspot lists (used by maxcond + fix ladders)
MAX5 = "A159,A7,A99,A63,A147,A66,A77,A70,A155,D30,E97,A143,A146,A167,A73,A156,A67,A80,A97,A84,E30,E96,A116,A152,A171,D50,E98,A5,A9,A59,A81,A45,A76,A114,A123,D31,E95"
MAXU = "E100,A159,A7,D30,A66,A77,A147,A167,A156,A63,A146,A155,E99,A99,A116,D93,A152,A97,A70,A9,A59,A171,E102,A73,A80,A81,A123,A143,D28,D91,D94,A5,A67,A114,E101,A150,E98,A45,A84,A163,D92,D95"

# fix-ladder C-chain templating fragments (residues of native peptide kept fixed)
FIX_C = {
    "fixall": "C1-10",
    "fix8":   "C1-4/2-2/C7-10",
    "fix6":   "C1-3/4-4/C8-10",
    "fix4":   "C1-2/6-6/C9-10",
    "fix2":   "1-1/C2-2/6-6/C9-9/1-1",
    "fix0":   "10-10",
}

# rows: cond, style, {6AM5:(hot,cur), 6AMU:(hot,cur)}, contig_suffix(optional per crystal)
rows = []

# ---- contact ladder (promising) ----
rows += [
 ("L1_nterm","promising",{"6AM5":("A159,A66,A70",82),"6AMU":("A159,A66,A70",71)}),
 ("L2_nterm_t1","promising",{"6AM5":("A159,A66,A70,E97",75),"6AMU":("A159,A66,A70,E100",46)}),
 ("L3_nterm_t2","promising",{"6AM5":("A159,A66,A70,E97,D30",101),"6AMU":("A159,A66,A70,E100,D30",67)}),
 ("L4_expanded","promising",{"6AM5":("A159,A66,A70,A9,A77,A80,E97,D30,E96",148),"6AMU":("A159,A66,A70,A9,A77,A80,E100,D30,E99",82)}),
 ("L5_max","promising",{"6AM5":("A159,A66,A70,A9,A77,A80,A116,A143,E97,D30,E96,E30",111),"6AMU":("A159,A66,A70,A9,A77,A80,A116,A143,E100,D30,E99,D93",87)}),
]

# ---- max-contact ladder (maxcond) ----
K18_5="A159,A7,A99,A63,A147,A66,A77,A70,A155,D30,E97,A143,A146,A167,A73,A156,A67,A80"
K18_U="E100,A159,A7,D30,A66,A77,A147,A167,A156,A63,A146,A155,E99,A99,A116,D93,A152,A97"
K24_5="A159,A7,A99,A63,A147,A66,A77,A70,A155,D30,E97,A143,A146,A167,A73,A156,A67,A80,A97,A84,E30,E96,A116,A152"
K24_U="E100,A159,A7,D30,A66,A77,A147,A167,A156,A63,A146,A155,E99,A99,A116,D93,A152,A97,A70,A9,A59,A171,E102,A73"
K14_5="A159,A7,A99,A63,A147,A66,A77,A70,A155,D30,E97,A143,A146,A167"
K14_U="E100,A159,A7,D30,A66,A77,A147,A167,A156,A63,A146,A155,E99,A99"
rows += [
 ("k14","maxcond",{"6AM5":(K14_5,1),"6AMU":(K14_U,0)}),
 ("k18","maxcond",{"6AM5":(K18_5,140),"6AMU":(K18_U,46)}),
 ("k24","maxcond",{"6AM5":(K24_5,26),"6AMU":(K24_U,26)}),
 ("max","maxcond",{"6AM5":(MAX5,26),"6AMU":(MAXU,26)}),
]

# ---- mhc/tcr hotspot subsets (ablation) ----
MHC="A9,A63,A66,A77,A80,A116,A143"
rows += [
 ("mhc","ablation",{"6AM5":(MHC,14),"6AMU":(MHC,16)}),
 ("tcr1","ablation",{"6AM5":("E97",12),"6AMU":("E100",12)}),
 ("tcr2","ablation",{"6AM5":("E97,D30",16),"6AMU":("E100,D30",16)}),
 ("mhc_tcr1","ablation",{"6AM5":(MHC+",E97",12),"6AMU":(MHC+",E100",10)}),
 ("mhc_tcr2","ablation",{"6AM5":(MHC+",E97,D30",14),"6AMU":(MHC+",E100,D30",12)}),
]

# ---- fix ladder (recover): C-chain templating, full max hotspots ----
# NOTE: fixall (contig ...C1-10) templates the ENTIRE peptide -> 0 de-novo residues, which
# RFdiffusion cannot diffuse ("Invalid shape in axis 0: 0"). It is degenerate (== native peptide),
# so it is dropped from the generation spec; the ladder endpoint is the crystal peptide itself.
FIXCUR={"fix8":(5,5),"fix6":(5,5),"fix4":(5,5),"fix2":(5,5),"fix0":(5,5)}
for cond,(c5,cu) in FIXCUR.items():
    rows.append((cond,"recover",{"6AM5":(MAX5,c5),"6AMU":(MAXU,cu)},FIX_C[cond]))

# ---- null baseline / negative control (null): NO hotspots, NO template ----
rows.append(("null0","null",{"6AM5":("",32),"6AMU":("",31)}))

out=[]
for r in rows:
    cond,style,perx = r[0],r[1],r[2]
    fixc = r[3] if len(r)>3 else None
    for x in ("6AM5","6AMU"):
        hot,cur = perx[x]
        tgt = max(0, TARGET-cur)
        suffix = fixc if fixc else DENOVO
        contig = f"{BASE[x]} {suffix}"
        out.append((x,cond,style,str(tgt),contig,hot))

with open("/global/scratch/users/sergiomar10/if-mhc/jobs/allcond150_spec.tsv","w") as f:
    for o in out:
        f.write("\t".join(o)+"\n")

tot=sum(int(o[3]) for o in out)
print(f"wrote {len(out)} cells, total NEW designs to generate = {tot}")
for o in out:
    print("\t".join(o[:4]))
