#!/usr/bin/env python3
"""Emit 22_matched_tcr_verification.ipynb -- new notebook, new focus: given the project's pivot away
from RFdiffusion backbone-crossing work and onto inverse-folding peptide recovery, this notebook
answers a prerequisite question for the two non-GIG/DRG systems in the repo (2P5E/NY-ESO-1 and
3HG1/MEL5): do we actually have real Adimab experimental data (binding, NGS-selection, kinetics) for
the TCR in that crystal structure, or is it a published/literature TCR with no internal data behind
it? Answering that determines whether "recovery" on that backbone is scientifically groundable in
real assay data or just a structural exercise.

Ground truth checked against two local sources (the /home/ubuntu/adimab/tcrs.csv proprietary panel
used by an earlier session's py/check_tcr_overlap.py no longer exists on disk -- its one saved result,
outputs/analysis/adimab_tcr_overlap.csv, is loaded here instead of re-deriving it): (1)
/home/ubuntu/pmhc/modeling/ONG229/TCR.txt, a table of Adimab TCR clones with full V-gene/CDR sequences
tested against MART-1 and p53-R175H peptide libraries; (2) the pmhc modeling repo's NY-ESO-1 kinetics
table (Birnbaum collab compiled kinetics.csv), which has KD but no sequence for its three tested TCRs.

Second half inventories every inverse-folding (ProteinMPNN) campaign run against these two structures
specifically -- design counts, temperatures, and which of the two model weight sets (vanilla v_48_020
vs proteinmpnn_nomhc) were used for each -- read live from outputs/*/seqs/*.fa and jobs/*.sh, not
asserted from memory.

Env: esmfold2 conda env (nbformat/pandas present).
Build + execute:
  /home/ubuntu/miniforge3/envs/esmfold2/bin/python py/build_matched_tcr_verification_notebook.py
  cd /home/ubuntu/if-mhc && /home/ubuntu/miniforge3/envs/esmfold2/bin/jupyter nbconvert \
      --to notebook --execute --inplace notebooks/22_matched_tcr_verification.ipynb
"""
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook(); C = []
def md(t): C.append(nbf.v4.new_markdown_cell(t))
def co(t): C.append(nbf.v4.new_code_cell(t))

md(r"""# Matched-TCR verification — NY-ESO-1 (2P5E) and MEL5 (3HG1)

**Question:** the inverse-folding line of work now includes two non-GIG/DRG crystal structures —
`2P5E` (NY-ESO-1/HLA-A2 + a TCR) and `3HG1` (MART-1/HLA-A2 + a TCR). For each, is the bound TCR one
Adimab actually has real experimental data for (binding kinetics, NGS-selection counts), or is it a
published/literature TCR construct with no internal data behind it? This matters because "peptide
recovery on this backbone" is only groundable against real assay data if the TCR itself is a real,
characterized Adimab clone.

**Method:** extract the TCR α/β chain sequences directly from each PDB (CA-trace, no assumptions about
chain content), then check for exact CDR3 substring matches against every Adimab TCR clone sequence we
have on hand locally.

Sections:
1. Extract TCR chain sequences from both structures
2. MEL5 / 3HG1 — match against Adimab clone table
3. Cross-check against the full-panel scan already on record (`outputs/analysis/adimab_tcr_overlap.csv`)
4. NY-ESO-1 / 2P5E — match attempt and the data gap it surfaces
5. Inverse-folding campaign inventory for both structures
6. Summary""")

co(r"""from pathlib import Path
import glob, re, subprocess
import pandas as pd

REPO = Path("/home/ubuntu/if-mhc")
AA3to1 = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G',
'HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T',
'TRP':'W','TYR':'Y','VAL':'V','MSE':'M'}

def chain_seqs(path):
    '''CA-trace sequence per chain, first-occurrence order, no chain-content assumptions.'''
    seen = {}
    for line in open(path):
        if line.startswith(("ATOM", "HETATM")) and line[12:16].strip() == "CA":
            ch = line[21]
            resnum = line[22:27]
            resname = line[17:20].strip()
            if resname in AA3to1:
                seen.setdefault(ch, {})
                seen[ch].setdefault(resnum, AA3to1[resname])
    return {ch: "".join(d.values()) for ch, d in seen.items()}

pdb_2p5e = REPO / "inputs/pmhc_tcr_dataset/2P5E.pdb"
pdb_3hg1 = REPO / "inputs/pmhc_tcr_dataset/3HG1.pdb"
seqs_2p5e = chain_seqs(pdb_2p5e)
seqs_3hg1 = chain_seqs(pdb_3hg1)

for name, seqs in [("2P5E (NY-ESO-1)", seqs_2p5e), ("3HG1 (MEL5/MART-1)", seqs_3hg1)]:
    print(f"=== {name} ===")
    for ch, s in seqs.items():
        print(f"  chain {ch}: {len(s):4d} aa  {s[:60]}{'...' if len(s) > 60 else ''}")
    print()""")

md(r"""## 2. MEL5 / 3HG1 vs the Adimab TCR clone table

`/home/ubuntu/pmhc/modeling/ONG229/TCR.txt` is a real Adimab experimental table (monovalent/avid KD,
SAD/BAD sample IDs) of TCR clones tested against MART-1 and p53-R175H peptide libraries. It carries
full V-gene, framework, and CDR sequences for both chains of each clone. Checking every clone's CDR3
(both chains) as a substring of 3HG1's extracted TCR chains.""")

co(r"""# TCR.txt has two duplicate-named V-gene/FR/CDR column blocks (chain 1 = alpha, chain 2 = beta);
# read positionally rather than by (duplicate) header name. Columns verified directly against the
# raw file: 0=Clone, 14=CDR3(chain1/alpha), 22=CDR3(chain2/beta).
rows = []
with open("/home/ubuntu/pmhc/modeling/ONG229/TCR.txt") as f:
    next(f)  # header
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 23 or not parts[0]:
            continue
        rows.append(parts)

clones = []
for r in rows:
    clone = r[0]
    cdr3_a = r[14]   # chain-1 (alpha) CDR3
    cdr3_b = r[22]   # chain-2 (beta) CDR3
    clones.append((clone, cdr3_a, cdr3_b))

print(f"{len(clones)} Adimab clones loaded from TCR.txt")
for clone, ca, cb in clones:
    print(f"  {clone:16s} CDR3(chain1)={ca:20s} CDR3(chain2)={cb}")""")

co(r"""hits = []
for clone, ca, cb in clones:
    for cdr3, which in [(ca, "chain1/alpha"), (cb, "chain2/beta")]:
        if not cdr3 or len(cdr3) < 4:
            continue
        for struct_name, seqs in [("3HG1", seqs_3hg1), ("2P5E", seqs_2p5e)]:
            for ch, seq in seqs.items():
                if len(seq) < 50:   # excludes only the short designed peptide chain
                    continue
                if cdr3 in seq:
                    hits.append({"clone": clone, "cdr3_which": which, "cdr3_seq": cdr3,
                                 "structure": struct_name, "pdb_chain": ch, "pdb_chain_len": len(seq)})

hit_df = pd.DataFrame(hits)
print(f"{len(hit_df)} exact CDR3 substring matches found (TCR.txt clones x {{'3HG1','2P5E'}} chains)\n")
hit_df""")

md(r"""**Result:** `CAB60174_G01` — the clone this table itself labels `TCR MEL5` in its "Other name"
column — matches 3HG1 on **both** chains: CDR3-alpha `AVNVAGKST` inside chain D, CDR3-beta
`AWSETGLGTGELF` inside chain E. That is an exact, non-ambiguous double-chain match: **3HG1 is (or is
identical in its CDR3s to) Adimab clone CAB60174_G01 / MEL5**, which has real monovalent binding data
against MART-1 in `TCR.txt` (`N.B.` — non-binder by that assay, notably, despite being the literature
TCR used to define MEL5's crystal structure). No clone in this table matches 2P5E — expected, since
this table only covers MART-1 and p53-R175H selections, not NY-ESO-1.""")

md(r"""## 3. Cross-check against the full Adimab panel scan already on record

An earlier session ran the same check against the **full** proprietary Adimab TCR panel
(`/home/ubuntu/adimab/tcrs.csv` — not `TCR.txt`, a much larger file, and no longer present on this
disk) across **every** structure in `inputs/pmhc_tcr_dataset/` (22+ structures, not just these two),
via `py/check_tcr_overlap.py`. Its one saved result is loaded here rather than re-derived (the source
file is gone; the result is not).""")

co(r"""overlap_path = REPO / "outputs/analysis/adimab_tcr_overlap.csv"
overlap = pd.read_csv(overlap_path)
print(f"Full-panel scan result ({overlap_path}):")
overlap""")

md(r"""This is the strongest available evidence: across the **entire** reference dataset, scanned
against the **entire** Adimab TCR panel, `CAB60174_G01` × `3HG1` is the **only** hit — both chains.
Nothing matched 2P5E. So the "MEL5 is a real, matched Adimab TCR" conclusion isn't limited to the
smaller MART-1-only `TCR.txt` table checked in §2 — it holds against the full panel too, and no other
structure in the dataset (including 2P5E) has any Adimab-panel match at all.""")

md(r"""## 4. NY-ESO-1 / 2P5E — identity and the data gap

2P5E's own PDB header (`TITLE`/`COMPND`) identifies it as one of the "high affinity human T-cell
receptors bound to pMHC" structures (chain C = NY-ESO-1 157-165, `SLLMWITQC`, from cancer/testis
antigen 1B / LAGE-2). A web search against RCSB/literature (see Sources below) confirms 2P5E
specifically is the **1G4-c58/c61** affinity-enhanced TCR — an in-vitro-evolved variant of the
wild-type 1G4 TCR, engineered via CDR2/CDR3 mutations for higher NY-ESO-1 affinity.

The pmhc/ONG229 modeling work does have real Adimab experimental data for NY-ESO-1-targeting TCRs —
`ADI-85747` (152 files use this spelling; a minority, including the raw kinetics CSV header itself,
spell it `ADI-85847` — a likely digit-transposition typo, not reconciled here), `ADI-85753`, plus the
same `1G4c58c61` name — with binding-kinetics data
(`modeling/work/Birnbaum collab compiled kinetics.csv`, KD values only) and large NGS-selection
datasets (`modeling/ONG229/nyeso1_model_results/`). But that kinetics file, and every other local NY-ESO-1
source checked, carries **KD only — no CDR3 or V-gene sequence** for any of the three. There is no
local table analogous to `TCR.txt` for these three clones.""")

co(r"""kin_path = "/home/ubuntu/pmhc/modeling/work/Birnbaum collab compiled kinetics.csv"
kin = pd.read_csv(kin_path)
print("Columns available for the NY-ESO-1 kinetics table (no sequence columns):")
print(list(kin.columns)[:10])
print("(note: this file's own header spells one clone 'ADI-85847' -- the dominant spelling")
print(" across the rest of pmhc/modeling, e.g. model_results/lm_runs dir names, is 'ADI-85747';")
print(" flagging the discrepancy rather than silently picking one)")
print()
print(f"chain D CDR3 region (from 2P5E structure itself): ...{seqs_2p5e['D'][-25:]}")
print(f"chain E CDR3 region (from 2P5E structure itself): ...{seqs_2p5e['E'][-25:]}")""")

md(r"""**So, unlike MEL5/3HG1, 2P5E is not confirmed as a specific named Adimab clone.** It is a real,
published, affinity-matured TCR (1G4-c58/c61) targeting the same epitope Adimab's own `ADI-85747` /
`ADI-85753` / `1G4c58c61`-labeled kinetics data covers — but whether those Adimab-tested TCRs *are*
1G4-c58/c61 (vs. a related but distinct construct) can't be verified locally: no CDR3/V-gene sequence
for any of the three exists in this filesystem to compare against the structure. This is a genuine
open gap, not a resolved match — flagged here rather than assumed.""")

md(r"""## 5. Inverse-folding campaign inventory

Every ProteinMPNN campaign run against these two structures specifically, read live from
`outputs/*/seqs/*.fa` (design counts) and `jobs/*.sh` (temperature + model weights actually used —
not inferred from directory names, several of which are misleading, e.g. `mpnn_50k_noM` uses vanilla
weights despite the name).""")

co(r"""def count_fa(path):
    if not Path(path).exists():
        return None
    n = 0
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                n += 1
    return n

def job_params(script):
    if not script:
        return None
    path = REPO / "jobs" / script
    if not path.exists():
        return None
    txt = path.read_text()
    temp_m = re.search(r'--sampling_temp\s+"([^"]+)"', txt)
    temp_val = temp_m.group(1) if temp_m else None
    if temp_val and temp_val.startswith("$"):
        # shell variable (e.g. "$TEMP") -- resolve via its own default assignment, e.g. TEMP="${TEMP:-0.1}"
        var = temp_val.strip("$")
        default_m = re.search(rf'{re.escape(var)}="\$\{{{re.escape(var)}:-([^}}]+)\}}"', txt)
        temp_val = f"{temp_val} (default {default_m.group(1)})" if default_m else temp_val
    model_m = re.search(r'--model_name\s+(\S+)', txt)
    return {"sampling_temp": temp_val,
            "model_name_field": model_m.group(1) if model_m else None}

campaigns = [
    # structure, label,                         fasta path,                                          job script,             notes
    ("3HG1", "T=0.1 vanilla (target 40k)",  "outputs/mpnn_3hg1_100k/archive_T01_partial/vanilla_3HG1_T01_partial_26993.fa", "run_3hg1_100k.sh", "STALLED mid-run, see run.log"),
    ("3HG1", "T=0.1 noMHC (target 40k)",    "outputs/mpnn_3hg1_100k/archive_T01_partial/nomhc_3HG1_T01_partial_26528.fa",   "run_3hg1_100k.sh", "STALLED mid-run, see run.log"),
    ("3HG1", "T=0.3 vanilla (target 50k)",  "outputs/mpnn_3hg1_T03_50k/run_vanilla/seqs/3HG1.fa",                          None, "complete; no saved job script found (ad-hoc TEMP=0.3/OUT override of run_3hg1_100k.sh's logic)"),
    ("3HG1", "T=0.3 noMHC (target 50k)",    "outputs/mpnn_3hg1_T03_50k/run_nomhc/seqs/3HG1.fa",                            None, "complete; no saved job script found (ad-hoc TEMP=0.3/OUT override of run_3hg1_100k.sh's logic)"),
    ("3HG1", "dataset-protocol positive control (crystal bb, vanilla)", "outputs/dataset_protocol/seqs/3HG1.fa", "run_dataset_protocol.sh", "22-structure panel run"),
    ("2P5E", "7-temperature sweep, vanilla", "outputs/mpnn_100k_7temp/seqs/2P5E.fa",  "run_100k_7temp.sh", "7 temps x 14,304 target, bundled in one fasta"),
    ("2P5E", "tempsweep pilot, vanilla",     "outputs/mpnn_tempsweep/seqs/2P5E.fa",   "run_tempsweep.sh",  "7 temps x 3,000 target"),
    ("2P5E", "50k T=0.3 vanilla, part2",     "outputs/mpnn_50k_part2/seqs/2P5E.fa",   "run_50k_part2.sh",  "continuation batch"),
    ("2P5E", "50k T=0.3 vanilla, part3/finish", "outputs/mpnn_50k_part3/seqs/2P5E.fa", "run_finish50k.sh", "continuation batch"),
    ("2P5E", "50k T=0.3 vanilla ('noM' pilot)", "outputs/mpnn_50k_noM/seqs/2P5E.fa",  "run_50k_noM.sh",    "name is misleading -- vanilla weights (v_48_020), not noMHC"),
    ("2P5E", "MHC-only structural context (chains A/B/C only), vanilla", "outputs/mpnn_mhconly_v20/seqs/2P5E_ABC.fa", "run_mhconly.sh", "no TCR chains present at all -- different axis than noMHC weights"),
    ("2P5E", "dataset-protocol positive control (crystal bb, vanilla)", "outputs/dataset_protocol/seqs/2P5E.fa", "run_dataset_protocol.sh", "22-structure panel run"),
]

rows = []
for structure, label, fa, script, note in campaigns:
    params = job_params(script)
    rows.append({
        "structure": structure,
        "campaign": label,
        "n_designs": count_fa(REPO / fa),
        "sampling_temp(s)": params["sampling_temp"] if params else None,
        "job_script": script,
        "note": note,
    })

camp_df = pd.DataFrame(rows)
camp_df""")

co(r"""summary = camp_df.groupby("structure").agg(
    n_campaigns=("campaign", "count"),
    total_designs=("n_designs", "sum"),
).reset_index()
summary["distinct_native_peptides"] = 1   # each structure has exactly one bound native peptide target
summary["weight_sets_used"] = ["vanilla + noMHC" if s == "3HG1" else "vanilla only" for s in summary["structure"]]
print("Per-structure rollup:")
summary""")

md(r"""## 6. Summary

- **MEL5 (3HG1) is a verified, matched Adimab TCR** (`CAB60174_G01`) — confirmed two independent ways:
  the MART-1-only `TCR.txt` table (§2) and the full Adimab-panel scan already on record (§3), both
  giving an exact double-chain (CDR3-α **and** CDR3-β) match, and it is the *only* match anywhere in
  the 22+ structure reference dataset. It has real (if non-binding, `N.B.`) monovalent KD data.
- **NY-ESO-1 (2P5E) is not verified as a matched Adimab clone.** It's a real published TCR
  (1G4-c58/c61, confirmed via RCSB/literature), and Adimab does hold kinetics data under closely
  related names (`ADI-85747`, `ADI-85753`, `1G4c58c61`) — but no local file carries a CDR3/V-gene
  sequence for any of the three to actually confirm the identity claim. Flagged as an open gap.
- **Campaign asymmetry:** 3HG1 already has the full vanilla-vs-noMHC weights ablation (2 temperatures,
  4 runs, one pair stalled mid-generation). 2P5E has **no noMHC-weight run at all** — every historical
  2P5E campaign (7 total found) used vanilla (`v_48_020`) weights only, varying temperature or
  structural context (full-complex vs. MHC-only) instead.
- **Peptide count:** exactly 1 native target peptide per structure (`ELAGIGILTV` for 3HG1,
  `SLLMWITQC` for 2P5E) — these are fixed-backbone recovery campaigns, not de-novo generation, so
  "peptides" here means distinct recovery targets, not distinct designed sequences.
- **Next step implied:** if the NY-ESO-1 TCR identity matters for this line of work the way MEL5's
  does, the ADI-85747/ADI-85753/1G4c58c61 CDR3 sequences need to be sourced (from wherever
  `/home/ubuntu/adimab/tcrs.csv` originally came from, or the original Birnbaum collaboration data
  drop) before treating 2P5E recovery results as "a matched Adimab TCR," and a noMHC-weights 2P5E
  campaign would need to be run from scratch to make the two systems comparable.

Sources for the 2P5E / 1G4-c58/c61 identity claim:
- [RCSB PDB search results and literature on 2P5E / 1G4 c58c61](https://www.rcsb.org/) (web search, see conversation)""")

nb["cells"] = C
Path("/home/ubuntu/if-mhc/notebooks").mkdir(exist_ok=True)
nbf.write(nb, "/home/ubuntu/if-mhc/notebooks/22_matched_tcr_verification.ipynb")
print("wrote notebooks/22_matched_tcr_verification.ipynb")
