# Paper-used designs, organized by conditioning

`outputs/` normally holds the full working set of RFdiffusion/MD outputs for this project
(~50GB across every exploratory campaign ever launched) and is not tracked in git. The one
exception is `paper_designs_by_condition.zip`, a curated subset containing **only the final
designed Cα backbones actually used in `MAIN_CLAIMS.md` / `paper/paper.pdf`** — no intermediate
diffusion trajectories, no `.trb` metadata, no exploratory campaigns that didn't make it into
the paper.

## Why "by conditioning," not "by campaign"

The raw campaign folders (`grind/`, `ladder/`, `promising/`, `rfd_denovo30/`, `rfd_maxcond/`,
`rfd_recover/`, `rfd_null/`) are organized by *which script launched the job*, not by *what the
job was conditioned on*. Several conditioning cells were incrementally topped up across multiple
campaigns launched at different times (see Methods §2.2 of the paper), so the campaign a file
lives in is an implementation detail, not the scientifically meaningful grouping. This zip
regroups every design by its actual `(conditioning, crystal)` identity instead.

## Contents

Unzipping produces one folder per conditioning cell, e.g. `condition_k18/`, `condition_L4_expanded/`,
`condition_fix0/`, `condition_null0/` — 21 folders in total, ~1,428 designs, 189MB uncompressed.
Filenames retain their crystal prefix (`6AM5_*` = conditioned toward/around the GIG crystal,
`6AMU_*` = DRG), so both crystals for a given conditioning coexist in the same folder.

| folder | what it is | used in |
|---|---|---|
| `condition_L1_nterm` … `condition_L5_max` | N-terminal contact ladder (3–12 contacts) | §3.2–3.4 corpus, Fig. 2/4 |
| `condition_k14`, `condition_k18`, `condition_k24`, `condition_max` | max-contact ladder (14–42 contacts) | §3.2–3.4 corpus, Fig. 2–4 |
| `condition_mhc`, `condition_tcr1`, `condition_tcr2`, `condition_mhc_tcr1`, `condition_mhc_tcr2` | MHC-only / TCR-only / combined hotspot subsets | §3.2–3.4 corpus |
| `condition_fixall` … `condition_fix0` | motif-templating ladder (10→0 residues templated) | §3.5, Table 6 |
| `condition_null0` | zero-hotspot, zero-template null baseline | §3.6 (still growing — see `jobs/run_null_baseline.sh`) |

### Design counts by condition and structure

| condition | 6AM5 (GIG) | 6AMU (DRG) | total |
|---|---:|---:|---:|
| `L4_expanded` | 148 | 82 | 230 |
| `L5_max` | 111 | 87 | 198 |
| `k18` | 140 | 46 | 186 |
| `L3_nterm_t2` | 101 | 67 | 168 |
| `L1_nterm` | 82 | 71 | 153 |
| `L2_nterm_t1` | 75 | 46 | 121 |
| `null0` | 32 | 31 | 63 |
| `k24` | 26 | 26 | 52 |
| `max` | 26 | 26 | 52 |
| `tcr2` | 16 | 16 | 32 |
| `mhc` | 14 | 16 | 30 |
| `mhc_tcr2` | 14 | 12 | 26 |
| `tcr1` | 12 | 12 | 24 |
| `mhc_tcr1` | 12 | 10 | 22 |
| `fixall` | 15 | 5 | 20 |
| `fix2` | 5 | 5 | 10 |
| `fix4` | 5 | 5 | 10 |
| `fix6` | 5 | 5 | 10 |
| `fix8` | 5 | 5 | 10 |
| `fix0` | 5 | 5 | 10 |
| `k14` | 1 | 0 | 1 |
| **total** | **850** | **578** | **1,428** |

`null0` will keep growing as `jobs/run_null_baseline.sh` continues (target 150/crystal); re-run
the regeneration steps below and re-zip to pick up new designs.

`condition_manifest.csv` (also tracked standalone, outside the zip) maps every file to its
conditioning, source campaign, crystal, and scored Cα RMSD/anchor values from
`py/score_denovo_designs.py`, so you don't need to unzip anything just to look up which design
is which.

## Regenerating this

```
python3 - <<'EOF'
import pandas as pd, os, shutil, glob
df = pd.read_csv('outputs/denovo_scores/per_design.csv')
df10 = df[df.pep_len == 10]
for _, row in df10.iterrows():
    dest_dir = f"outputs/condition_{row['cond']}"
    os.makedirs(dest_dir, exist_ok=True)
    shutil.copy2(row['file'], os.path.join(dest_dir, os.path.basename(row['file'])))
EOF
```

plus the `null0` designs from `outputs/rfd_null/pdb/*.pdb` (§2.5/§3.6 of the paper).
