#!/usr/bin/env bash
# ============================================================================
# evaluate.sh — re-score ALL designs and print the sampling-vs-conditioning view.
#
# USAGE:
#   bash jobs/evaluate.sh                 # score every campaign dir (default set)
#   bash jobs/evaluate.sh outputs/promising/pdb   # or restrict to given dirs
#
# Writes outputs/denovo_scores/{per_design,summary}.csv, then prints per
# (source x crystal x conditioning): N, best/median to_cognate & to_other,
# fraction seated in groove, and the hard wall check (# with to_other < 1.45).
# ============================================================================
set -uo pipefail
ABS=/home/ubuntu/if-mhc; cd "$ABS"
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh; conda activate SE3nv 2>/dev/null || true
python py/score_denovo_designs.py "$@" >/dev/null 2>&1 || python py/score_denovo_designs.py "$@"
python - <<'PY'
import pandas as pd, numpy as np
df = pd.read_csv('outputs/denovo_scores/per_design.csv')
df = df[df.pep_len==10] if 'pep_len' in df else df
g = df.groupby(['source','pid','context','cond'])
r = g.agg(N=('to_other','size'),
          best_toOther=('to_other','min'), med_toOther=('to_other','median'),
          best_toCog=('to_cognate','min'),
          seated=('fpocket_dist', lambda s:(s<=6.5).mean()),
          under145=('to_other', lambda s:(s<1.45).sum())).reset_index()
r=r.round(2).sort_values(['source','pid','best_toOther'])
pd.set_option('display.width',200); pd.set_option('display.max_rows',200)
print("=== per (source x crystal x context x conditioning) — de-novo (10-mer) ===")
print(r.to_string(index=False))
print(f"\nTOTAL designs scored: {len(df)}")
print(f"HARD WALL: designs with to_other < 1.45 A (genuine crossing): {(df.to_other<1.45).sum()} of {len(df)}")
best=df.loc[df.to_other.idxmin()]
print(f"closest-to-other overall: to_other={best.to_other:.2f}  {best.source}/{best.pid}/{best.cond}  {best.file}")
PY
