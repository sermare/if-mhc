#!/usr/bin/env bash
# PAIRED partial-diffusion worker. For each (crystal, partial_T) it runs the own- and cross-register arms
# each as ONE deterministic num_designs=N job. RFdiffusion's inference.deterministic=True seeds
# make_deterministic(i_des) BY DESIGN INDEX, so own-design-i and cross-design-i share the identical RNG
# draw and differ ONLY in ppi.hotspot_res -> a matched-pair barrier-crossing curve (the correct, powered
# design). Idempotent per (crystal,pt,arm): skips if the cell already holds N designs. Seed the SAME native
# crystal for both arms; contig keeps the full peptide (C1-10) as motif; diffuser.partial_T re-noises it.
set -uo pipefail
ABS=/global/scratch/users/sergiomar10/if-mhc
SPEC="${SPEC:?set SPEC}"; OUT="${OUT_DIR:?set OUT_DIR}"; PDB="$OUT/pdb"; mkdir -p "$PDB" "$OUT/logs"
N="${NPAIR:-24}"; SOFT_MIN="${SOFT_MIN:-45}"
JOB="${SLURM_JOB_ID:-local}"; ARR="${SLURM_ARRAY_TASK_ID:-0}"; DEADLINE=$(( SECONDS + SOFT_MIN*60 ))
source /clusterfs/nilah/sergio/miniconda3/etc/profile.d/conda.sh; conda activate SE3nv
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export DGLBACKEND=pytorch PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2
cd "$ABS/RFdiffusion"
log(){ echo "[$(date '+%F %T')] pw${ARR}($JOB) $*"; }
cnt(){ ls "$PDB/${1}_${2}_${3}"_*.pdb 2>/dev/null | grep -v traj | wc -l | tr -d ' '; }

# ranked hotspot lists (own = crystal's own; cross = the OTHER crystal's)
R5="A159,A7,A99,A63,A147,A66,A77,A70,A155,D30,E97,A143,A146,A167,A73,A156,A67,A80,A97,A84,E30,E96,A116,A152,A171,D50,E98,A5,A9,A59,A81,A45,A76,A114,A123,D31,E95"
RU="E100,A159,A7,D30,A66,A77,A147,A167,A156,A63,A146,A155,E99,A99,A116,D93,A152,A97,A70,A9,A59,A171,E102,A73,A80,A81,A123,A143,D28,D91,D94,A5,A67,A114,E101,A150,E98,A45,A84,A163,D92,D95"
own_hot(){ [ "$1" = 6AM5 ] && echo "$R5" || echo "$RU"; }
cross_hot(){ [ "$1" = 6AM5 ] && echo "$RU" || echo "$R5"; }
recp(){ [ "$1" = 6AM5 ] && echo "A1-180/0 B1-100/0 D1-115/0 E1-120/0" || echo "A2-180/0 B0-99/0 D2-115/0 E4-120/0"; }

while IFS=$'\t' read -r x pt; do
  [ -z "${x:-}" ] && continue
  [ "$SECONDS" -ge "$DEADLINE" ] && break
  for arm in own cross; do
    [ "$SECONDS" -ge "$DEADLINE" ] && break
    have=$(cnt "$x" "pd${pt}" "$arm"); [ "${have:-0}" -ge "$N" ] && continue
    hot=$([ "$arm" = own ] && own_hot "$x" || cross_hot "$x")
    prefix="$PDB/${x}_pd${pt}_${arm}"
    log "gen ${x} partial_T=${pt} ${arm} (have=$have/$N, deterministic seeds 0..$((N-1)))"
    timeout 2800 python run_inference.py \
      inference.input_pdb="$ABS/inputs/focus_6am/${x}_trim.pdb" \
      "contigmap.contigs=[$(recp "$x") C1-10]" \
      inference.num_designs="$N" inference.deterministic=True \
      diffuser.T=50 diffuser.partial_T="$pt" \
      inference.ckpt_override_path=models/Complex_base_ckpt.pt \
      "ppi.hotspot_res=[$hot]" \
      inference.output_prefix="$prefix" >"$prefix.log" 2>&1 \
      && log "done ${x}_pd${pt}_${arm}" || log "WARN ${x}_pd${pt}_${arm} ($(tail -2 "$prefix.log"|tr '\n' ' '))"
  done
done < "$SPEC"
log "worker done (elapsed ${SECONDS}s)"
