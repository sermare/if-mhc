#!/usr/bin/env bash
# allcond150 worker -- ONE short GPU job. Idempotent top-up: round-robins over every
# (crystal,cond) cell in jobs/allcond150_spec.tsv and generates de-novo 10-mer peptides
# (T=30, Complex_base) until it has produced BATCH new designs OR a soft deadline hits,
# whichever comes first. Counts existing designs on disk, so preemption/requeue and many
# concurrent workers all just converge toward each cell's target (mild overshoot only).
set -uo pipefail
ABS=/global/scratch/users/sergiomar10/if-mhc
SPEC="${SPEC:-$ABS/jobs/allcond150_spec.tsv}"
OUT="${OUT_DIR:-$ABS/outputs/allcond150}"; PDB="$OUT/pdb"; mkdir -p "$PDB"
REVERSE="${REVERSE:-0}"   # 1 = walk the spec bottom-to-top (fill from the other end, less clash)
BATCH="${BATCH:-30}"; SOFT_MIN="${SOFT_MIN:-42}"
JOB="${SLURM_JOB_ID:-local}"; ARR="${SLURM_ARRAY_TASK_ID:-0}"
DEADLINE=$(( SECONDS + SOFT_MIN*60 ))   # SECONDS is bash-elapsed since script start

source /clusterfs/nilah/sergio/miniconda3/etc/profile.d/conda.sh
conda activate SE3nv
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export DGLBACKEND=pytorch PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2
cd "$ABS/RFdiffusion"

log(){ echo "[$(date '+%F %T')] w${ARR}(job $JOB) $*"; }
count_cell(){ ls "$PDB/${1}_${2}_j"*.pdb 2>/dev/null | grep -v traj | wc -l | tr -d ' '; }

log "start on ${SLURM_JOB_PARTITION:-?} $(hostname); BATCH=$BATCH SOFT=${SOFT_MIN}m"
made=0
progress=1
while [ "$progress" = 1 ] && [ "$made" -lt "$BATCH" ] && [ "$SECONDS" -lt "$DEADLINE" ]; do
  progress=0
  while IFS=$'\t' read -r x cond style target contig hot; do
    [ -z "${x:-}" ] && continue
    [ "$target" -le 0 ] 2>/dev/null && continue
    [ "$made" -ge "$BATCH" ] && break
    [ "$SECONDS" -ge "$DEADLINE" ] && break
    have=$(count_cell "$x" "$cond"); have=${have:-0}
    [ "$have" -ge "$target" ] && continue
    progress=1
    prefix="$PDB/${x}_${cond}_j${JOB}_${ARR}_${made}"
    args=( inference.input_pdb="$ABS/inputs/focus_6am/${x}_trim.pdb"
           "contigmap.contigs=[$contig]"
           inference.num_designs=1 diffuser.T=30
           inference.ckpt_override_path=models/Complex_base_ckpt.pt
           inference.output_prefix="$prefix" )
    [ -n "$(echo "$hot" | tr -d '[:space:]')" ] && args+=( "ppi.hotspot_res=[$hot]" )
    log "gen ${x}_${cond} have=$have/$target (#$((made+1)))"
    if timeout 1200 python run_inference.py "${args[@]}" >"$prefix.log" 2>&1; then
      made=$((made+1))
    else
      log "WARN ${x}_${cond} failed (see $prefix.log)"
    fi
  done < <(if [ "$REVERSE" = 1 ]; then tac "$SPEC"; else cat "$SPEC"; fi)
done
log "done: produced $made new designs this job (progress=$progress, elapsed=${SECONDS}s)"
