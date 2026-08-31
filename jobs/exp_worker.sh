#!/usr/bin/env bash
# Generalized experiment worker -- ONE short GPU job, idempotent top-up, spec-driven.
# Reads SPEC (env) rows: crystal  cond  style  target  contig  hotspots  extra  inpdb
#   - hotspots may be empty; extra ALWAYS carries diffuser.T=<n> (+ any overrides)
#   - inpdb empty -> inputs/focus_6am/{x}_trim.pdb
# Round-robins all cells, generating until BATCH new designs or the soft deadline.
# Counts existing designs in OUT_DIR/pdb so many workers converge on each cell's target.
set -uo pipefail
ABS=/global/scratch/users/sergiomar10/if-mhc
SPEC="${SPEC:?set SPEC}"
OUT="${OUT_DIR:?set OUT_DIR}"; PDB="$OUT/pdb"; mkdir -p "$PDB" "$OUT/logs"
REVERSE="${REVERSE:-0}"; BATCH="${BATCH:-24}"; SOFT_MIN="${SOFT_MIN:-42}"
JOB="${SLURM_JOB_ID:-local}"; ARR="${SLURM_ARRAY_TASK_ID:-0}"
DEADLINE=$(( SECONDS + SOFT_MIN*60 ))

source /clusterfs/nilah/sergio/miniconda3/etc/profile.d/conda.sh
conda activate SE3nv
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export DGLBACKEND=pytorch PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2
cd "$ABS/RFdiffusion"

log(){ echo "[$(date '+%F %T')] w${ARR}(job $JOB) $*"; }
count_cell(){ ls "$PDB/${1}_${2}_j"*.pdb 2>/dev/null | grep -v traj | wc -l | tr -d ' '; }

log "start on ${SLURM_JOB_PARTITION:-?} $(hostname); SPEC=$(basename "$SPEC") OUT=$(basename "$OUT") BATCH=$BATCH"
made=0; progress=1
while [ "$progress" = 1 ] && [ "$made" -lt "$BATCH" ] && [ "$SECONDS" -lt "$DEADLINE" ]; do
  progress=0
  while IFS=$'\t' read -r x cond style target contig hot extra inpdb ckpt; do
    [ -z "${x:-}" ] && continue
    [ "${target:-0}" -le 0 ] 2>/dev/null && continue
    [ "$made" -ge "$BATCH" ] && break
    [ "$SECONDS" -ge "$DEADLINE" ] && break
    have=$(count_cell "$x" "$cond"); have=${have:-0}
    [ "$have" -ge "$target" ] && continue
    progress=1
    ipdb="${inpdb:-}"; { [ -z "$ipdb" ] || [ "$ipdb" = "-" ]; } && ipdb="$ABS/inputs/focus_6am/${x}_trim.pdb"
    ck="${ckpt:-}"; [ -z "$(echo "$ck" | tr -d '[:space:]')" ] && ck="models/Complex_base_ckpt.pt"
    prefix="$PDB/${x}_${cond}_j${JOB}_${ARR}_${made}"
    args=( inference.input_pdb="$ipdb"
           "contigmap.contigs=[$contig]"
           inference.num_designs=1
           inference.ckpt_override_path="$ck"
           inference.output_prefix="$prefix" )
    [ -n "$(echo "${hot:-}"   | tr -d '[:space:]')" ] && args+=( "ppi.hotspot_res=[$hot]" )
    [ -n "$(echo "${extra:-}" | tr -d '[:space:]')" ] && args+=( ${extra} )
    log "gen ${x}_${cond} have=$have/$target (#$((made+1)))"
    if timeout 1500 python run_inference.py "${args[@]}" >"$prefix.log" 2>&1; then
      made=$((made+1))
    else
      log "WARN ${x}_${cond} failed (tail: $(tail -3 "$prefix.log" 2>/dev/null | tr '\n' ' '))"
    fi
  done < <(if [ "$REVERSE" = 1 ]; then tac "$SPEC"; else cat "$SPEC"; fi)
done
log "done: produced $made new designs (progress=$progress, elapsed=${SECONDS}s)"
