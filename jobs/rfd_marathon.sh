#!/usr/bin/env bash
# 10-HOUR RFD design marathon on best MD-frame seeds, cycling CONDITIONING SCHEMES x partial_T.
# Each round picks (scheme, T); loops all 60 seeds 3-concurrent; repeats until the 10h wall-clock limit.
# Robust: failures swallowed. Output name encodes scheme+T+round so results are scorable per conditioning.
# Env: DUR_S (36000=10h), PAR (3). Seeds: outputs/md_seeds/*.pdb.
set -uo pipefail
ABS=/home/ubuntu/if-mhc
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh; conda activate SE3nv
export LD_LIBRARY_PATH="$(cat $ABS/RFdiffusion/SE3NV_LDPATH.txt):${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2
DUR_S="${DUR_S:-36000}"; PAR="${PAR:-3}"
MAR=$ABS/outputs/rfd_marathon; mkdir -p "$MAR/logs"
[ -f "$MAR/START" ] || date +%s > "$MAR/START"
START=$(cat "$MAR/START"); END=$((START+DUR_S))
mapfile -t SEEDS < <(ls $ABS/outputs/md_seeds/*.pdb)
TS=(20 30 40 50)
# CONDITIONING SCHEMES: "name|hotspot_res|inpaint_str"  (empty hotspot => none; empty inpaint => none)
SCHEMES=(
  "fpocket|A77,A80,A84,A116,A123,A143|"                                  # F-pocket floor (original)
  "fpocket_arm|A77,A80,A84,A116,A123,A143,A146,A147|"                    # + a2 arm (145-152 shifts for DRG)
  "dual_AF|A7,A63,A66,A70,A99,A159,A171,A77,A116,A143|"                  # A-pocket (N-anchor) + F-pocket (both ends)
  "cterm_release|A77,A80,A84,A116,A123,A143|C6-10"                       # freeze p1-5, diffuse only C-term half
  "free||"                                                              # no hotspots (unconstrained peptide)
  "reg_res|A116,A146|"                                                   # the two register-differing rotamer residues
)
cd "$ABS/RFdiffusion"
echo "$(date) MARATHON(multi-cond) start, ${#SEEDS[@]} seeds, ${#SCHEMES[@]} schemes x ${#TS[@]} T, DUR=${DUR_S}s" >> "$MAR/marathon.log"
r=0
while [ "$(date +%s)" -lt "$END" ]; do
  spec=${SCHEMES[$(( r % ${#SCHEMES[@]} ))]}
  T=${TS[$(( (r / ${#SCHEMES[@]}) % ${#TS[@]} ))]}
  name_s=${spec%%|*}; rest=${spec#*|}; hot=${rest%%|*}; inp=${rest#*|}
  r=$((r+1))
  echo "$(date) round $r scheme=$name_s T=$T hot=[$hot] inpaint=[$inp]" >> "$MAR/marathon.log"
  i=0
  for pdb in "${SEEDS[@]}"; do
    [ "$(date +%s)" -ge "$END" ] && break
    name="$(basename "$pdb" .pdb)_${name_s}_T${T}_r${r}"
    [ -f "$MAR/${name}_0.pdb" ] && continue
    args=( inference.input_pdb="$pdb" inference.output_prefix="$MAR/${name}"
           inference.ckpt_override_path=models/Complex_base_ckpt.pt
           'contigmap.contigs=[A1-180/0 B1-100/0 C1-10/0]' 'contigmap.provide_seq=[0-279]'
           diffuser.partial_T="$T" inference.num_designs=1 )
    [ -n "$hot" ] && args+=( "ppi.hotspot_res=[$hot]" )
    [ -n "$inp" ] && args+=( "+contigmap.inpaint_str=[$inp]" )
    ( python run_inference.py "${args[@]}" > "$MAR/logs/${name}.log" 2>&1 ) || true &
    i=$((i+1)); [ $(( i % PAR )) -eq 0 ] && wait
  done
  wait
done
echo "$(date) MARATHON_DONE round=$r designs=$(ls $MAR/*.pdb 2>/dev/null|wc -l)" >> "$MAR/marathon.log"
touch "$MAR/DONE"
