#!/usr/bin/env bash
# 10-HOUR parallel RFD crossing campaign v2. Seeds = 34 FULL-COMPLEX (TCR-kept) MD frames:
#   INT = barrier-top design-MD (swap≈0, no home basin) ; NAT = native-MD excursed toward other register.
# Uniform chains A(MHC180)/B(b2m100)/C(pep10)/D(TCRa110)/E(TCRb115). Explore MANY protocols in parallel:
#   partial_T {10,20,30,40,50} x hotspot {fpocket, free, dual_AF}. Peptide diffused inside the full TCR
#   complex. Loops for DUR_S, PAR concurrent. Robust. Output tag encodes T+hotspot so results are scorable.
set -uo pipefail
ABS=/home/ubuntu/if-mhc
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh; conda activate SE3nv
export LD_LIBRARY_PATH="$(cat $ABS/RFdiffusion/SE3NV_LDPATH.txt):${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2
DUR_S="${DUR_S:-36000}"; PAR="${PAR:-4}"
OUT=$ABS/outputs/rfd_cross2; mkdir -p "$OUT/logs"
[ -f "$OUT/START" ] || date +%s > "$OUT/START"; START=$(cat "$OUT/START"); END=$((START+DUR_S))
CONTIG="A1-180/0 B1-100/0 C1-10/0 D1-110/0 E1-115"; PS="0-279,290-514"
declare -A HOTS=( [fpocket]="A77,A80,A84,A116,A123,A143" [dual]="A7,A63,A66,A70,A99,A159,A171,A77,A116,A143" [free]="" )
mapfile -t SEEDS < <(ls $ABS/inputs/full_seeds/*.pdb)
COMBOS=(); for T in 10 20 30 40 50; do for H in fpocket free dual; do COMBOS+=("$T:$H"); done; done
cd "$ABS/RFdiffusion"
echo "$(date) CROSS2 start, ${#SEEDS[@]} seeds x ${#COMBOS[@]} (T,hot) combos, PAR=$PAR DUR=${DUR_S}s" >> "$OUT/marathon.log"
r=0
while [ "$(date +%s)" -lt "$END" ]; do
  combo=${COMBOS[$(( r % ${#COMBOS[@]} ))]}; T=${combo%%:*}; H=${combo##*:}; hot=${HOTS[$H]}; r=$((r+1))
  echo "$(date) round $r T=$T hot=$H" >> "$OUT/marathon.log"; i=0
  for s in "${SEEDS[@]}"; do
    [ "$(date +%s)" -ge "$END" ] && break
    nm="$(basename "$s" .pdb)_T${T}_${H}_r${r}"
    [ -f "$OUT/${nm}_0.pdb" ] && continue
    args=( inference.input_pdb="$s" inference.output_prefix="$OUT/${nm}"
           inference.ckpt_override_path=models/Complex_base_ckpt.pt
           "contigmap.contigs=[$CONTIG]" "contigmap.provide_seq=[$PS]"
           diffuser.partial_T="$T" inference.num_designs=1 )
    [ -n "$hot" ] && args+=( "ppi.hotspot_res=[$hot]" )
    ( python run_inference.py "${args[@]}" > "$OUT/logs/${nm}.log" 2>&1 ) || true &
    i=$((i+1)); [ $(( i % PAR )) -eq 0 ] && wait
  done
  wait
done
echo "$(date) CROSS2_DONE round=$r designs=$(ls $OUT/*.pdb 2>/dev/null|wc -l)" >> "$OUT/marathon.log"; touch "$OUT/DONE"
