#!/usr/bin/env bash
# 8-HOUR RFD CROSSING campaign — the untried MOTIF-based levers to force the OTHER register:
#  (P1) C-term anchor motif-transfer: FIX the target register's C-terminal anchor (p{k}-10) as a rigid
#       motif and de-novo diffuse the N-terminal rest around it — forces register via the anchor, not the
#       seed. Cycled over fix-lengths p6-10/p8-10/p9-10 for BOTH registers (6AM5=GIG, 6AMU=DRG).
#  (P2) Chimera-seeded partial diffusion: seed from the grafted other-register backbone (0.78 Å) and
#       partial-diffuse — does RFD HOLD a valid other-register state or collapse it.
# Loops for DUR_S, 3 combos concurrent, num_designs each, robust. Output tag encodes protocol.
set -uo pipefail
ABS=/home/ubuntu/if-mhc
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh; conda activate SE3nv
export LD_LIBRARY_PATH="$(cat $ABS/RFdiffusion/SE3NV_LDPATH.txt):${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2
DUR_S="${DUR_S:-28800}"; NDES="${NDES:-5}"; FPHOT="A77,A80,A84,A116,A123,A143"
OUT=$ABS/outputs/rfd_cross; mkdir -p "$OUT/logs"
[ -f "$OUT/START" ] || date +%s > "$OUT/START"; START=$(cat "$OUT/START"); END=$((START+DUR_S))
I=$ABS/inputs/focus_6am
# NATIVE crystals, FULL COMPLEX (TCR kept). tag|input|contig(incl D,E)|provide_seq(A,B,D,E)|partial_T
# 6AM5: A180 B100 C10 D115 E120 -> receptor 0-279,290-524 ; 6AMU: A179 B100 C10 D114 E117 -> 0-278,289-519
G5="D1-115/0 E1-120"; GU="D2-115/0 E4-120"; PS5="0-279,290-524"; PSU="0-278,289-519"
G5H="A1-180/0 B1-100/0"; GUH="A2-180/0 B0-99/0"
COMBOS=(
 # partial diffusion of the WHOLE peptide, in the TCR complex, across partial_T (the crossing attempt WITH TCR)
 "GIG_pt10|$I/6AM5_trim.pdb|$G5H C1-10/0 $G5|$PS5|10"
 "GIG_pt20|$I/6AM5_trim.pdb|$G5H C1-10/0 $G5|$PS5|20"
 "GIG_pt30|$I/6AM5_trim.pdb|$G5H C1-10/0 $G5|$PS5|30"
 "GIG_pt40|$I/6AM5_trim.pdb|$G5H C1-10/0 $G5|$PS5|40"
 "DRG_pt10|$I/6AMU_trim.pdb|$GUH C1-10/0 $GU|$PSU|10"
 "DRG_pt20|$I/6AMU_trim.pdb|$GUH C1-10/0 $GU|$PSU|20"
 "DRG_pt30|$I/6AMU_trim.pdb|$GUH C1-10/0 $GU|$PSU|30"
 "DRG_pt40|$I/6AMU_trim.pdb|$GUH C1-10/0 $GU|$PSU|40"
 # C-term anchor motif-transfer (fix own C-term p6-10, de-novo N-term) in the TCR complex
 "GIGc6|$I/6AM5_trim.pdb|$G5H 5-5/C6-10/0 $G5|$PS5|"
 "DRGc6|$I/6AMU_trim.pdb|$GUH 5-5/C6-10/0 $GU|$PSU|"
)
cd "$ABS/RFdiffusion"
echo "$(date) CROSS start, ${#COMBOS[@]} protocols, DUR=${DUR_S}s NDES=$NDES" >> "$OUT/marathon.log"
run_combo(){
  local spec="$1" r="$2"
  local tag=${spec%%|*}; local rest=${spec#*|}; local inp=${rest%%|*}; rest=${rest#*|}
  local con=${rest%%|*}; rest=${rest#*|}; local ps=${rest%%|*}; local pt=${rest#*|}
  local args=( inference.input_pdb="$inp" inference.output_prefix="$OUT/${tag}_r${r}"
               inference.ckpt_override_path=models/Complex_base_ckpt.pt
               "contigmap.contigs=[$con]" "contigmap.provide_seq=[$ps]"
               "ppi.hotspot_res=[$FPHOT]" inference.num_designs="$NDES" )
  [ -n "$pt" ] && args+=( diffuser.partial_T="$pt" )
  ( python run_inference.py "${args[@]}" > "$OUT/logs/${tag}_r${r}.log" 2>&1 ) || true
}
export -f run_combo 2>/dev/null || true
r=0
while [ "$(date +%s)" -lt "$END" ]; do
  r=$((r+1)); n=${#COMBOS[@]}; i=0
  echo "$(date) round $r" >> "$OUT/marathon.log"
  while [ $i -lt $n ]; do
    [ "$(date +%s)" -ge "$END" ] && break
    run_combo "${COMBOS[$i]}" "$r" &
    i=$((i+1)); [ $(( i % 3 )) -eq 0 ] && wait
  done
  wait
done
echo "$(date) CROSS_DONE round=$r designs=$(ls $OUT/*.pdb 2>/dev/null|wc -l)" >> "$OUT/marathon.log"; touch "$OUT/DONE"
