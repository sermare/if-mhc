#!/usr/bin/env bash
# noMHC ProteinMPNN, 50K sequences EACH, on 7 targets: native 6AM5, native 6AMU, and the 5 closest
# de-novo "crossing" designs found to date (to_other Å): 6AM5_k18_44 (1.85), 6AM5_L4_expanded_4992623_0
# (2.06), 6AM5_k18_12 (2.07), 6AM5_k18_109 (2.07), 6AMU_L1_nterm_0082638_0 (2.23). Same recipe as the
# existing noMHC campaign (nomhc_model_weights/proteinmpnn_nomhc, peptide chain designed, rest fixed),
# T=0.1 per instruction. Resumable per-target (skips any target whose .fa already has >=50000 seqs).
set -uo pipefail
cd /home/ubuntu/if-mhc; ABS=/home/ubuntu/if-mhc
PY=/home/ubuntu/miniforge3/envs/esmcba/bin/python
OUT="$ABS/outputs/mpnn_nomhc_topcross_50k"; STAGE="$OUT/pdb_in"; mkdir -p "$STAGE" "$OUT/seqs"
WEIGHTS=ProteinMPNN/nomhc_model_weights/; MODEL=proteinmpnn_nomhc
NSEQ="${NSEQ:-50000}"; BATCH="${BATCH:-8}"; TEMP="${TEMP:-0.1}"
LOG="$OUT/run.log"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
gpu_wait(){ while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -lt "$1" ]; do sleep 20; done; }
exec 9>"$OUT/runner.lock"; flock -n 9 || { echo "another runner holds lock — exit"; exit 0; }
echo $$ >"$OUT/runner.pid"; trap 'rm -f "$OUT/runner.pid"' EXIT

declare -A SRC=(
  [nat_6AM5]="$ABS/inputs/focus_6am/6AM5.pdb"
  [nat_6AMU]="$ABS/inputs/focus_6am/6AMU.pdb"
  [rfd_k18_44]="$ABS/outputs/rfd_maxcond/pdb/6AM5_k18_44.pdb"
  [rfd_L4_4992623]="$ABS/outputs/promising/pdb/6AM5_L4_expanded_w1_4992623_0.pdb"
  [rfd_k18_12]="$ABS/outputs/rfd_maxcond/pdb/6AM5_k18_12.pdb"
  [rfd_k18_109]="$ABS/outputs/rfd_maxcond/pdb/6AM5_k18_109.pdb"
  [rfd_L1_0082638]="$ABS/outputs/promising/pdb/6AMU_L1_nterm_wA_0082638_0.pdb"
)
for name in "${!SRC[@]}"; do
  [ -f "$STAGE/$name.pdb" ] || ln -s "${SRC[$name]}" "$STAGE/$name.pdb"
done

echo "[$(date '+%F %T')] === noMHC topcross50k: 7 targets x $NSEQ seqs, T=$TEMP ===" | tee -a "$LOG"

# parse once (cheap; re-parsing is harmless/idempotent)
$PY ProteinMPNN/helper_scripts/parse_multiple_chains.py \
   --input_path "$STAGE" --output_path "$OUT/parsed_chains.jsonl" >>"$LOG" 2>&1

$PY - <<PYEOF 2>>"$LOG"
import json
# self-contained peptide-chain detector (no core_load/prep_nomhc_allbb import -- those pull in
# pandas.read_parquet which needs pyarrow, not installed in the esmcba env this script runs under)
assigned = {}
for line in open("$OUT/parsed_chains.jsonl"):
    d = json.loads(line); name = d["name"]
    seqlens = {k.split("_")[-1]: len(d[k]) for k in d if k.startswith("seq_chain_")}
    des = next((c for c, n in seqlens.items() if n == 10), None)
    if des is None:
        raise SystemExit(f"no 10-residue peptide chain found for {name}: {seqlens}")
    fixed = sorted(c for c in seqlens if c != des)
    assigned[name] = [[des], fixed]
json.dump(assigned, open("$OUT/assigned_chains.jsonl", "w"))
print("assigned:", assigned)
PYEOF

count_seqs(){ [ -f "$OUT/seqs/$1.fa" ] && grep -c "^>" "$OUT/seqs/$1.fa" 2>/dev/null || echo 0; }

for name in "${!SRC[@]}"; do
  have=$(count_seqs "$name"); have=${have:-0}
  if [ "$have" -ge "$NSEQ" ]; then
    echo "[$(date '+%F %T')] $name already at $have/$NSEQ -- skip" | tee -a "$LOG"; continue
  fi
  # per-target parsed/assigned subset (protein_mpnn_run.py takes one jsonl covering all names inside;
  # run one target per invocation so failures/resume are per-target, not all-or-nothing)
  $PY - <<PYEOF
import json
d = None
for line in open("$OUT/parsed_chains.jsonl"):
    j = json.loads(line)
    if j["name"] == "$name": d = line; break
open("$OUT/parsed_$name.jsonl", "w").write(d)
asg = json.load(open("$OUT/assigned_chains.jsonl"))
json.dump({"$name": asg["$name"]}, open("$OUT/assigned_$name.jsonl", "w"))
PYEOF
  gpu_wait 4000
  echo "[$(date '+%F %T')] target=$name have=$have -> generating $NSEQ (T=$TEMP)" | tee -a "$LOG"
  $PY ProteinMPNN/protein_mpnn_run.py \
     --jsonl_path "$OUT/parsed_$name.jsonl" \
     --chain_id_jsonl "$OUT/assigned_$name.jsonl" \
     --out_folder "$OUT" \
     --num_seq_per_target "$NSEQ" \
     --batch_size "$BATCH" \
     --sampling_temp "$TEMP" \
     --seed 37 \
     --path_to_model_weights "$WEIGHTS" \
     --model_name "$MODEL" >>"$LOG" 2>&1
  got=$(count_seqs "$name")
  echo "[$(date '+%F %T')] target=$name done: $got seqs" | tee -a "$LOG"
  [ "$got" -lt "$NSEQ" ] && echo "[$(date '+%F %T')] WARN $name short of target ($got/$NSEQ) -- will retry next invocation" | tee -a "$LOG"
done

all_done=1
for name in "${!SRC[@]}"; do
  have=$(count_seqs "$name"); [ "${have:-0}" -lt "$NSEQ" ] && all_done=0
done
if [ "$all_done" = 1 ]; then
  echo "[$(date '+%F %T')] === noMHC topcross50k ALL TARGETS REACHED $NSEQ ===" | tee -a "$LOG"
  touch "$OUT/COMPLETE"
else
  echo "[$(date '+%F %T')] === pass complete, some targets still short -- supervisor will retry ===" | tee -a "$LOG"
fi
