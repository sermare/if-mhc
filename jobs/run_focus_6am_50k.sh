#!/usr/bin/env bash
# Regular (vanilla, WITH MHC) ProteinMPNN, 50K sequences each, on native 6AM5 and 6AMU only (skips
# 6AMT -- not requested). Same recipe as the original focus_6am run (v_48_020, unconstrained all-20-AA,
# peptide chain designed / rest fixed) but scaled to 50k and T=0.1 (unchanged from the original run,
# which already used T=0.1). Separate output dir so the original 5000-seq run (outputs/focus_6am/,
# with its recovery_table.txt) is untouched. Resumable per-target.
set -uo pipefail
cd /home/ubuntu/if-mhc; ABS=/home/ubuntu/if-mhc
PY=/home/ubuntu/miniforge3/envs/esmcba/bin/python
OUT="$ABS/outputs/focus_6am_50k"; mkdir -p "$OUT/seqs"
NSEQ="${NSEQ:-50000}"; BATCH="${BATCH:-8}"; TEMP="${TEMP:-0.1}"
LOG="$OUT/run.log"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2
exec 9>"$OUT/runner.lock"; flock -n 9 || { echo "another runner holds lock — exit"; exit 0; }
echo $$ >"$OUT/runner.pid"; trap 'rm -f "$OUT/runner.pid"' EXIT

# filter the existing parsed.jsonl / chain_id.jsonl (already built for 6AMU/6AMT/6AM5) down to just
# 6AM5 + 6AMU, once, into this job's own copies
if [ ! -f "$OUT/parsed.jsonl" ]; then
  $PY - <<PYEOF
import json
keep = {"6AM5", "6AMU"}
with open("$OUT/parsed.jsonl", "w") as o:
    for line in open("$ABS/outputs/focus_6am/parsed.jsonl"):
        d = json.loads(line)
        if d["name"] in keep: o.write(line)
chain = json.load(open("$ABS/outputs/focus_6am/chain_id.jsonl"))
json.dump({k: v for k, v in chain.items() if k in keep}, open("$OUT/chain_id.jsonl", "w"))
PYEOF
fi

echo "[$(date '+%F %T')] === focus_6am_50k: 6AM5 + 6AMU x $NSEQ seqs, T=$TEMP (vanilla v_48_020) ===" | tee -a "$LOG"

count_seqs(){ [ -f "$OUT/seqs/$1.fa" ] && grep -c "^>" "$OUT/seqs/$1.fa" 2>/dev/null || echo 0; }

for name in 6AM5 6AMU; do
  have=$(count_seqs "$name"); have=${have:-0}
  if [ "$have" -ge "$NSEQ" ]; then
    echo "[$(date '+%F %T')] $name already at $have/$NSEQ -- skip" | tee -a "$LOG"; continue
  fi
  $PY - <<PYEOF
import json
d = None
for line in open("$OUT/parsed.jsonl"):
    j = json.loads(line)
    if j["name"] == "$name": d = line; break
open("$OUT/parsed_$name.jsonl", "w").write(d)
chain = json.load(open("$OUT/chain_id.jsonl"))
json.dump({"$name": chain["$name"]}, open("$OUT/chain_id_$name.jsonl", "w"))
PYEOF
  echo "[$(date '+%F %T')] target=$name have=$have -> generating $NSEQ (T=$TEMP)" | tee -a "$LOG"
  $PY ProteinMPNN/protein_mpnn_run.py \
     --jsonl_path "$OUT/parsed_$name.jsonl" \
     --chain_id_jsonl "$OUT/chain_id_$name.jsonl" \
     --out_folder "$OUT" \
     --num_seq_per_target "$NSEQ" \
     --batch_size "$BATCH" \
     --sampling_temp "$TEMP" \
     --seed 37 \
     --model_name v_48_020 >>"$LOG" 2>&1
  echo "[$(date '+%F %T')] target=$name done: $(count_seqs "$name") seqs" | tee -a "$LOG"
done

echo "[$(date '+%F %T')] === focus_6am_50k ALL DONE ===" | tee -a "$LOG"
touch "$OUT/COMPLETE"
