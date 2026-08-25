#!/usr/bin/env bash
# NEW temperature-matched native campaign: T=0.3, 10K sequences EACH, vanilla (v_48_020) AND noMHC
# (proteinmpnn_nomhc) weights, on native 6AM5 + 6AMU only. Complements the existing T=0.1/50k native
# pair (outputs/focus_6am_50k + outputs/mpnn_nomhc_topcross_50k) with a second, hotter temperature --
# separate output dir so neither existing T=0.1 campaign is touched. Reuses focus_6am's parsed/chain
# jsonl (chain C designed, A+B+D+E fixed -- same complex context both weight sets already use).
# Resumable per (weights, crystal) via flock + per-target skip, same convention as every other
# campaign script in jobs/.
set -uo pipefail
cd /home/ubuntu/if-mhc; ABS=/home/ubuntu/if-mhc
PY=/home/ubuntu/miniforge3/envs/esmcba/bin/python
OUT="$ABS/outputs/native_T03_10k"; mkdir -p "$OUT/seqs"
NSEQ="${NSEQ:-10000}"; BATCH="${BATCH:-8}"; TEMP="${TEMP:-0.3}"
LOG="$OUT/run.log"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2
gpu_wait(){ while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -lt "$1" ]; do sleep 20; done; }
exec 9>"$OUT/runner.lock"; flock -n 9 || { echo "another runner holds lock — exit"; exit 0; }
echo $$ >"$OUT/runner.pid"; trap 'rm -f "$OUT/runner.pid"' EXIT

# filter focus_6am's existing parsed.jsonl/chain_id.jsonl down to 6AM5+6AMU, once, into our own copies
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

echo "[$(date '+%F %T')] === native_T03_10k: 6AM5+6AMU x $NSEQ seqs, T=$TEMP, vanilla+noMHC ===" | tee -a "$LOG"

count_seqs(){ [ -f "$OUT/seqs/$1.fa" ] && grep -c "^>" "$OUT/seqs/$1.fa" 2>/dev/null || echo 0; }

declare -A WEIGHTS_DIR=( [vanilla]="" [nomhc]="ProteinMPNN/nomhc_model_weights/" )
declare -A MODEL_NAME=( [vanilla]="v_48_020" [nomhc]="proteinmpnn_nomhc" )

for weights in vanilla nomhc; do
  for name in 6AM5 6AMU; do
    tag="${weights}_${name}"
    # a bare $name.fa left over from an interrupted run (killed before the post-run mv) belongs to
    # whichever (weights,name) was in flight -- recover it into its tagged file before checking progress
    if [ -f "$OUT/seqs/$name.fa" ]; then
      mv -f "$OUT/seqs/$name.fa" "$OUT/seqs/$tag.fa"
      echo "[$(date '+%F %T')] recovered interrupted-run leftover seqs/$name.fa -> seqs/$tag.fa" | tee -a "$LOG"
    fi
    have=$(count_seqs "$tag"); have=${have:-0}
    if [ "$have" -ge "$NSEQ" ]; then
      echo "[$(date '+%F %T')] $tag already at $have/$NSEQ -- skip" | tee -a "$LOG"; continue
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
    gpu_wait 4000
    echo "[$(date '+%F %T')] target=$tag have=$have -> generating $NSEQ (T=$TEMP, weights=$weights)" | tee -a "$LOG"
    wdir="${WEIGHTS_DIR[$weights]}"
    extra_args=()
    [ -n "$wdir" ] && extra_args=(--path_to_model_weights "$wdir")
    $PY ProteinMPNN/protein_mpnn_run.py \
       --jsonl_path "$OUT/parsed_$name.jsonl" \
       --chain_id_jsonl "$OUT/chain_id_$name.jsonl" \
       --out_folder "$OUT" \
       --num_seq_per_target "$NSEQ" \
       --batch_size "$BATCH" \
       --sampling_temp "$TEMP" \
       --seed 37 \
       --model_name "${MODEL_NAME[$weights]}" \
       "${extra_args[@]}" >>"$LOG" 2>&1
    # protein_mpnn_run.py always writes seqs/<name>.fa -- rename to our tagged filename immediately
    mv -f "$OUT/seqs/$name.fa" "$OUT/seqs/$tag.fa" 2>/dev/null
    got=$(count_seqs "$tag")
    echo "[$(date '+%F %T')] target=$tag done: $got seqs" | tee -a "$LOG"
    [ "$got" -lt "$NSEQ" ] && echo "[$(date '+%F %T')] WARN $tag short of target ($got/$NSEQ) -- will retry next invocation" | tee -a "$LOG"
  done
done

all_done=1
for weights in vanilla nomhc; do
  for name in 6AM5 6AMU; do
    have=$(count_seqs "${weights}_${name}"); [ "${have:-0}" -lt "$NSEQ" ] && all_done=0
  done
done
if [ "$all_done" = 1 ]; then
  echo "[$(date '+%F %T')] === native_T03_10k ALL 4 TARGETS REACHED $NSEQ ===" | tee -a "$LOG"
  touch "$OUT/COMPLETE"
else
  echo "[$(date '+%F %T')] === pass complete, some targets still short -- supervisor will retry ===" | tee -a "$LOG"
fi
