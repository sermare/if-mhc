#!/usr/bin/env bash
# noMHC ProteinMPNN design on the in-basin diffused backbones (Q30 campaign, Stage 4).
# Peptide chain C designed against the full fixed MHC+TCR context. Parameterized so the V3 variants
# reuse ONE shared staging (pdb_in/parsed/assigned/manifest/fixed_positions) built by
# py/prep_q30_rfdiff_mpnn.py.
#
# Env knobs:
#   STAGE  shared staging dir (default outputs/mpnn_q30) : pdb_in/, manifest.csv, fixed_positions.jsonl
#   OUT    where seqs/ + designs.csv go     (default = STAGE)
#   TEMP   ProteinMPNN sampling temperature (default 0.3)
#   FIXED  fixed_positions jsonl path, or "none" for ALL-FREE design (default $STAGE/fixed_positions.jsonl)
#   NSEQ   sequences per backbone           (default 512)
#   BATCH  batch size                        (default 16)
set -euo pipefail
cd /home/ubuntu/if-mhc
PY=/home/ubuntu/miniforge3/envs/esmcba/bin/python
STAGE="${STAGE:-outputs/mpnn_q30}"
OUT="${OUT:-$STAGE}"
IN="$STAGE/pdb_in"
TEMP="${TEMP:-0.3}"
FIXED="${FIXED:-$STAGE/fixed_positions.jsonl}"
NSEQ="${NSEQ:-512}"; BATCH="${BATCH:-16}"
WEIGHTS=ProteinMPNN/nomhc_model_weights/; MODEL=proteinmpnn_nomhc
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"
echo "[$(date)] START MPNN | STAGE=$STAGE OUT=$OUT temp=$TEMP fixed=$FIXED | $(ls $IN/*.pdb 2>/dev/null | wc -l) targets x $NSEQ" | tee "$OUT/run.log"

# 1) parse + assigned ONCE in STAGE (shared across variants)
if [ ! -f "$STAGE/parsed_chains.jsonl" ]; then
  $PY ProteinMPNN/helper_scripts/parse_multiple_chains.py \
     --input_path "$IN" --output_path "$STAGE/parsed_chains.jsonl" >> "$OUT/run.log" 2>&1
  $PY - "$STAGE" <<'PYEOF' 2>>"$OUT/run.log"
import json, csv, sys
STAGE=sys.argv[1]
pep={r["target"]:r["pep_chain"] for r in csv.DictReader(open(f"{STAGE}/manifest.csv"))}
assigned={}
for line in open(f"{STAGE}/parsed_chains.jsonl"):
    d=json.loads(line); name=d["name"]
    chains=sorted(k.split("_")[-1] for k in d if k.startswith("seq_chain_"))
    des=pep.get(name,"C")
    if des not in chains: des=chains[0]
    assigned[name]=[[des],[c for c in chains if c!=des]]
json.dump(assigned, open(f"{STAGE}/assigned_chains.jsonl","w"))
print("assigned targets:",len(assigned))
PYEOF
fi

# 2) run noMHC ProteinMPNN (fixed positions only if FIXED != none)
FIXED_ARG=""; [ "$FIXED" != "none" ] && FIXED_ARG="--fixed_positions_jsonl $FIXED"
$PY ProteinMPNN/protein_mpnn_run.py \
   --jsonl_path "$STAGE/parsed_chains.jsonl" \
   --chain_id_jsonl "$STAGE/assigned_chains.jsonl" \
   $FIXED_ARG \
   --out_folder "$OUT" \
   --num_seq_per_target "$NSEQ" --batch_size "$BATCH" \
   --sampling_temp "$TEMP" --seed 37 \
   --path_to_model_weights "$WEIGHTS" --model_name "$MODEL" >> "$OUT/run.log" 2>&1

echo "[$(date)] MPNN done, aggregating" | tee -a "$OUT/run.log"

# 3) aggregate -> designs.csv joined to shared manifest
$PY - "$STAGE" "$OUT" "$TEMP" "$FIXED" <<'PYEOF' 2>>"$OUT/run.log"
import csv, re, glob, os, sys
STAGE,OUT,TEMP,FIXED=sys.argv[1:5]
meta={r["target"]:r for r in csv.DictReader(open(f"{STAGE}/manifest.csv"))}
rows=[]
for fa in sorted(glob.glob(f"{OUT}/seqs/*.fa")):
    target=os.path.splitext(os.path.basename(fa))[0]; m=meta.get(target,{})
    lines=open(fa).read().splitlines()
    for i in range(0,len(lines)-1,2):
        h=lines[i]
        if not h.startswith(">T="): continue
        d=dict(re.findall(r'(\w+)=([-\d.]+)', h))
        rows.append({"target":target,"pid":m.get("pid"),"seed":m.get("seed"),"basin":m.get("basin"),
                     "toGIG":m.get("toGIG"),"toDRG":m.get("toDRG"),"temp":TEMP,
                     "fixed":("none" if FIXED=="none" else "anchors"),"peptide":lines[i+1].strip(),
                     "score":d.get("score"),"global_score":d.get("global_score")})
fields=["target","pid","seed","basin","toGIG","toDRG","temp","fixed","peptide","score","global_score"]
with open(f"{OUT}/designs.csv","w",newline="") as o:
    w=csv.DictWriter(o,fieldnames=fields); w.writeheader(); w.writerows(rows)
print("sequences:",len(rows),"| unique:",len(set(r["peptide"] for r in rows)),"| targets:",len(set(r["target"] for r in rows)))
PYEOF
echo "[$(date)] DONE $OUT" | tee -a "$OUT/run.log"; touch "$OUT/COMPLETE"
