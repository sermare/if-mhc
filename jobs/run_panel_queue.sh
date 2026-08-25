#!/usr/bin/env bash
# Master queue for the wider-panel campaign: 20 structures (the pMHC-TCR panel minus 3HG1/2P5E,
# which already have full NGS-grounded campaigns, and minus the 3 structures flagged invalid=False
# in inputs/pmhc_tcr_dataset/dataset.csv: 1AO7, 3UTQ, 5HHM) x 2 context conditions (full = MHC+TCR,
# mhconly = MHC only, no TCR) x 3 tool scripts (mpnn produces vanilla+noMHC together = 4 models total)
# x 10k sequences each. One shared GPU -> strictly serial, one work item at a time. Resumable: each
# underlying job script skips to COMPLETE if already done, so re-running this queue after a kill/crash
# just fast-forwards through finished items.
set -uo pipefail
cd /home/ubuntu/if-mhc; ABS=/home/ubuntu/if-mhc
export NSEQ="${NSEQ:-10000}"
QLOG="$ABS/outputs/panel/queue.log"
mkdir -p "$ABS/outputs/panel"

STRUCTS=(2P5W 1QSF 1QRN 2BNR 2GJ6 2F53 2F54 3QDG 3QEQ 3QFJ 3GSN 1OGA 3UTS 5C0A 5C0B 5HHO 5EU6 2VLR 4MJI 5NME)
CONDITIONS=(full mhconly)
TOOLS=(mpnn esmif ligandmpnn)

is_complete(){ [ -f "$ABS/outputs/panel/$1/$2/$3/COMPLETE" ]; }

total=0; done_count=0
for pdb in "${STRUCTS[@]}"; do
  for cond in "${CONDITIONS[@]}"; do
    for tool in "${TOOLS[@]}"; do
      total=$((total+1))
      is_complete "$pdb" "$cond" "$tool" && done_count=$((done_count+1))
    done
  done
done
echo "[$(date '+%F %T')] === queue start: $done_count/$total items already complete ===" | tee -a "$QLOG"

for pdb in "${STRUCTS[@]}"; do
  for cond in "${CONDITIONS[@]}"; do
    for tool in "${TOOLS[@]}"; do
      if is_complete "$pdb" "$cond" "$tool"; then
        continue
      fi
      echo "[$(date '+%F %T')] --- running $pdb/$cond/$tool ---" | tee -a "$QLOG"
      PDB="$pdb" CONDITION="$cond" bash "$ABS/jobs/run_panel_${tool}.sh" >>"$QLOG" 2>&1
      if is_complete "$pdb" "$cond" "$tool"; then
        echo "[$(date '+%F %T')] +++ $pdb/$cond/$tool complete +++" | tee -a "$QLOG"
      else
        echo "[$(date '+%F %T')] !!! $pdb/$cond/$tool did NOT complete this pass (will retry next queue run) !!!" | tee -a "$QLOG"
      fi
    done
  done
done

# recompute
done_count=0
for pdb in "${STRUCTS[@]}"; do
  for cond in "${CONDITIONS[@]}"; do
    for tool in "${TOOLS[@]}"; do
      is_complete "$pdb" "$cond" "$tool" && done_count=$((done_count+1))
    done
  done
done
echo "[$(date '+%F %T')] === queue pass done: $done_count/$total complete ===" | tee -a "$QLOG"
if [ "$done_count" -eq "$total" ]; then
  touch "$ABS/outputs/panel/ALL_COMPLETE"
  echo "[$(date '+%F %T')] === ALL PANEL CAMPAIGNS COMPLETE ===" | tee -a "$QLOG"
fi
