#!/usr/bin/env bash
# T=0.3 companion to run_panel_queue.sh: same 20 structures x 2 conditions x 3 tool scripts, but at
# T=0.3 instead of T=0.1, output to outputs/panel_T03/ (separate tree, does not touch the T=0.1
# results). GATED: waits for the T=0.1 queue's ALL_COMPLETE marker before starting any work, so it
# never competes with the still-running T=0.1 campaign for the shared GPU. Resumable exactly like
# the T=0.1 queue.
set -uo pipefail
cd /home/ubuntu/if-mhc; ABS=/home/ubuntu/if-mhc
export NSEQ="${NSEQ:-10000}"
QLOG="$ABS/outputs/panel_T03/queue.log"
mkdir -p "$ABS/outputs/panel_T03"

GATE="$ABS/outputs/panel/ALL_COMPLETE"
if [ ! -f "$GATE" ]; then
  echo "[$(date '+%F %T')] waiting on $GATE -- T=0.1 panel campaign not complete yet" >> "$QLOG"
  exit 0
fi

STRUCTS=(2P5W 1QSF 1QRN 2BNR 2GJ6 2F53 2F54 3QDG 3QEQ 3QFJ 3GSN 1OGA 3UTS 5C0A 5C0B 5HHO 5EU6 2VLR 4MJI 5NME)
CONDITIONS=(full mhconly)
TOOLS=(mpnn esmif ligandmpnn)

is_complete(){ [ -f "$ABS/outputs/panel_T03/$1/$2/$3/COMPLETE" ]; }

total=0; done_count=0
for pdb in "${STRUCTS[@]}"; do
  for cond in "${CONDITIONS[@]}"; do
    for tool in "${TOOLS[@]}"; do
      total=$((total+1))
      is_complete "$pdb" "$cond" "$tool" && done_count=$((done_count+1))
    done
  done
done
echo "[$(date '+%F %T')] === T03 queue start: $done_count/$total items already complete ===" | tee -a "$QLOG"

for pdb in "${STRUCTS[@]}"; do
  for cond in "${CONDITIONS[@]}"; do
    for tool in "${TOOLS[@]}"; do
      if is_complete "$pdb" "$cond" "$tool"; then
        continue
      fi
      echo "[$(date '+%F %T')] --- running $pdb/$cond/$tool (T=0.3) ---" | tee -a "$QLOG"
      PDB="$pdb" CONDITION="$cond" bash "$ABS/jobs/run_panel_${tool}_T03.sh" >>"$QLOG" 2>&1
      if is_complete "$pdb" "$cond" "$tool"; then
        echo "[$(date '+%F %T')] +++ $pdb/$cond/$tool complete +++" | tee -a "$QLOG"
      else
        echo "[$(date '+%F %T')] !!! $pdb/$cond/$tool did NOT complete this pass (will retry next queue run) !!!" | tee -a "$QLOG"
      fi
    done
  done
done

done_count=0
for pdb in "${STRUCTS[@]}"; do
  for cond in "${CONDITIONS[@]}"; do
    for tool in "${TOOLS[@]}"; do
      is_complete "$pdb" "$cond" "$tool" && done_count=$((done_count+1))
    done
  done
done
echo "[$(date '+%F %T')] === T03 queue pass done: $done_count/$total complete ===" | tee -a "$QLOG"
if [ "$done_count" -eq "$total" ]; then
  touch "$ABS/outputs/panel_T03/ALL_COMPLETE"
  echo "[$(date '+%F %T')] === ALL PANEL T=0.3 CAMPAIGNS COMPLETE ===" | tee -a "$QLOG"
fi
