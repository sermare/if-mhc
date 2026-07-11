#!/usr/bin/env bash
# Wait 10 minutes, then launch both campaigns. Each campaign internally waits for its own
# prerequisites (OpenMM / SE3nv+weights) and for GPU headroom, so this is safe to fire blind.
cd /home/ubuntu/if-mhc
LOG=outputs/campaigns_arm.log
echo "[$(date)] ARMED — launching relax + rfdiff campaigns in 10 min" > "$LOG"
sleep 600
echo "[$(date)] T-0: launching campaigns" >> "$LOG"
setsid bash ./run_relax_campaign.sh >/dev/null 2>&1 &
echo "  relax campaign launched (pid $!)" >> "$LOG"
setsid bash ./run_rfdiff_campaign.sh >/dev/null 2>&1 &
echo "  rfdiff campaign launched (pid $!)" >> "$LOG"
echo "[$(date)] both campaigns launched (self-guarding on prereqs + GPU)" >> "$LOG"
touch outputs/CAMPAIGNS_LAUNCHED
