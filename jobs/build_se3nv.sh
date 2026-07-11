#!/usr/bin/env bash
# Build a modernized SE3nv env that supports the L4 (Ada, sm_89 -> needs CUDA >= 11.8).
# Default RFdiffusion env (torch1.9/cu11.1) does NOT support sm_89.
set -uo pipefail
cd /home/ubuntu/if-mhc
LOG=/home/ubuntu/if-mhc/RFdiffusion/build_env.log
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
echo "[$(date)] creating SE3nv (py3.10, torch2.1.2+cu118)" | tee "$LOG"

conda create -y -n SE3nv python=3.10 >>"$LOG" 2>&1
conda activate SE3nv

set -x
pip install --no-cache-dir torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118 >>"$LOG" 2>&1
pip install --no-cache-dir "numpy<2" scipy hydra-core omegaconf pyrsistent icecream >>"$LOG" 2>&1
# DGL matching torch2.1 + cu118
pip install --no-cache-dir dgl -f https://data.dgl.ai/wheels/torch-2.1/cu118/repo.html >>"$LOG" 2>&1
# SE3Transformer deps (wandb pin relaxed to avoid conflicts)
pip install --no-cache-dir e3nn==0.5.5 pynvml decorator wandb "git+https://github.com/NVIDIA/dllogger#egg=dllogger" >>"$LOG" 2>&1
set +x

echo "[$(date)] building NVIDIA SE3Transformer" | tee -a "$LOG"
cd /home/ubuntu/if-mhc/RFdiffusion/env/SE3Transformer
python setup.py install >>"$LOG" 2>&1

echo "[$(date)] import smoke test (CPU-only; GPU kernels validated separately)" | tee -a "$LOG"
python - >>"$LOG" 2>&1 <<'PY'
import torch, dgl, hydra, e3nn
print("torch", torch.__version__, "cuda_built", torch.version.cuda)
print("dgl", dgl.__version__, "| e3nn", e3nn.__version__)
import se3_transformer; print("se3_transformer import OK")
PY
echo "[$(date)] SE3nv build script finished (exit $?)" | tee -a "$LOG"
touch /home/ubuntu/if-mhc/RFdiffusion/ENV_BUILD_DONE
