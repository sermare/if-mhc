#!/usr/bin/env bash
set -uo pipefail; ABS=/global/scratch/users/sergiomar10/if-mhc
source /clusterfs/nilah/sergio/miniconda3/etc/profile.d/conda.sh; conda activate SE3nv
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}" DGLBACKEND=pytorch
cd "$ABS/RFdiffusion"; O=$ABS/outputs/exp_detcheck; mkdir -p $O
R5="A159,A7,A99,A63,A147,A66,A77,A70,A155,D30,E97,A143"
for rep in A B; do
  python run_inference.py inference.input_pdb=$ABS/inputs/focus_6am/6AM5_trim.pdb \
    "contigmap.contigs=[A1-180/0 B1-100/0 D1-115/0 E1-120/0 C1-10]" \
    inference.num_designs=3 inference.deterministic=True diffuser.T=50 diffuser.partial_T=15 \
    inference.ckpt_override_path=models/Complex_base_ckpt.pt "ppi.hotspot_res=[$R5]" \
    inference.output_prefix=$O/rep${rep} >$O/rep${rep}.log 2>&1
done
echo "=== determinism: are repA_i and repB_i bit-identical? ==="
for i in 0 1 2; do
  if [ -f "$O/repA_$i.pdb" ] && [ -f "$O/repB_$i.pdb" ]; then
    md5A=$(grep '^ATOM' $O/repA_$i.pdb | md5sum | cut -d' ' -f1)
    md5B=$(grep '^ATOM' $O/repB_$i.pdb | md5sum | cut -d' ' -f1)
    [ "$md5A" = "$md5B" ] && echo "  design $i: IDENTICAL (pairing is REAL)" || echo "  design $i: DIFFERS (pairing only nominal!)"
  fi
done
