#!/bin/bash
# Local OpenMM MD of every relaxed DESIGN complex at 300 K and 370 K (implicit gbn2, OpenCL).
# Concurrency-limited pool (PAR jobs at once) so we share the L4 with other GPU work without OOM.
cd /home/ubuntu/if-mhc
P=/home/ubuntu/miniforge3/envs/openmm/bin/python
export P; export NS="${NS:-2}"; export PAR="${PAR:-5}"
mkdir -p outputs/design_md
run_one(){
  local pdb="$1" T="$2"
  local name; name=$(basename "$(dirname "$pdb")" | sed 's/ifmhc_relax_//')
  PDB="$pdb" TAG="${name}_${T}K" T_K="$T" NS="$NS" CHUNK_PS=50 "$P" py/design_md.py
}
export -f run_one
: > outputs/design_md/joblist.txt
for pdb in outputs/tamarind/results/ifmhc_relax_6AM*/relaxed.pdb; do
  for T in 300 370; do echo "$pdb $T" >> outputs/design_md/joblist.txt; done
done
echo "launching $(wc -l < outputs/design_md/joblist.txt) design-MD jobs, $PAR concurrent, NS=$NS"
cat outputs/design_md/joblist.txt | xargs -P "$PAR" -L1 bash -c 'run_one "$0" "$1"'
echo "DESIGN_MD_ALL_DONE"
