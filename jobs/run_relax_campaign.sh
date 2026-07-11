#!/usr/bin/env bash
# PHASE 1 campaign: OpenMM relaxation -> backbone ensemble -> ProteinMPNN per backbone.
set -uo pipefail
cd /home/ubuntu/if-mhc
source /home/ubuntu/miniforge3/etc/profile.d/conda.sh
OUT=outputs/relax_campaign; mkdir -p "$OUT/snapshots"
LOG="$OUT/campaign.log"
OMM=/home/ubuntu/miniforge3/envs/openmm/bin/python
MPNN=/home/ubuntu/miniforge3/envs/esmcba/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
N_SNAP=12; SEQS_PER=2000
gpu_wait(){ while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)" -lt "$1" ]; do sleep 30; done; }

echo "[$(date)] RELAX campaign: waiting for OpenMM install..." | tee "$LOG"
while [ ! -f OPENMM_DONE ]; do sleep 20; done
echo "[$(date)] OpenMM ready. Building relaxed backbone ensemble (N=$N_SNAP)" | tee -a "$LOG"

# 1) OpenMM: fix -> minimize -> short implicit-solvent MD -> N snapshot PDBs (chains preserved)
$OMM - "$N_SNAP" >>"$LOG" 2>&1 <<'PY'
import sys
from pdbfixer import PDBFixer
from openmm.app import *
from openmm import *
from openmm.unit import *
N=int(sys.argv[1])
fixer=PDBFixer(filename="inputs/2P5E.pdb")
fixer.removeHeterogens(False)      # drop GOL/ligands/waters (no FF template)
fixer.findMissingResidues(); fixer.findMissingAtoms(); fixer.addMissingAtoms(); fixer.addMissingHydrogens(7.0)
PDBFile.writeFile(fixer.topology, fixer.positions, open("outputs/relax_campaign/fixed.pdb","w"), keepIds=True)
ff=ForceField("amber14-all.xml","implicit/gbn2.xml")
system=ff.createSystem(fixer.topology, nonbondedMethod=CutoffNonPeriodic,
                       nonbondedCutoff=1.0*nanometer, constraints=HBonds)
integ=LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
sim=Simulation(fixer.topology, system, integ, Platform.getPlatformByName("CPU"))
sim.context.setPositions(fixer.positions)
sim.minimizeEnergy(maxIterations=500)
sim.context.setVelocitiesToTemperature(300*kelvin)
top=fixer.topology
for i in range(N):
    sim.step(1000)                       # ~2 ps between snapshots
    st=sim.context.getState(getPositions=True)
    PDBFile.writeFile(top, st.getPositions(), open(f"outputs/relax_campaign/snapshots/snap_{i:02d}.pdb","w"), keepIds=True)
    print("snapshot", i, "saved", flush=True)
print("ENSEMBLE DONE")
PY
nsnap=$(ls "$OUT"/snapshots/*.pdb 2>/dev/null | wc -l)
echo "[$(date)] snapshots produced: $nsnap" | tee -a "$LOG"
[ "$nsnap" -lt 1 ] && { echo "RELAX FAILED: no snapshots"; touch "$OUT/FAILED"; exit 1; }

# 2) ProteinMPNN on each backbone (design peptide chain C, full-complex context)
$MPNN ProteinMPNN/helper_scripts/parse_multiple_chains.py --input_path="$OUT/snapshots" --output_path="$OUT/parsed.jsonl" >>"$LOG" 2>&1
$MPNN ProteinMPNN/helper_scripts/assign_fixed_chains.py --input_path="$OUT/parsed.jsonl" --output_path="$OUT/assigned.jsonl" --chain_list "C" >>"$LOG" 2>&1
# forbid Met at peptide position 1 (chain C) across all backbones
$MPNN make_omit_M_pos1.py "$OUT/parsed.jsonl" "$OUT/omit.jsonl" C 1 M >>"$LOG" 2>&1
gpu_wait 8000
echo "[$(date)] running ProteinMPNN over ensemble ($SEQS_PER seqs/backbone, no Met@P1)" | tee -a "$LOG"
$MPNN ProteinMPNN/protein_mpnn_run.py --jsonl_path "$OUT/parsed.jsonl" --chain_id_jsonl "$OUT/assigned.jsonl" \
   --omit_AA_jsonl "$OUT/omit.jsonl" \
   --out_folder "$OUT" --num_seq_per_target $SEQS_PER --batch_size 16 \
   --sampling_temp "0.3" --seed 37 --model_name v_48_020 >>"$LOG" 2>&1

# 3) aggregate
$MPNN - "$OUT" >>"$LOG" 2>&1 <<'PY'
import sys,glob,re,csv
out=sys.argv[1]; rows=[]
for fa in glob.glob(f"{out}/seqs/*.fa"):
    ls=open(fa).read().splitlines()
    for i in range(0,len(ls)-1,2):
        h,s=ls[i],ls[i+1].strip()
        if "sample=" not in h or len(s)!=9: continue
        rows.append({"backbone":fa.split("/")[-1].replace(".fa",""),"peptide":s})
csv.DictWriter(open(f"{out}/ensemble_peptides.csv","w",newline=""),fieldnames=["backbone","peptide"]).writerows(
    [{"backbone":"backbone","peptide":"peptide"}]+rows)
print(f"ensemble designs: {len(rows)} | unique: {len(set(r['peptide'] for r in rows))}")
PY
echo "[$(date)] RELAX campaign DONE" | tee -a "$LOG"
touch "$OUT/COMPLETE"
