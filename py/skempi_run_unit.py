#!/usr/bin/env python3
"""Run one work unit of the SKEMPI inverse-folding sweep.

A unit is (model, arm, complex, chunk): a fixed number of epitope sequences
from one model on one structure. Units are sized to finish well inside a 30 min
worker walltime, and each writes its own part file, so preemption costs at most
one chunk. Merge with skempi_merge.py.

  models: esmif | proteinmpnn | proteinmpnn_nomhc | ligandmpnn
"""
import argparse, os, shutil, subprocess, sys, tempfile, time
import pandas as pd

ROOT = "/global/scratch/users/sergiomar10/if-mhc"
SKDIR = {"skempi": f"{ROOT}/inputs/skempi", "pmhc25": f"{ROOT}/inputs/pmhc25",
         "focus6am": f"{ROOT}/inputs/focus6am"}
OUT = f"{ROOT}/outputs/skempi_if"
PMPNN = "/global/scratch/users/sergiomar10/TCera/ProteinMPNN"
LMPNN = "/global/scratch/users/sergiomar10/tools/LigandMPNN"
PY = sys.executable

WEIGHTS = {
    "proteinmpnn":      ("", "v_48_020"),
    "proteinmpnn_nomhc": (f"{PMPNN}/hf_repo/", "proteinmpnn_nomhc"),
}

# Sampling batch per model, sized for the smallest card in savio3_gpu (11 GB
# GTX 2080 Ti). The constraint is activation size (batch x ~830 residues), not
# model size: the MPNNs replicate the whole complex across the batch, while
# ESM-IF only broadcasts one cached encoder output.
BATCH = {"esmif": 64, "proteinmpnn": 16, "proteinmpnn_nomhc": 16, "ligandmpnn": 16}


def vram_scale():
    """Batch multiplier from the card we landed on.

    Workers are not pinned to a GPU type, so the same unit may run on an 11 GB
    2080 Ti or a 48 GB A40. BATCH is sized for the smallest card; scale up when
    there is room rather than leaving a big card two-thirds idle. Read via
    nvidia-smi to avoid paying for a CUDA context just to size a batch.
    """
    try:
        mb = int(subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30).stdout.split("\n")[0])
    except Exception:
        return 1
    return 4 if mb >= 40000 else (2 if mb >= 20000 else 1)


def run(cmd, **kw):
    r = subprocess.run(cmd, **kw)
    if r.returncode != 0:
        raise SystemExit(f"command failed ({r.returncode}): {' '.join(map(str, cmd))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["esmif", "proteinmpnn", "proteinmpnn_nomhc", "ligandmpnn"])
    ap.add_argument("--arm", required=True, choices=["full", "notcr"])
    ap.add_argument("--complex", required=True)
    ap.add_argument("--dataset", default="skempi", choices=["skempi", "pmhc25", "focus6am"])
    ap.add_argument("--chunk", type=int, required=True)
    ap.add_argument("--nseq", type=int, required=True)
    ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=0,
                    help="0 = per-model default sized for an 11 GB card")
    a = ap.parse_args()
    if not a.batch:
        a.batch = BATCH[a.model] * vram_scale()

    # seed varies with chunk AND temperature so no two phases share draws
    seed = 37 + 1000 * a.chunk + int(round(a.temp * 10000))
    phase = f"{a.dataset}/T{a.temp:g}"
    part = f"{OUT}/{phase}/{a.model}/parts/{a.complex}__{a.arm}__c{a.chunk:02d}.csv"
    os.makedirs(os.path.dirname(part), exist_ok=True)
    if os.path.exists(part):
        print(f"already done: {part}")
        return
    t0 = time.time()
    tmp = tempfile.mkdtemp(prefix="skunit_", dir=os.environ.get("TMPDIR", "/tmp"))
    try:
        if a.model == "esmif":
            run([PY, f"{ROOT}/py/esmif_sample.py", "--arm", a.arm, "--dataset", a.dataset,
                 "--complex", a.complex, "--nseq", str(a.nseq), "--temp", str(a.temp),
                 "--batch", str(a.batch), "--seed", str(seed), "--out", tmp])
            src = f"{tmp}/{a.complex}__{a.arm}.csv"
            df = pd.read_csv(src)
            df["model"] = "esmif"

        elif a.model in WEIGHTS:
            wpath, wname = WEIGHTS[a.model]
            pc = f"{OUT}/mpnn_inputs/{a.dataset}/{a.arm}/per_complex/{a.complex}"
            cmd = [PY, f"{PMPNN}/protein_mpnn_run.py",
                   "--jsonl_path", f"{pc}.parsed.jsonl",
                   "--chain_id_jsonl", f"{pc}.assigned.jsonl",
                   "--fixed_positions_jsonl", f"{pc}.fixed.jsonl",
                   "--out_folder", tmp,
                   "--num_seq_per_target", str(a.nseq),
                   "--batch_size", str(a.batch),
                   "--sampling_temp", str(a.temp),
                   "--seed", str(seed), "--model_name", wname]
            if wpath:
                cmd += ["--path_to_model_weights", wpath]
            run(cmd)
            run([PY, f"{ROOT}/py/mpnn_collect.py",
                 "--fasta", f"{tmp}/seqs/{a.complex}.fa", "--complex", a.complex,
                 "--arm", a.arm, "--model", a.model, "--tool", "proteinmpnn", "--dataset", a.dataset,
                 "--temp", str(a.temp), "--out", f"{tmp}/collected.csv"])
            df = pd.read_csv(f"{tmp}/collected.csv")

        else:  # ligandmpnn
            man = pd.read_csv(f"{SKDIR[a.dataset]}/manifest.csv").set_index("complex").loc[a.complex]
            redes = " ".join(f"{man['pep_chain']}{r}"
                             for r in str(man["pep_resids"]).split(";"))
            nbatch = max(1, a.nseq // a.batch)
            run([PY, f"{LMPNN}/run.py",
                 "--model_type", "ligand_mpnn",
                 "--checkpoint_ligand_mpnn", f"{LMPNN}/model_params/ligandmpnn_v_32_010_25.pt",
                 "--pdb_path", f"{SKDIR[a.dataset]}/arm_{a.arm}/{a.complex}.pdb",
                 "--redesigned_residues", redes,
                 "--out_folder", tmp,
                 "--batch_size", str(a.batch),
                 "--number_of_batches", str(nbatch),
                 "--temperature", str(a.temp),
                 "--seed", str(seed), "--verbose", "0"], cwd=LMPNN)
            run([PY, f"{ROOT}/py/mpnn_collect.py",
                 "--fasta", f"{tmp}/seqs/{a.complex}.fa", "--complex", a.complex,
                 "--arm", a.arm, "--model", a.model, "--tool", "ligandmpnn", "--dataset", a.dataset,
                 "--temp", str(a.temp), "--out", f"{tmp}/collected.csv"])
            df = pd.read_csv(f"{tmp}/collected.csv")

        df["dataset"] = a.dataset
        df["chunk"] = a.chunk
        df["seed"] = seed
        df.to_csv(part + ".tmp", index=False)
        os.replace(part + ".tmp", part)        # atomic: readers never see a partial file
        print(f"UNIT OK {a.model}/{a.arm}/{a.complex}/c{a.chunk} "
              f"n={len(df)} uniq={df.seq.nunique()} rec={df.recovery.mean():.3f} "
              f"{time.time()-t0:.1f}s", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
