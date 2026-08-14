#!/usr/bin/env python3
"""Work-unit ledger for the SKEMPI inverse-folding sweep.

The sweep is 4 models x 2 arms x 28 complexes x 10,000 sequences at T=0.1.
That is split into chunks small enough to finish inside a 30 min worker
walltime, so a preemption costs one chunk rather than a whole complex.

  --list     print TODO units (model arm complex chunk nseq), least-covered first
  --status   per-model progress summary; exit 3 when everything is done
  --sweep    delete claim directories older than --stale-min with no part file
"""
import argparse, os, random, signal, sys, time
import pandas as pd

ROOT = "/global/scratch/users/sergiomar10/if-mhc"
OUT = f"{ROOT}/outputs/skempi_if"

# A run is one (dataset, temperature) phase. Both live in the path so phases
# never collide and a part file is self-describing.
#   skempi -> the 28 SKEMPI TCR/pMHC complexes staged in inputs/skempi
#   pmhc25 -> the pre-existing inputs/pmhc_tcr_dataset structures
DATASET = os.environ.get("SK_DATASET", "skempi")
TEMP = os.environ.get("SK_TEMP", "0.1")
MANIFEST = {"skempi":   f"{ROOT}/inputs/skempi/manifest.csv",
            "pmhc25":   f"{ROOT}/inputs/pmhc25/manifest.csv",
            "focus6am": f"{ROOT}/inputs/focus6am/manifest.csv"}[DATASET]
PHASE = f"{DATASET}/T{TEMP}"
CLAIMS = f"{OUT}/{PHASE}/claims"

TOTAL_SEQS = 10000
MODELS = ["esmif", "proteinmpnn", "proteinmpnn_nomhc", "ligandmpnn"]
ARMS = ["full", "notcr"]
# Sequences per chunk, from the measured rate on the largest complex (833 res,
# 11 GB card): each chunk must land well inside the worker's ~24 min compute
# budget. ESM-IF amortises one encoder pass over the whole chunk, so splitting
# it would just pay the model load repeatedly for no resilience gain.
#   esmif       10k in ~280s   -> one chunk
#   proteinmpnn 10k in ~2160s  -> 2.5k chunks (~9 min)
#   nomhc       10k in ~2220s  -> 2.5k chunks (~9 min)
#   ligandmpnn  10k in ~2680s  -> 2.5k chunks (~11 min)
CHUNK = {"esmif": 10000, "proteinmpnn": 2500,
         "proteinmpnn_nomhc": 2500, "ligandmpnn": 2500}

# (dataset, model, complex) cells excluded from the ledger.
#
# LigandMPNN drops zero-occupancy atoms when parsing, unlike ESM-IF and the two
# ProteinMPNNs. In these five SKEMPI complexes that leaves residues with a
# 3-atom backbone, which (a) crashes its output writer on a 4-atoms-per-residue
# reshape and (b) would have had it design against a structure 1-10 residues
# shorter than the one the other three models see. Excluding them keeps the
# cross-model comparison on identical inputs. The other 50 arm-structures parse
# identically in all four models and are unaffected.
BLOCKED = {("skempi", "ligandmpnn", c) for c in
           ["1LP9_ABC_EF", "2J8U_ABC_EF", "2JCC_ABC_EF", "2UWE_ABC_EF", "4P23_CD_AB"]}


def units():
    cplx = list(pd.read_csv(MANIFEST)["complex"])
    for m in MODELS:
        n = CHUNK[m]
        nch = (TOTAL_SEQS + n - 1) // n
        for arm in ARMS:
            for c in cplx:
                if (DATASET, m, c) in BLOCKED:
                    continue
                for k in range(nch):
                    yield m, arm, c, k, min(n, TOTAL_SEQS - k * n)


def part_path(m, arm, c, k):
    return f"{OUT}/{PHASE}/{m}/parts/{c}__{arm}__c{k:02d}.csv"


def claim_path(m, arm, c, k):
    return f"{CLAIMS}/{m}__{arm}__{c}__c{k:02d}"


def main():
    # workers read this through `| head`, so a closed pipe is normal, not an error
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--stale-min", type=float, default=45.0)
    ap.add_argument("--shuffle-seed", type=int, default=None)
    a = ap.parse_args()
    os.makedirs(CLAIMS, exist_ok=True)
    all_units = list(units())

    if a.sweep:
        now, freed = time.time(), 0
        for m, arm, c, k in [(u[0], u[1], u[2], u[3]) for u in all_units]:
            cp = claim_path(m, arm, c, k)
            if os.path.isdir(cp) and not os.path.exists(part_path(m, arm, c, k)):
                if (now - os.path.getmtime(cp)) / 60 > a.stale_min:
                    os.rmdir(cp)
                    freed += 1
        print(f"swept {freed} stale claims (> {a.stale_min:.0f} min, no part file)")
        return

    todo = [u for u in all_units if not os.path.exists(part_path(*u[:4]))]

    if a.status:
        rows = []
        for m in MODELS:
            for arm in ARMS:
                tot = [u for u in all_units if u[0] == m and u[1] == arm]
                dn = [u for u in tot if os.path.exists(part_path(*u[:4]))]
                rows.append({"model": m, "arm": arm, "units": len(tot),
                             "done": len(dn), "pct": round(100 * len(dn) / len(tot), 1),
                             "seqs": sum(u[4] for u in dn)})
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
        claimed = len([1 for u in all_units
                       if os.path.isdir(claim_path(*u[:4])) and
                       not os.path.exists(part_path(*u[:4]))])
        print(f"\nTOTAL[{PHASE}] {len(all_units)-len(todo)}/{len(all_units)} units "
              f"({100*(len(all_units)-len(todo))/len(all_units):.1f}%), "
              f"{df.seqs.sum():,} sequences, {claimed} in flight")
        sys.exit(3 if not todo else 0)

    if a.list:
        # unclaimed first so workers spread out instead of colliding
        free = [u for u in todo if not os.path.isdir(claim_path(*u[:4]))]
        rnd = random.Random(a.shuffle_seed if a.shuffle_seed is not None
                            else os.getpid())
        rnd.shuffle(free)
        for m, arm, c, k, n in free:
            print(f"{m} {arm} {c} {k} {n} {DATASET} {TEMP}")


if __name__ == "__main__":
    main()
