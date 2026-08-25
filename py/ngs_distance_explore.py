import sys, re
sys.path.append("/home/ubuntu/pmhc/modeling/ONG229/py")
import ong229_ranking_lib as lib
import numpy as np
import pandas as pd

# ---- load NGS ----
tab_data = lib.load_tab_data()
ngs = tab_data["MART1_10mer__CAB60174_G01"]

R3 = ngs[ngs["R3"] > 0]
p95, p99 = R3["R3"].quantile(0.95), R3["R3"].quantile(0.99)
set_top5 = set(R3[R3["R3"] >= p95]["Peptide"])
set_top1 = set(R3[R3["R3"] >= p99]["Peptide"])
set_r2r3 = set(ngs[(ngs["R2"] > 0) & (ngs["R3"] > 0)]["Peptide"])
set_R0 = set(ngs[ngs["R0"] > 0]["Peptide"])
print(f"R3 p95={p95}, p99={p99}")
print(f"NGS sets: top5%={len(set_top5)}, top1%={len(set_top1)}, persists-R2&R3={len(set_r2r3)}, R0-naive={len(set_R0)}")

# ---- load our IF designs ----
def mpnn_seqs(path):
    seqs = []
    lines = open(path).readlines()
    for i in range(0, len(lines), 2):
        if lines[i].startswith(">") and "seq_recovery=" in lines[i]:
            seqs.append(lines[i+1].strip())
    return seqs

def esmif_seqs(path):
    seqs = []
    lines = open(path).readlines()
    for i in range(0, len(lines), 2):
        if lines[i].startswith(">") and "recovery=" in lines[i]:
            seqs.append(lines[i+1].strip())
    return seqs

def ligandmpnn_seqs(path):
    seqs = []
    lines = open(path).readlines()
    for i in range(0, len(lines), 2):
        if lines[i].startswith(">") and "seq_rec=" in lines[i]:
            seqs.append(lines[i+1].strip().split(":")[2])
    return seqs

ROOT = "/home/ubuntu/if-mhc/"
datasets = {
    "vanilla_T0.1": mpnn_seqs(ROOT+"outputs/mpnn_3hg1_100k/archive_T01_partial/vanilla_3HG1_T01_partial_26993.fa"),
    "noMHC_T0.1": mpnn_seqs(ROOT+"outputs/mpnn_3hg1_100k/archive_T01_partial/nomhc_3HG1_T01_partial_26528.fa"),
    "vanilla_T0.3": mpnn_seqs(ROOT+"outputs/mpnn_3hg1_T03_50k/run_vanilla/seqs/3HG1.fa"),
    "noMHC_T0.3": mpnn_seqs(ROOT+"outputs/mpnn_3hg1_T03_50k/run_nomhc/seqs/3HG1.fa"),
    "ESM-IF_T0.1": esmif_seqs(ROOT+"outputs/esmif_3hg1_pilot/seqs/3HG1.fa"),
    "LigandMPNN_T0.1": ligandmpnn_seqs(ROOT+"outputs/ligandmpnn_3hg1_pilot/seqs/3HG1.fa"),
}

# ---- nearest hamming distance from each unique designed peptide to nearest peptide in an NGS set ----
def nearest_hamming(query_peps, ref_peps_arr):
    # ref_peps_arr: numpy array of shape (N, L) uint8
    out = np.empty(len(query_peps), dtype=np.int16)
    for i, q in enumerate(query_peps):
        qa = np.frombuffer(q.encode(), dtype=np.uint8)
        d = (ref_peps_arr != qa).sum(axis=1)
        out[i] = d.min()
    return out

def to_arr(peps):
    return np.array([np.frombuffer(p.encode(), dtype=np.uint8) for p in peps])

ngs_sets = {"top5%(R3)": set_top5, "top1%(R3)": set_top1, "persists_R2&R3": set_r2r3, "R0_naive": set_R0}
ngs_arrs = {name: to_arr(list(s)) for name, s in ngs_sets.items()}

# ---- random baseline: sample random 10-mers respecting the library's anchor constraint (P2 in LIM, P10 in LVI) ----
rng = np.random.default_rng(42)
AA = list("ACDEFGHIKLMNPQRSTVWY")
def random_peptide():
    mid = rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), size=8)
    p2 = rng.choice(list("LIM"))
    p10 = rng.choice(list("LVI"))
    return p2 + "".join(mid) + p10
random_baseline = [random_peptide() for _ in range(2000)]

print()
print(f"{'source':<18}{'n_total':>9}{'n_unique':>10}  " + "".join(f"{k:>16}" for k in ngs_sets))
all_rows = []
for name, seqs in {**datasets, "RANDOM_baseline": random_baseline}.items():
    uniq = sorted(set(seqs))
    row = {"source": name, "n_total": len(seqs), "n_unique": len(uniq)}
    line = f"{name:<18}{len(seqs):>9}{len(uniq):>10}  "
    for ngs_name, arr in ngs_arrs.items():
        d = nearest_hamming(uniq, arr)
        row[f"mean_dist_{ngs_name}"] = d.mean()
        row[f"frac_le2_{ngs_name}"] = (d<=2).mean()
        line += f"{d.mean():>16.2f}"
    all_rows.append(row)
    print(line)

df = pd.DataFrame(all_rows)
df.to_csv("/home/ubuntu/if-mhc/outputs/analysis/ngs_hamming_distance_summary.csv", index=False)
print("\nsaved outputs/analysis/ngs_hamming_distance_summary.csv")
