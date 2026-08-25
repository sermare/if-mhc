#!/usr/bin/env python3
"""Provenance scanner: for every outputs/*/seqs/*.fa in the repo, resolve (a) which job script
produced it (by grepping jobs/*.sh for the containing output directory), (b) vanilla vs noMHC
weights + sampling temperature from that script, and (c) the canonical source backbone PDB (via
manifest.csv's `src` column when present, else by resolving pdb_in/*.pdb symlinks). Then group by
canonical source to find genuinely matched vanilla/noMHC pairs -- the only fair basis for a
same-structure model comparison.

No mocks / no guessing: a fasta file with no resolvable job script or no resolvable source is
reported as "unresolved", not silently dropped or assumed.
"""
import os, re, glob, csv, json
from pathlib import Path

ROOT = Path("/home/ubuntu/if-mhc")
JOBS = sorted(glob.glob(str(ROOT / "jobs/*.sh")))
JOB_TEXT = {j: Path(j).read_text(errors="ignore") for j in JOBS}

def find_owning_jobs(out_dir_rel):
    """out_dir_rel like 'outputs/mpnn_nomhc_allbb'. Return job scripts that assign this as OUT=/STAGE=."""
    hits = []
    pat = re.compile(r'(?:^|[\s"])(?:OUT|STAGE|SRC)=["\']?(?:\$ABS/)?' + re.escape(out_dir_rel) + r'(?:["\'\s/]|$)', re.M)
    for j, txt in JOB_TEXT.items():
        if pat.search(txt) or out_dir_rel in txt:
            hits.append(j)
    return hits

def weights_from_jobs(job_paths):
    for j in job_paths:
        txt = JOB_TEXT[j]
        if "nomhc_model_weights" in txt:
            return "noMHC", j
    for j in job_paths:
        if "protein_mpnn_run.py" in JOB_TEXT[j]:
            return "vanilla", j
    return "unknown", None

def temp_from_job(job_path):
    if job_path is None:
        return None
    txt = JOB_TEXT[job_path]
    m = re.search(r'sampling_temp\s+"?\$?\{?(?:TEMP|T)?\}?"?', txt)
    m2 = re.search(r'--sampling_temp\s+"([^"]+)"', txt)
    if m2:
        return m2.group(1)
    return None

# ---- enumerate every seqs/ dir with .fa files ----
seqs_dirs = sorted(set(str(Path(p).parent) for p in glob.glob(str(ROOT / "outputs/**/seqs"), recursive=True)))
rows = []
for sd in seqs_dirs:
    fa_files = glob.glob(str(Path(sd) / "seqs" / "*.fa"))
    if not fa_files:
        continue
    out_rel = os.path.relpath(sd, ROOT)
    owning = find_owning_jobs(out_rel)
    weights, wjob = weights_from_jobs(owning)
    temp = temp_from_job(wjob)

    # try manifest.csv src column (may live in this dir or a SRC= parent dir referenced by the job)
    manifest = Path(sd) / "manifest.csv"
    src_by_target = {}
    if manifest.exists():
        with open(manifest) as f:
            for r in csv.DictReader(f):
                if "target" in r and "src" in r:
                    src_by_target[r["target"]] = r["src"]

    for fa in fa_files:
        target = Path(fa).stem
        src = src_by_target.get(target)
        if src is None:
            # fall back: resolve pdb_in/<target>.pdb symlink
            cand = Path(sd) / "pdb_in" / f"{target}.pdb"
            if cand.exists() and cand.is_symlink():
                src = str(cand.resolve())
            elif cand.exists():
                src = str(cand.resolve())
        rows.append(dict(out_dir=out_rel, fasta=os.path.relpath(fa, ROOT), target=target,
                          weights=weights, owning_job=(os.path.relpath(wjob, ROOT) if wjob else None),
                          temp=temp, src=src))

MANIFEST = ROOT / "outputs/analysis/mpnn_provenance_manifest.csv"
MANIFEST.parent.mkdir(parents=True, exist_ok=True)
with open(MANIFEST, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["out_dir","fasta","target","weights","owning_job","temp","src"])
    w.writeheader()
    for r in rows:
        w.writerow(r)

print(f"scanned {len(seqs_dirs)} seqs/ dirs, {len(rows)} fasta records -> {MANIFEST}")

# ---- summary by out_dir ----
from collections import Counter, defaultdict
by_dir = defaultdict(list)
for r in rows:
    by_dir[r["out_dir"]].append(r)

print(f"\n{'out_dir':45s} {'n_fasta':>8s} {'weights':>8s} {'temp':>6s}  owning_job")
for d in sorted(by_dir):
    rs = by_dir[d]
    wset = Counter(r["weights"] for r in rs)
    wstr = "/".join(f"{k}:{v}" for k, v in wset.items())
    temps = Counter(r["temp"] for r in rs)
    tstr = ",".join(str(t) for t in temps if t) or "?"
    jobs = Counter(r["owning_job"] for r in rs if r["owning_job"])
    jstr = ",".join(jobs.keys()) if jobs else "UNRESOLVED"
    print(f"{d:45s} {len(rs):8d} {wstr:>8s} {tstr:>6s}  {jstr}")

# ---- group by resolved source to find matched pairs ----
by_src = defaultdict(set)
for r in rows:
    if r["src"]:
        by_src[r["src"]].add(r["weights"])

matched = {s: ws for s, ws in by_src.items() if len(ws) > 1}
print(f"\n{len(by_src)} distinct resolved source structures; {len(matched)} have >1 weight-set (genuine matched pairs)")
if matched:
    for s, ws in list(matched.items())[:20]:
        print(f"  {s}: {ws}")
