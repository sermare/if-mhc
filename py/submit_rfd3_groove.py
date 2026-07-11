#!/usr/bin/env python3
"""RFdiffusion3 protein-binder-design, conditioned on the L4 hotspot set that performed
best in our RFdiffusion contact ladder: N-term MHC anchors + MHC pocket + 3 TCR contacts.

  L4 hotspots (from outputs/ladder/run.log, ncontacts=9):
    6AMU: A159,A66,A70 (N-term) + A9,A77,A80 (pocket) + E100,E99,D30 (3 TCR)
    6AM5: A159,A66,A70 (N-term) + A9,A77,A80 (pocket) + E97,E96,D30 (3 TCR)

  targetChains = ["A","B","D","E"]  -> MHC heavy + b2m + BOTH TCR chains kept.
  binderLength "10,10", numDesigns 3.
"""
import json, os, time, requests

API_KEY = "REDACTED_TAMARIND_KEY"
HEADERS = {"x-api-key": API_KEY}
BASE = "https://app.tamarind.bio/api/"
ROOT = "/home/ubuntu/if-mhc"

# L4 hotspots: N-term + pocket on chain A, 3 TCR contacts on chains E/D. Per crystal.
HOT = {
    "6AMU": {"A": "159 66 70 9 77 80", "E": "100 99", "D": "30"},
    "6AM5": {"A": "159 66 70 9 77 80", "E": "97 96", "D": "30"},
}
JOBS = {
    "6AMU": "inputs/focus_6am/6AMU.pdb",
    "6AM5": "inputs/focus_6am/6AM5.pdb",
}


def upload(pdb_path, remote_name):
    with open(pdb_path, "rb") as fh:
        data = fh.read()
    r = requests.put(BASE + f"upload/{remote_name}", headers=HEADERS, data=data, timeout=120)
    return r.status_code, r.text[:200]


def submit(job_name, remote_pdb, hotspots):
    settings = {
        "task": "protein-binder-design",
        "pdbFile": remote_pdb,
        "targetChains": ["A", "B", "D", "E"],
        "hotspots": hotspots,
        "binderLength": "10,10",
        "ligands": [],
        "numDesigns": 3,
    }
    params = {"jobName": job_name, "type": "rfdiffusion3", "settings": settings}
    r = requests.post(BASE + "submit-job", headers=HEADERS, json=params, timeout=120)
    return r.status_code, r.text, settings


def main():
    manifest = []
    for pid, path in JOBS.items():
        remote = f"{pid}_L4.pdb"
        uc, ut = upload(os.path.join(ROOT, path), remote)
        print(f"[upload] {remote}: HTTP {uc} {ut}")
        jn = f"ifmhc_rfd3_L4_{pid}"
        sc, txt, settings = submit(jn, remote, HOT[pid])
        ok = sc == 200
        print(f"[{'OK ' if ok else 'ERR'}] {jn}: HTTP {sc} {txt[:160]}")
        manifest.append({"jobName": jn, "pid": pid, "input": "native-fixed",
                         "conditioning": "L4: N-term + pocket + 3 TCR (all 4 target chains)",
                         "settings": settings, "status_code": sc, "ok": ok, "response": txt[:200]})
        time.sleep(1)
    out = os.path.join(ROOT, "outputs/tamarind/rfd3_L4_manifest.json")
    with open(out, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nmanifest -> {out}")


if __name__ == "__main__":
    main()
