#!/usr/bin/env python3
"""RFD3 binder-design hotspots derived from the native 10 ns MD contact map
(outputs/struct_ood/md_native_contacts.json).

Lessons folded in:
  - L4 anchored only the N-term -> C-terminus built OUT of the groove. Fix: hotspot the
    FULL groove floor spanning BOTH ends (N-term A/B pocket .. C-term F pocket) so the
    whole peptide threads in. This is the dominant anchor.
  - First native run engaged only one TCR chain. Fix: add the top MD contacts on BOTH
    TCR chains (D and E), kept few so the groove anchor dominates and the peptide doesn't
    get pulled onto the rim.

Selection (data-driven from the MD occupancies):
  chain A : residues with occ >= 0.95, thinned to ~12 spanning the full resSeq range
  chain D : top-3 TCRa contacts by occupancy
  chain E : top-3 TCRb contacts by occupancy
targetChains = A,B,D,E ; binderLength 10,10 ; numDesigns 3.
"""
import json, os, time, requests

API_KEY = os.environ.get("TAMARIND_API_KEY", "")
HEADERS = {"x-api-key": API_KEY}
BASE = "https://app.tamarind.bio/api/"
ROOT = "/home/ubuntu/if-mhc"
CONTACTS = json.load(open(f"{ROOT}/outputs/struct_ood/md_native_contacts.json"))


def span_thin(resseqs, k=12):
    """Keep ~k residues evenly spanning the sorted resSeq range (always keep ends)."""
    rs = sorted(resseqs)
    if len(rs) <= k:
        return rs
    idx = [round(i * (len(rs) - 1) / (k - 1)) for i in range(k)]
    return sorted({rs[i] for i in idx})


def hotspots_for(pid):
    rows = CONTACTS[pid]
    A = [r["resSeq"] for r in rows if r["chain"] == "A" and r["occ"] >= 0.95]
    A = span_thin(A, 12)
    def top(ch, n=3):
        rs = [r for r in rows if r["chain"] == ch]
        rs.sort(key=lambda r: -r["occ"])
        return sorted(r["resSeq"] for r in rs[:n])
    D, E = top("D", 3), top("E", 3)
    return {"A": " ".join(map(str, A)), "D": " ".join(map(str, D)), "E": " ".join(map(str, E))}


def upload(pdb_path, remote):
    r = requests.put(BASE + f"upload/{remote}", headers=HEADERS, data=open(pdb_path, "rb").read(), timeout=120)
    return r.status_code


def submit(job_name, remote_pdb, hot):
    settings = {"task": "protein-binder-design", "pdbFile": remote_pdb,
                "targetChains": ["A", "B", "D", "E"], "hotspots": hot,
                "binderLength": "10,10", "ligands": [], "numDesigns": 3}
    r = requests.post(BASE + "submit-job", headers=HEADERS,
                      json={"jobName": job_name, "type": "rfdiffusion3", "settings": settings}, timeout=120)
    return r.status_code, r.text, settings


def main():
    manifest = []
    for pid in ["6AMU", "6AM5"]:
        hot = hotspots_for(pid)
        print(f"{pid} MD-derived hotspots: {hot}")
        remote = f"{pid}_mdc.pdb"
        uc = upload(f"{ROOT}/inputs/focus_6am/{pid}.pdb", remote)
        jn = f"ifmhc_rfd3_mdc_{pid}"
        sc, txt, settings = submit(jn, remote, hot)
        print(f"  [upload {uc}] [{'OK ' if sc==200 else 'ERR'}] {jn}: HTTP {sc} {txt[:140]}")
        manifest.append({"jobName": jn, "pid": pid, "input": "native-fixed",
                         "conditioning": "MD-derived: full groove floor (both ends) + top-3 each TCR chain",
                         "settings": settings, "status_code": sc, "ok": sc == 200, "response": txt[:200]})
        time.sleep(1)
    out = f"{ROOT}/outputs/tamarind/rfd3_mdc_manifest.json"
    json.dump(manifest, open(out, "w"), indent=2)
    print(f"\nmanifest -> {out}")


if __name__ == "__main__":
    main()
