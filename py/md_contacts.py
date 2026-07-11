#!/usr/bin/env python3
"""Derive the peptide's TARGET contacts from the native-epitope 10 ns MD (300 K).
The first RFD3 runs only anchored the N-term, so the designed C-terminus built out of
the groove and only one TCR chain was engaged. Here we measure which target residues the
NATIVE peptide actually contacts across the trajectory -> an empirical hotspot set that
pins the whole peptide (N+C anchors, groove floor) and BOTH TCR chains.

Contact = peptide heavy atom within 4.5 A of a target-residue heavy atom in a frame.
We report each target residue's contact OCCUPANCY (fraction of frames in contact),
mapped back to native chain + residue number.
"""
import mdtraj as md, numpy as np, warnings, json, os
warnings.filterwarnings("ignore")
from Bio.PDB import PDBParser
three2one = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G',
             'HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S',
             'THR':'T','TRP':'W','TYR':'Y','VAL':'V','MSE':'M'}
ROOT = "/home/ubuntu/if-mhc"
CUTOFF = 0.45  # nm (4.5 A) heavy-atom contact
STRIDE = 5
PEP = {"6AMU": "MMWDRGLGMM", "6AM5": "SMLGIGIVPV"}


def native_order(pid):
    """Ordered list of (chain, resSeq, code) for native chains A,B,C,D,E."""
    p = PDBParser(QUIET=True)
    m = p.get_structure("n", f"{ROOT}/inputs/focus_6am/{pid}.pdb")[0]
    order = []
    for cid in ["A", "B", "C", "D", "E"]:
        if cid not in m:
            continue
        for r in m[cid]:
            if r.id[0] == ' ' and r.resname in three2one:
                order.append((cid, r.id[1], three2one[r.resname]))
    return order


def analyze(pid):
    d = f"{ROOT}/outputs/tamarind/results/ifmhc_{pid}_md_300K"
    top = f"{d}/topology_no_water.pdb"
    xtc = f"{d}/traj_prod_no_water_seg1.xtc"
    t = md.load(xtc, top=top, stride=STRIDE)
    nat = native_order(pid)
    if t.topology.n_residues != len(nat):
        print(f"  WARN {pid}: MD res {t.topology.n_residues} != native {len(nat)} (mapping by order may be off)")
    n = min(t.topology.n_residues, len(nat))
    # peptide residue indices in MD = positions where native chain == 'C'
    pep_idx = [i for i in range(n) if nat[i][0] == 'C']
    tgt_idx = [i for i in range(n) if nat[i][0] != 'C' and nat[i][0] != 'B']  # MHC heavy + TCR (skip b2m)
    # residue pairs (peptide x target) for closest-heavy contacts
    pairs = np.array([[pi, ti] for ti in tgt_idx for pi in pep_idx])
    dists, _ = md.compute_contacts(t, contacts=pairs, scheme="closest-heavy")  # (frames, n_pairs)
    npep = len(pep_idx)
    occ = {}
    for k, ti in enumerate(tgt_idx):
        sub = dists[:, k*npep:(k+1)*npep] if False else None
    # reshape: pairs ordered ti-outer, pi-inner
    dr = dists.reshape(t.n_frames, len(tgt_idx), npep)
    mind = dr.min(axis=2)            # min over peptide atoms, per frame per target res
    occupancy = (mind < CUTOFF).mean(axis=0)   # fraction of frames in contact
    rows = []
    for k, ti in enumerate(tgt_idx):
        cid, resseq, code = nat[ti]
        rows.append({"chain": cid, "resSeq": resseq, "aa": code, "occ": round(float(occupancy[k]), 3)})
    rows.sort(key=lambda r: -r["occ"])
    return t.n_frames, rows


def main():
    summary = {}
    for pid in ["6AMU", "6AM5"]:
        nframes, rows = analyze(pid)
        summary[pid] = rows
        print(f"\n=== {pid}  ({nframes} frames analyzed, contact>={CUTOFF*10:.1f}A heavy) ===")
        for cid in ["A", "D", "E"]:
            top = [r for r in rows if r["chain"] == cid and r["occ"] >= 0.30]
            tag = {"A": "MHC", "D": "TCRa", "E": "TCRb"}[cid]
            s = "  ".join(f"{r['aa']}{r['resSeq']}:{r['occ']:.2f}" for r in top)
            print(f"  {tag} (chain {cid}) occ>=0.30: {s if s else '(none)'}")
    out = f"{ROOT}/outputs/struct_ood/md_native_contacts.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(summary, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
