#!/usr/bin/env python3
"""Analysis library for the Q30 conditioning campaign.

THE QUESTION: did interaction-only conditioning pull rebuilt peptide backbones OFF the
toGIG/toDRG diagonal INTO a basin? (Not generic structure QC.)

Make-or-break: the superposition frame. Every design is superposed on the MHC beta-sheet FLOOR
Cα only (excludes peptide, TCR, and the mobile α2 short arm His145-Val152). That invariant frame
(6AM5↔6AMU floor RMSD ≈ 0.21 Å) is what makes the basin scatter interpretable. Peptide Cα-RMSD is
then measured to THREE references (a single ref cannot resolve register):
  GIG        = 6AM5 chain C  (canonical, P10 F-pocket)
  DRG_shift  = 6AMU chain C  (shifted register, P9 F-pocket)
  DRG_unshift= 6AMT chain C  (unshifted DRG, excluded from seeding)

Handles both output layouts: partial-diffusion arms = one merged chain in contig order
(peptide = residues nA+nB+1..+10); N-slide scaffold arm = peptide chain A, receptor chain B.
"""
import os, re, glob, csv
import numpy as np, pandas as pd
from Bio.PDB import PDBParser

ROOT = "/home/ubuntu/if-mhc"
OUTS = f"{ROOT}/outputs"
_pp = PDBParser(QUIET=True)
AA = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
      'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V','MSE':'M'}

# ── MHC β-sheet groove floor (author numbering), helices + mobile α2 arm 145-152 EXCLUDED ──
FLOOR = (list(range(3, 13)) + list(range(21, 29)) + list(range(31, 38)) + list(range(46, 53)) +
         list(range(94, 105)) + list(range(108, 119)) + list(range(121, 128)))
# F-pocket floor residues (C-terminal pocket), for the register centroid readout
FPOCKET = [77, 80, 84, 116, 123, 143]

# hotspots conditioned per arm (epitope-specific), for the contact-satisfaction test
ARM_HOTSPOTS = {
    "tcr":     {"6AM5": [("D", 30), ("E", 97)],           "6AMU": [("D", 30), ("E", 100)]},
    "apoc":    {"6AM5": [("D", 30), ("E", 97)]  + [("A", n) for n in (7,63,66,70,99,159,171)],
                "6AMU": [("D", 30), ("E", 100)] + [("A", n) for n in (7,63,66,70,99,159,171)]},
    "nslide":  {"6AM5": [("A", n) for n in (7,63,66,70,99,159,171)],
                "6AMU": [("A", n) for n in (7,63,66,70,99,159,171)]},
    "fpocket": {"6AM5": [("D",30),("A",77),("A",84),("A",116),("A",123)],
                "6AMU": [("D",30),("A",77),("A",84),("A",116),("A",123)]},
}
ARM_DIR = {"tcr": "rfdiff_q30_tcr", "apoc": "rfdiff_q30_apoc", "nslide": "rfdiff_q30_nslide", "fpocket": "rfdiff_q30",
           "sc_tcr": "rfdiff_q30_sc_tcr", "sc_apoc": "rfdiff_q30_sc_apoc", "sc_fpocket": "rfdiff_q30_sc_fpocket"}
# scaffold arms reuse their partial counterpart's hotspot sets
ARM_HOTSPOTS.update({"sc_tcr": ARM_HOTSPOTS["tcr"], "sc_apoc": ARM_HOTSPOTS["apoc"], "sc_fpocket": ARM_HOTSPOTS["fpocket"]})
# v4 arm 1: region-specific partial diffusion (inpaint_str releases pocket+arm+C-term); partial-style _T_ naming
ARM_DIR["v4arm1"] = "rfdiff_q30_v4arm1"; ARM_HOTSPOTS["v4arm1"] = ARM_HOTSPOTS["fpocket"]
SCAFFOLD_ARMS = {"nslide", "sc_tcr", "sc_apoc", "sc_fpocket"}
_SEG_RE = re.compile(r"([A-Z])(-?\d+)-(-?\d+)")


# ── geometry ─────────────────────────────────────────────────────────────────
def kabsch(P, Q):
    """R,t mapping P onto Q (P@R+t ≈ Q) + rmsd, over matched rows."""
    Pc, Qc = P.mean(0), Q.mean(0)
    V, S, Wt = np.linalg.svd((P - Pc).T @ (Q - Qc))
    d = np.sign(np.linalg.det(V @ Wt))
    R = V @ np.diag([1, 1, d]) @ Wt
    t = Qc - Pc @ R
    rmsd = float(np.sqrt(((P @ R + t - Q) ** 2).sum() / len(P)))
    return R, t, rmsd


def rmsd(a, b):
    return float(np.sqrt(((a - b) ** 2).sum() / len(a)))


# ── seed contig segments (for reconstructing author numbering in renumbered outputs) ──
def _seed_segments():
    segs = {}
    man = f"{OUTS}/rfdiff_q30/seeds_manifest.tsv"
    for r in csv.DictReader(open(man), delimiter="\t"):
        segs[r["name"]] = [(c, int(lo), int(hi), int(hi) - int(lo) + 1)
                           for c, lo, hi in _SEG_RE.findall(r["contig"])]
    return segs
SEED_SEG = _seed_segments()


def reconstruct(pdb, segs):
    """(chain, author_resnum) -> Bio residue, undoing RFdiffusion renumbering via the seed contig.
    partial = one merged chain in full contig order; scaffold = peptide chain A + receptor chain B."""
    m = _pp.get_structure("d", pdb)[0]
    chains = {c.id: [r for r in c if r.id[0] == " " and "CA" in r] for c in m}
    resmap = {}
    if "A" in chains and len(chains["A"]) == 10:                      # SCAFFOLD (peptide = chain A)
        for i, r in enumerate(chains["A"]):
            resmap[("C", 1 + i)] = r
        recv = chains.get("B", [])
        idx = 0
        for ch, lo, hi, L in [s for s in segs if s[0] != "C"]:
            for a in range(L):
                if idx < len(recv):
                    resmap[(ch, lo + a)] = recv[idx]; idx += 1
    else:                                                            # PARTIAL (single merged chain)
        allres = [r for c in m for r in c if r.id[0] == " " and "CA" in r]
        idx = 0
        for ch, lo, hi, L in segs:
            for a in range(L):
                if idx < len(allres):
                    resmap[(ch, lo + a)] = allres[idx]; idx += 1
    return resmap


# ── reference peptides, all placed in the 6AMU floor frame ────────────────────
def _resmap_crystal(pdb):
    m = _pp.get_structure("c", pdb)[0]
    rm = {}
    for cid in m:
        for r in cid:
            if r.id[0] == " " and "CA" in r:
                rm[(cid.id, r.id[1])] = r
    return rm

def _floor_ca(resmap):
    return {n: resmap[("A", n)]["CA"].coord for n in FLOOR if ("A", n) in resmap}

def _to_ref_frame(resmap, ref_floor):
    """superpose this structure's floor CA onto the reference floor; return (R,t,floor_rmsd, common)."""
    fc = _floor_ca(resmap)
    common = [n for n in ref_floor if n in fc]
    if len(common) < 20:
        return None
    P = np.array([fc[n] for n in common]); Q = np.array([ref_floor[n] for n in common])
    R, t, fr = kabsch(P, Q)
    return R, t, fr, common

_REF6AMU = _resmap_crystal(f"{ROOT}/inputs/focus_6am/6AMU.pdb")
REF_FLOOR = _floor_ca(_REF6AMU)                                       # frame anchor (identity)
def _pep_ca(resmap, chain="C"):
    return np.array([resmap[(chain, i)]["CA"].coord for i in range(1, 11) if (chain, i) in resmap])

def _ref_pep_in_frame(pdb):
    rm = _resmap_crystal(pdb)
    tf = _to_ref_frame(rm, REF_FLOOR)
    pep = _pep_ca(rm)
    if tf is None:
        return pep
    R, t, _, _ = tf
    return pep @ R + t

REFS = {
    "GIG":         _ref_pep_in_frame(f"{ROOT}/inputs/focus_6am/6AM5.pdb"),
    "DRG_shift":   _pep_ca(_REF6AMU),                                 # 6AMU already in frame
    "DRG_unshift": _ref_pep_in_frame(f"{ROOT}/inputs/focus_6am/6AMT.pdb"),
}
# F-pocket centroid (ref frame) for the register readout
FPOCKET_CENTROID = np.mean([_REF6AMU[("A", n)]["CA"].coord for n in FPOCKET if ("A", n) in _REF6AMU], axis=0)

_FNAME = re.compile(r"^(?P<seed>.+?)_(?:T(?P<T>\d+)|(?P<W>W\d+))_(?P<idx>\d+)(?:_split)?\.pdb$")


def _list_designs(arm):
    d = f"{OUTS}/{ARM_DIR[arm]}"
    if arm in SCAFFOLD_ARMS:
        return sorted(f for f in glob.glob(f"{d}/*_W*_*.pdb") if "_split" not in f)
    return sorted(f for f in glob.glob(f"{d}/*_T*_*.pdb") if "_split" not in f)


def analyze_design(path, arm):
    m = _FNAME.match(os.path.basename(path))
    if not m:
        return None
    seed = m["seed"]
    segs = SEED_SEG.get(seed)
    if segs is None:
        return None
    pid = "6AM5" if seed.startswith("6AM5") else "6AMU"
    resmap = reconstruct(path, segs)
    tf = _to_ref_frame(resmap, REF_FLOOR)
    pep = _pep_ca(resmap)
    if tf is None or len(pep) != 10:
        return None
    R, t, floor_rmsd, _ = tf
    pcf = pep @ R + t                                                # peptide CA in floor frame

    row = dict(arm=arm, seed=seed, pid=pid, cond=m["T"] or m["W"], idx=int(m["idx"]),
               file=path, floor_rmsd=round(floor_rmsd, 3))
    # 1) RMSD to the three references
    for name, ref in REFS.items():
        if len(ref) == 10:
            row[f"to_{name}"] = round(rmsd(pcf, ref), 3)
    # 2) N- vs C-terminal split (register is a C-terminal event; N is what we conditioned)
    for name, ref in REFS.items():
        if len(ref) == 10:
            row[f"Nterm_{name}"] = round(rmsd(pcf[:5], ref[:5]), 3)   # P1-5
            row[f"Cterm_{name}"] = round(rmsd(pcf[5:], ref[5:]), 3)   # P6-10
    # 3) basin assignment on the GIG vs DRG_shift headline axis
    g, d = row.get("to_GIG", np.nan), row.get("to_DRG_shift", np.nan)
    # STRICT membership: threshold = half the GIG↔DRG separation (2.91/2 = 1.45 Å); beyond it the two
    # basins overlap and membership is undefined ("ambiguous"), so a design must be <1.45 Å to belong.
    THR = 1.45
    row["basin"] = "GIG" if g < d else "DRG_shift"                    # nearest (only meaningful if seated)
    row["min_ref"] = round(min(g, d), 3)
    row["seated"] = int(min(g, d) < THR)                             # clearly in a basin
    row["membership"] = ("GIG" if g < THR else "DRG_shift" if d < THR else "ambiguous")
    row["off_diagonal"] = round(abs(g - d), 3)                       # distance from y=x
    # 4) F-pocket register readout (RMSD-independent): which peptide CA sits in the F-pocket centroid
    dists = np.linalg.norm(pcf - FPOCKET_CENTROID, axis=1)
    row["fpocket_pos"] = int(np.argmin(dists) + 1)                   # 10 => GIG-like, 9 => DRG-shifted
    row["fpocket_dist"] = round(float(dists.min()), 2)
    # 5) contact satisfaction (backbone-only designs -> Cβ-Cβ, RFdiffusion's hotspot convention):
    #    per conditioned residue, min over peptide positions of Cβ(pep)-Cβ(hotspot). ≤8 Å = contact.
    csat = {}
    pcb = _pep_cb(resmap)
    for (ch, num) in ARM_HOTSPOTS[arm][pid]:
        hb = _res_cb(resmap.get((ch, num)))
        if hb is None or len(pcb) == 0:
            continue
        csat[f"{ch}{num}"] = round(float(np.min(np.linalg.norm(pcb - hb, axis=1))), 2)
    row["contacts"] = csat
    row["contacts_frac"] = round(float(np.mean([v < 8.0 for v in csat.values()])), 2) if csat else 0.0
    # 6) physical sanity: Cα-Cα virtual bond lengths + self-clashes
    bl = np.linalg.norm(np.diff(pcf, axis=0), axis=1)
    row["bad_bond"] = int(((bl < 3.0) | (bl > 4.5)).any())           # ideal ~3.8 Å
    nb = [np.linalg.norm(pcf[i] - pcf[j]) for i in range(10) for j in range(i + 2, 10)]
    row["clash"] = int(min(nb) < 3.0) if nb else 0
    row["broken"] = int(row["bad_bond"] or row["clash"])
    return row, pcf


def _res_cb(r):
    if r is None:
        return None
    if "CB" in r:
        return r["CB"].coord
    return r["CA"].coord if "CA" in r else None

def _pep_cb(resmap):
    out = [_res_cb(resmap.get(("C", i))) for i in range(1, 11)]
    return np.array([c for c in out if c is not None])


def load_all(arms=("fpocket", "tcr", "apoc", "nslide")):
    """DataFrame of per-design metrics + (N,10,3) array of floor-frame peptide Cα, row-aligned."""
    rows, coords = [], []
    for arm in arms:
        for f in _list_designs(arm):
            try:
                res = analyze_design(f, arm)
            except Exception:
                res = None
            if res is None:
                continue
            row, pcf = res
            rows.append(row); coords.append(pcf)
    df = pd.DataFrame(rows).reset_index(drop=True)
    return df, (np.array(coords) if coords else np.zeros((0, 10, 3)))


if __name__ == "__main__":
    df, CO = load_all()
    print(f"designs analyzed: {len(df)}  coords {CO.shape}")
    if len(df):
        print(df.groupby(["arm", "pid"]).agg(n=("seated", "size"), seated=("seated", "mean"),
              broken=("broken", "mean"), csat=("contacts_frac", "mean")).round(2).to_string())
