#!/usr/bin/env python3
"""Component-wise structural distance between the two native pMHC-TCR complexes (6AM5/GIG vs 6AMU/DRG)
and across their unbiased MD ensembles (300 K & 370 K). For every structural sub-object -- whole complex,
MHC, MHC groove (a1a2), MHC alpha3, the beta-sheet groove floor ("basin"), the F-pocket, the peptide,
the whole TCR, TCRa, TCRb, and the six CDR loops -- we report Ca-RMSD under two frames:

  * CONFORMATION  : the object is Kabsch-superposed onto its counterpart by its OWN atoms
                    -> pure internal shape difference (rigidity check).
  * GROOVE (pose) : everything is superposed on the invariant MHC a1a2 Ca (the register-preserving
                    "common groove frame" the whole project uses) then measured with NO further fit
                    -> where the object physically sits relative to the groove. For the peptide this
                    reproduces the canonical 2.87 A GIG<->DRG register separation.

CDR loops are framework-superposed (V-domain minus CDR3) as in plot_tcr_pose_space.py.

Residue addressing is anchored on conserved sequence motifs so it is identical for crystals and the
merged-chain MD topology:  MHC on "SHSMRYFF" (its S = HLA author resid 2), chains split by motif.

Env: esmfold2 python (mdtraj 1.11, biopython 1.87). Writes CSVs + PNGs into outputs/native_md_rmsd/.
"""
import os, re, glob, warnings, json
warnings.filterwarnings("ignore")
import numpy as np
import mdtraj as mdt
from Bio.PDB import PDBParser

ROOT = "/home/ubuntu/if-mhc"
OUT = f"{ROOT}/outputs/native_md_rmsd"; os.makedirs(OUT, exist_ok=True)
AA = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
      'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
      'MSE':'M','HID':'H','HIE':'H','HIP':'H'}

# ------------------------------------------------------------------ geometry
def kabsch(P, Q):
    """R,t mapping P onto Q (both (n,3))."""
    Pc, Qc = P.mean(0), Q.mean(0)
    V, S, Wt = np.linalg.svd((P - Pc).T @ (Q - Qc))
    d = np.sign(np.linalg.det(V @ Wt))
    R = V @ np.diag([1, 1, d]) @ Wt
    return R, Qc - Pc @ R

def rmsd(P, Q):
    return float(np.sqrt(((P - Q) ** 2).sum() / len(P)))

def sup_rmsd(P, Q):
    """conformational RMSD: superpose P onto Q by their own atoms."""
    R, t = kabsch(P, Q); return rmsd(P @ R + t, Q)

from Bio.Align import PairwiseAligner
_ALN = PairwiseAligner()
_ALN.mode = "global"; _ALN.match_score = 2; _ALN.mismatch_score = -1
_ALN.open_gap_score = -5; _ALN.extend_gap_score = -0.5

def align_map(query, ref):
    """query position -> ref position, via global alignment (near-identical seqs -> gap-aware 1:1)."""
    if not query or not ref: return {}
    aln = _ALN.align(query, ref)[0]
    qm = {}
    for (q0, q1), (r0, r1) in zip(aln.aligned[0], aln.aligned[1]):
        for k in range(q1 - q0):
            qm[q0 + k] = r0 + k
    return qm

# ------------------------------------------------------------------ a residue-CA container
class Model:
    """Uniform CA container keyed by (chain, CANONICAL residue label). Canonical labels come from
    aligning every chain's sequence to the 6AM5 crystal reference frame, so cross-comparison is a
    gap-aware 1:1 correspondence regardless of missing residues or author renumbering."""
    def __init__(self, seq, ca_by_resindex, chain_seqs):
        self.seq = seq                    # full concatenated 1-letter sequence (residue order)
        self.ca = ca_by_resindex          # dict resindex -> (3,) CA coord
        self.chain_seqs = chain_seqs      # dict logical_chain -> (start_resindex, seq_str)
        self.canon = {}                   # (chain, ref_pos) -> resindex, filled by _set_canon

    def _set_canon(self, ref_chain_seqs):
        for ch, (start, sq) in self.chain_seqs.items():
            if ch not in ref_chain_seqs: continue
            m = align_map(sq, ref_chain_seqs[ch][1])
            for qpos, rpos in m.items():
                self.canon[(ch, rpos)] = start + qpos
        return self

# motif anchors ------------------------------------------------------
MHC_MOTIF = "SHSMRYFF"     # S = HLA author resid 2
B2M_MOTIF = "MIQRTP"
TCRA_MOTIF = "EVEQNSGPL"   # DMF5 TCR alpha V-region start (as used across the repo)
TCRB_MOTIF = "IAGITQAP"

PEP_SEQS = ["SMLGIGIVPV", "MMWDRGLGMM"]   # native GIG / DRG 10-mers (register study system)

def segment(seq):
    """Return logical-chain start indices in a concatenated residue sequence."""
    mhc = seq.find(MHC_MOTIF)
    b2m = seq.find(B2M_MOTIF, mhc if mhc >= 0 else 0)
    ta = seq.find(TCRA_MOTIF, b2m if b2m >= 0 else 0)
    tb = seq.find(TCRB_MOTIF, ta if ta >= 0 else 0)
    # MHC start: motif S is author-2, so residue author-1 (leading G) is one before if present
    mhc0 = mhc - 1 if (mhc > 0 and seq[mhc - 1] == "G") else mhc
    # peptide: locate by native sequence (robust); fall back to "10 before TCRa motif"
    pep0 = -1
    for ps in PEP_SEQS:
        j = seq.find(ps, b2m if b2m >= 0 else 0)
        if 0 <= j < ta:
            pep0 = j; break
    if pep0 < 0: pep0 = ta - 10
    return dict(mhc=mhc0, mhc_motif=mhc, b2m=b2m, pep=pep0, tcra=ta, tcrb=tb)

# ------------------------------------------------------------------ loaders
_p = PDBParser(QUIET=True)

def load_crystal(pid):
    """Crystal -> Model. Chains A(MHC) B(b2m) C(pep) D(TCRa) E(TCRb)."""
    m = _p.get_structure(pid, f"{ROOT}/inputs/focus_6am/{pid}.pdb")[0]
    seq_parts, ca, idx = [], {}, 0
    chain_seqs = {}
    order = [("A", "mhc"), ("B", "b2m"), ("C", "pep"), ("D", "tcra"), ("E", "tcrb")]
    for ch, name in order:
        if ch not in m: continue
        start = idx; s = ""
        for r in m[ch]:
            if r.id[0] != ' ' or 'CA' not in r: continue
            s += AA.get(r.resname, 'x'); ca[idx] = r['CA'].coord.astype(float); idx += 1
        chain_seqs[name] = (start, s); seq_parts.append(s)
    return Model("".join(seq_parts), ca, chain_seqs)

# 6AM5 crystal is the canonical author frame (complete chains, MHC author 1..275)
_REF = None
def ref_chain_seqs():
    global _REF
    if _REF is None:
        _REF = {k: v for k, v in load_crystal("6AM5").chain_seqs.items()}
    return _REF

def load_md(job, nframes=120):
    """MD trajectory -> list[Model], evenly subsampled to nframes."""
    d = f"{ROOT}/outputs/tamarind/results/{job}"
    t = mdt.load(f"{d}/traj_prod_no_water_seg1.xtc", top=f"{d}/topology_no_water.pdb")
    res = list(t.topology.residues)
    seq = "".join(AA.get(r.name, 'x') for r in res)
    seg = segment(seq)
    # CA atom index per residue index
    ca_atom = {a.residue.index: a.index for a in t.topology.atoms if a.name == "CA"}
    resids = sorted(ca_atom)
    xyz = t.xyz * 10.0  # nm -> A
    # logical chain sequences (start resindex, seq) for CDR detection etc.
    pend = seg["pep"] + 10                       # peptide is exactly 10 residues
    chain_seqs = {
        "mhc": (seg["mhc"], seq[seg["mhc"]:seg["b2m"]]),
        "b2m": (seg["b2m"], seq[seg["b2m"]:seg["pep"]]),
        "pep": (seg["pep"], seq[seg["pep"]:pend]),
        "tcra": (pend, seq[pend:seg["tcrb"]]),   # TCRa chain = after peptide .. TCRb motif
        "tcrb": (seg["tcrb"], seq[seg["tcrb"]:len(res)]),
    }
    sel = np.linspace(0, t.n_frames - 1, min(nframes, t.n_frames)).astype(int)
    rcs = ref_chain_seqs()
    models = []
    for fi in sel:
        ca = {ri: xyz[fi, ca_atom[ri], :] for ri in resids}
        models.append(Model(seq, ca, chain_seqs)._set_canon(rcs))
    return models, seg

# ------------------------------------------------------------------ component residue selection
# MHC author residues: 6AM5 crystal chain A starts at author 1 -> canonical ref_pos = author - 1
def mhc_ca(model, authors):
    ca = model.ca; out = {}
    for a in authors:
        ri = model.canon.get(("mhc", a - 1))
        if ri is not None and ri in ca: out[a] = ca[ri]
    return out

FPOCKET = [77, 80, 84, 116, 123, 143]              # F-pocket floor set used across the repo (design_md)
FPOCKET_EXT = [77, 80, 81, 84, 95, 116, 123, 143, 146, 147]   # canonical extended HLA-A2 F-pocket
# beta-sheet groove floor ("basin"): the 8 strands under the peptide (HLA author numbering, a1 + a2)
FLOOR = ([r for r in range(3, 13)] + [r for r in range(21, 29)] + [r for r in range(31, 38)] +
         [r for r in range(46, 54)] + [r for r in range(94, 104)] + [r for r in range(109, 119)] +
         [r for r in range(121, 128)] + [r for r in range(133, 139)])
A1A2 = list(range(2, 181))                          # groove helices+floor (superposition frame)
A3 = list(range(181, 276))

def comp_ca(model, kind):
    """Return CA for a component as dict keyed by a MODEL-INDEPENDENT canonical label so that
    intersecting two models' dicts gives correct 1:1 residue correspondence:
        MHC components -> key = HLA author resid (int)
        peptide        -> key = ("P", position 0..9)
        TCR chains     -> key = (chain, offset within that chain)."""
    ca = model.ca
    if kind == "a1a2":     return mhc_ca(model, A1A2)
    if kind == "a3":       return mhc_ca(model, A3)
    if kind == "floor":    return mhc_ca(model, FLOOR)
    if kind == "fpocket":  return mhc_ca(model, FPOCKET)
    if kind == "fpocket_ext": return mhc_ca(model, FPOCKET_EXT)
    if kind == "mhc":      return mhc_ca(model, list(range(2, 276)))
    if kind == "pep":
        s, sq = model.chain_seqs["pep"]
        return {("P", i): ca[s + i] for i in range(len(sq)) if s + i in ca}
    if kind in ("tcra", "tcrb"):
        return {(kind, rp): ca[ri] for (ch, rp), ri in model.canon.items()
                if ch == kind and ri in ca}
    if kind == "tcr":
        return {(ch, rp): ca[ri] for (ch, rp), ri in model.canon.items()
                if ch in ("tcra", "tcrb") and ri in ca}
    if kind == "tcr_v":     # Vα + Vβ variable domains only (comparable to plot_tcr_pose_space)
        return {(ch, rp): ca[ri] for (ch, rp), ri in model.canon.items()
                if ch in ("tcra", "tcrb") and rp < VDOM and ri in ca}
    raise ValueError(kind)

# ---- CDR loops (framework-superposed) ------------------------------
def find_cdr3(seq):
    """CDR3 residue offsets within a V-domain seq: conserved 2nd Cys ... just before J-motif [FW]G.G."""
    ms = list(re.finditer("[FW]G.G", seq))
    if not ms: return []
    end = ms[-1].start(); cys = seq.rfind("C", 0, end)
    if cys < 0: return []
    return list(range(cys + 1, end))               # loop between Cys and J-motif

def find_cdr12(seq):
    """Approximate CDR1/CDR2 offsets anchored on the first conserved Cys (~pos 22, IMGT Cys23).
    CDR1 ~ Cys+2..Cys+9 ; CDR2 ~ Cys+34..Cys+42 (IMGT canonical spacing). Heuristic."""
    c1 = seq.find("C")
    if c1 < 0 or c1 > 40: return [], []
    cdr1 = list(range(c1 + 2, c1 + 10))
    cdr2 = list(range(c1 + 34, c1 + 43))
    return cdr1, cdr2

VDOM = 120                                     # V-domain length (contains all CDRs)
def vdomain_ca(model, chain, ndom=VDOM):
    """First ndom residues of a TCR chain = the V-domain (contains all CDRs). Returns (seq, [ca...])."""
    s, sq = model.chain_seqs[chain]
    sq = sq[:ndom]
    cas = [model.ca[s + i] for i in range(len(sq)) if s + i in model.ca]
    return sq, np.array(cas), s

def loop_positions(chain):
    """CDR loop offsets in the 6AM5 REFERENCE V-domain (canonical ref positions, gap-aware). Detecting
    on the shared reference — not per-structure — guarantees the loop is the *same residues* in both."""
    refseq = ref_chain_seqs()[chain][1][:VDOM]
    c1, c2 = find_cdr12(refseq)
    return {"cdr1": c1, "cdr2": c2, "cdr3": find_cdr3(refseq)}

# ------------------------------------------------------------------ pairwise distances
def groove_frame(model_ref, model_mov):
    """Kabsch mapping model_mov's a1a2 -> model_ref's a1a2 (common groove frame), on shared residues."""
    a = comp_ca(model_ref, "a1a2"); b = comp_ca(model_mov, "a1a2")
    keys = [k for k in a if k in b]  # dict-ordered; keys are mixed int/tuple
    P = np.array([b[k] for k in keys]); Q = np.array([a[k] for k in keys])
    R, t = kabsch(P, Q); return R, t, rmsd(P @ R + t, Q)

def comp_dist(model_ref, model_mov, kind, mode):
    """Ca-RMSD of a component between two models.
       mode='conf' : superpose on the component's own CA.
       mode='groove': superpose on a1a2 then direct RMSD (pose/register)."""
    a = comp_ca(model_ref, kind); b = comp_ca(model_mov, kind)
    keys = [k for k in a if k in b]  # dict-ordered; keys are mixed int/tuple
    if len(keys) < 3: return None
    P = np.array([b[k] for k in keys]); Q = np.array([a[k] for k in keys])
    if mode == "conf":
        return sup_rmsd(P, Q)
    R, t, _ = groove_frame(model_ref, model_mov)
    return rmsd(P @ R + t, Q)

def cdr_dist(model_ref, model_mov, chain, loop_kind):
    """Framework-superposed CDR-loop Cα-RMSD (pose removed, internal loop shape only). Residues are
    matched by CANONICAL ref position (comp_ca keys), so 6AM5↔6AMU pairs the same residues even with
    gaps / author renumbering. Framework = V-domain minus CDR3; loop = the requested CDR."""
    loops = loop_positions(chain)
    cdr3 = set(loops["cdr3"]); loop_pos = set(loops[loop_kind])
    A = comp_ca(model_ref, chain); B = comp_ca(model_mov, chain)   # keys (chain, ref_pos)
    fw = [(chain, p) for p in range(VDOM) if p not in cdr3 and (chain, p) in A and (chain, p) in B]
    loop = [(chain, p) for p in sorted(loop_pos) if (chain, p) in A and (chain, p) in B]
    if len(fw) < 10 or len(loop) < 2: return None
    P = np.array([B[k] for k in fw]); Q = np.array([A[k] for k in fw])
    R, t = kabsch(P, Q)
    Pl = np.array([B[k] for k in loop]) @ R + t; Ql = np.array([A[k] for k in loop])
    return rmsd(Pl, Ql)

COMPONENTS = [
    ("Whole complex", "complex", "conf"),
    ("MHC (a1a2a3)", "mhc", "conf"),
    ("MHC groove a1a2", "a1a2", "conf"),
    ("MHC alpha3", "a3", "conf"),
    ("MHC groove floor (basin)", "floor", "conf"),
    ("MHC F-pocket (6)", "fpocket", "conf"),
    ("MHC F-pocket (ext 10)", "fpocket_ext", "conf"),
    ("Peptide (conformation)", "pep", "conf"),
    ("Peptide (in-groove/register)", "pep", "groove"),
    ("Whole TCR", "tcr", "conf"),
    ("TCR alpha", "tcra", "conf"),
    ("TCR beta", "tcrb", "conf"),
    ("TCR Vdom (conformation)", "tcr_v", "conf"),
    ("TCR Vdom (docking pose, groove frame)", "tcr_v", "groove"),
    ("TCR (docking pose, groove frame)", "tcr", "groove"),
]
CDRS = [("CDR1a", "tcra", "cdr1"), ("CDR2a", "tcra", "cdr2"), ("CDR3a", "tcra", "cdr3"),
        ("CDR1b", "tcrb", "cdr1"), ("CDR2b", "tcrb", "cdr2"), ("CDR3b", "tcrb", "cdr3")]

def complex_ca(model):
    """All CA across MHC+pep+TCR (skip b2m for stability) as dict for whole-complex conf RMSD."""
    d = {}
    for k in ("mhc", "pep", "tcr"):
        d.update(comp_ca(model, k))
    return d

def comp_dist_complex(a_model, b_model, mode="conf"):
    a = complex_ca(a_model); b = complex_ca(b_model)
    keys = [k for k in a if k in b]  # dict-ordered; keys are mixed int/tuple
    P = np.array([b[k] for k in keys]); Q = np.array([a[k] for k in keys])
    return sup_rmsd(P, Q)

def distance(ref, mov, kind, mode):
    if kind == "complex": return comp_dist_complex(ref, mov, mode)
    return comp_dist(ref, mov, kind, mode)

# ------------------------------------------------------------------ full drivers
MD_JOBS = [("6AM5", "300K", "ifmhc_6AM5_md_300K"), ("6AM5", "370K", "ifmhc_6AM5_md_370K"),
           ("6AMU", "300K", "ifmhc_6AMU_md_300K"), ("6AMU", "370K", "ifmhc_6AMU_md_370K")]

def crystal_table(C5, CU):
    """6AM5 vs 6AMU per-component Ca-RMSD (both modes) + CDR loops."""
    rows = []
    for name, kind, mode in COMPONENTS:
        rows.append(dict(component=name, kind=kind, mode=mode, rmsd=round(distance(C5, CU, kind, mode), 2)))
    for name, chain, lk in CDRS:
        rows.append(dict(component=name, kind=f"{chain}:{lk}", mode="framework",
                         rmsd=round(cdr_dist(C5, CU, chain, lk), 2)))
    return rows

def md_series(models, C5, CU, kind, mode):
    """toGIG/toDRG Ca-RMSD of a component for every MD frame model."""
    g = np.array([distance(C5, m, kind, mode) for m in models], float)
    d = np.array([distance(CU, m, kind, mode) for m in models], float)
    return g, d

def md_cdr_series(models, C5, CU, chain, lk):
    g = np.array([cdr_dist(C5, m, chain, lk) for m in models], float)
    d = np.array([cdr_dist(CU, m, chain, lk) for m in models], float)
    return g, d

def summ(a):
    a = a[np.isfinite(a)]
    if len(a) == 0: return (np.nan, np.nan, np.nan)
    return (float(a.mean()), float(np.median(a)), float(a.std()))

if __name__ == "__main__":
    import csv
    print("loading crystals ...")
    rcs = ref_chain_seqs()
    C5 = load_crystal("6AM5")._set_canon(rcs); CU = load_crystal("6AMU")._set_canon(rcs)
    ctab = crystal_table(C5, CU)
    with open(f"{OUT}/crystal_component_rmsd.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["component", "kind", "mode", "rmsd"]); w.writeheader(); w.writerows(ctab)
    print("\n=== CRYSTAL 6AM5 vs 6AMU (wrote crystal_component_rmsd.csv) ===")
    for r in ctab: print(f"  {r['component']:38s} [{r['mode']:9s}] {r['rmsd']:.2f} A")

    print("\nloading MD (120 frames/traj) ...")
    NF = int(os.environ.get("NF", "120"))
    MD = {}
    for pid, T, job in MD_JOBS:
        models, _ = load_md(job, NF); MD[(pid, T)] = models
        print(f"  {job}: {len(models)} frames")
    # per-frame series -> long CSV + summary CSV
    ALL = [(n, k, m) for n, k, m in COMPONENTS] + [(n, f"{c}:{l}", "framework") for n, c, l in CDRS]
    long_rows, summ_rows = [], []
    for pid, T, job in MD_JOBS:
        models = MD[(pid, T)]
        for name, kind, mode in ALL:
            if mode == "framework":
                chain, lk = kind.split(":"); g, d = md_cdr_series(models, C5, CU, chain, lk)
            else:
                g, d = md_series(models, C5, CU, kind, mode)
            for i in range(len(models)):
                long_rows.append(dict(traj=f"{pid}_{T}", pid=pid, T=T, component=name, mode=mode,
                                      frame=i, toGIG=round(float(g[i]), 3), toDRG=round(float(d[i]), 3)))
            gm = summ(g); dm = summ(d)
            summ_rows.append(dict(traj=f"{pid}_{T}", pid=pid, T=T, component=name, mode=mode,
                                  toGIG_mean=round(gm[0], 2), toGIG_med=round(gm[1], 2), toGIG_sd=round(gm[2], 2),
                                  toDRG_mean=round(dm[0], 2), toDRG_med=round(dm[1], 2), toDRG_sd=round(dm[2], 2)))
    with open(f"{OUT}/md_component_rmsd_frames.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(long_rows[0].keys())); w.writeheader(); w.writerows(long_rows)
    with open(f"{OUT}/md_component_rmsd_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ_rows[0].keys())); w.writeheader(); w.writerows(summ_rows)
    print(f"wrote md_component_rmsd_frames.csv ({len(long_rows)} rows) + summary ({len(summ_rows)} rows)")
    print("\nsanity — peptide (in-groove) toGIG/toDRG per traj (cf. summary: 6AM5-300K 0.40/2.48):")
    for r in summ_rows:
        if r["component"] == "Peptide (in-groove/register)":
            print(f"  {r['traj']}: toGIG {r['toGIG_mean']:.2f} toDRG {r['toDRG_mean']:.2f}")
