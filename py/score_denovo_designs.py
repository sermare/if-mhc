#!/usr/bin/env python3
"""Score every RFD1 de-novo 10-mer design (OLD grind/ladder/promising + NEW rfd_denovo30), on the
register map, and write tables. Run anytime — it just re-reads the PDBs on disk.

For each design: superpose its MHC alpha1/alpha2 onto the common groove frame (committor/refs.npz),
map the peptide in, and report register-preserving Ca-RMSD to GIG (6AM5) and DRG (6AMU). Then:
  to_cognate = RMSD to the design's OWN crystal register ; to_other = RMSD to the opposite register.
  crossed    = to_other < 1.45 A (a true register crossing; never yet achieved).
TCR presence is detected from the actual chains (TCR V-region motifs), so the withTCR/noTCR split is
verified from the file, not the campaign name.

Usage:
  python py/score_denovo_designs.py                      # default dirs (old + new)
  python py/score_denovo_designs.py outputs/rfd_denovo30 # only these dir(s)
Outputs (under outputs/denovo_scores/):
  per_design.csv     one row per design (file-level, source-identifiable)
  summary.csv        per (source, crystal, context, conditioning): n, TCR?, to_cog/to_oth med+best, #crossed
Env: any python with numpy + pandas (e.g. esmfold2).
"""
import os, sys, glob, re
import numpy as np, pandas as pd

ROOT = "/home/ubuntu/if-mhc"
OUT = f"{ROOT}/outputs/denovo_scores"; os.makedirs(OUT, exist_ok=True)
CROSS = 2.5                                             # crossing threshold (Å)
AA = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
      'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
      'MSE':'M','HID':'H','HIE':'H','HIP':'H'}
_r = np.load(f"{ROOT}/outputs/committor/refs.npz")
GIG, DRG, REF_CA, REFSEQ = _r["GIG"], _r["DRG"], _r["REF_CA"], str(_r["REFseq"])
TCRA_MOTIF, TCRB_MOTIF, MHC_MOTIF, B2M_MOTIF = "EVEQNSGPL", "IAGITQAP", "SHSMRYFF", "MIQRTP"

def _RT(P, Q):
    Pc, Qc = P.mean(0), Q.mean(0)
    V, S, Wt = np.linalg.svd((P - Pc).T @ (Q - Qc)); d = np.sign(np.linalg.det(V @ Wt))
    R = V @ np.diag([1, 1, d]) @ Wt; return R, Qc - Pc @ R
def _robust(Po, Pn):
    R, t = _RT(Po, Pn)
    for _ in range(3):
        keep = np.linalg.norm(Po @ R + t - Pn, axis=1) < 3.0
        if keep.sum() < 30: break
        R, t = _RT(Po[keep], Pn[keep])
    return R, t
def _offset(so, sn):
    best = (-1, 0)
    for k in range(-8, 40):
        mv = sum(1 for i, c in enumerate(so) if 0 <= i + k < len(sn) and c == sn[i + k])
        if mv > best[0]: best = (mv, k)
    return best[1]

def _chains(path):
    ch = {}
    for l in open(path):
        if l.startswith("ATOM") and l[12:16].strip() == "CA":
            c = l[21]
            ch.setdefault(c, []).append((AA.get(l[17:20].strip(), 'x'),
                                         np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])))
    return ch
def _has_tcr(seq): return (TCRA_MOTIF in seq) or (TCRB_MOTIF in seq)

def _map_peptide(path):
    """Locate MHC, superpose onto the groove frame, and map the peptide Cα in.
    -> (pa, tcr, pep_len): pa is the (10,3) frame-aligned peptide Cα (or None if not a locatable
    10-mer); tcr is the TCR-present flag; pep_len is the detected peptide length. Returns None only
    when the MHC itself can't be located/aligned. Shared by score() and score_perres()."""
    ch = _chains(path)
    if not ch: return None
    allseq = "".join(n for c in ch for n, _ in ch[c])
    tcr = _has_tcr(allseq)
    # MHC chain (contains SHSMRYFF)
    mhc = None
    for c, rs in ch.items():
        s = "".join(n for n, _ in rs)
        if MHC_MOTIF in s: mhc = (c, rs, s); break
    if not mhc: return None
    c, rs, ms = mhc; mca = np.array([x for _, x in rs])
    m = ms.find(MHC_MOTIF); mhc_local = list(range(m, min(m + 179, len(rs))))
    k = _offset("".join(ms[i] for i in mhc_local), REFSEQ)
    idx = [j for j, i in enumerate(mhc_local) if 0 <= j + k < len(REF_CA)]
    if len(idx) < 50: return None
    R, t = _robust(mca[[mhc_local[j] for j in idx]], REF_CA[[j + k for j in idx]])
    # peptide: separate 8-12 chain (not MHC/b2m/TCR) preferred; else embedded 10-block in a merged chain
    pep = None
    for c2, rs2 in ch.items():
        if c2 == c: continue
        s2 = "".join(n for n, _ in rs2)
        if 8 <= len(rs2) <= 12 and B2M_MOTIF not in s2 and not _has_tcr(s2) and MHC_MOTIF not in s2:
            pep = np.array([x for _, x in rs2]); break
    if pep is None:                                       # merged single-chain fallback (MHC[+b2m][+TCR]+peptide)
        # our contigs always append the peptide LAST (…E1-120/0 C1-10), and RFd preserves that order
        # in the merged output chain (verified: tail == native peptide), so the peptide is the last 10 CA.
        s = "".join(n for n, _ in rs)
        tail = s[-10:]
        if len(rs) >= 190 and MHC_MOTIF not in tail and B2M_MOTIF not in tail and not _has_tcr(s[-30:]):
            pep = mca[-10:]
    pep_len = (len(pep) if pep is not None else 0)
    if pep is None or len(pep) != 10: return (None, tcr, pep_len)
    return (pep @ R + t, tcr, pep_len)

def score(path):
    """-> dict(toGIG,toDRG,tcr,pep_len) or None if MHC/peptide not locatable."""
    r = _map_peptide(path)
    if r is None: return None
    pa, tcr, pep_len = r
    if pa is None: return dict(toGIG=None, toDRG=None, tcr=tcr, pep_len=pep_len)
    return dict(toGIG=float(np.sqrt(((pa - GIG) ** 2).sum() / 10)),
                toDRG=float(np.sqrt(((pa - DRG) ** 2).sum() / 10)), tcr=tcr, pep_len=10)

def score_perres(path):
    """Per-position variant of score(): keeps the full 10-vector of per-residue Cα deviations instead
    of collapsing to a scalar. -> dict(devGIG[10], devDRG[10], tcr, pep_len) or None. The scalar RMSD
    is exactly sqrt(mean(devGIG**2)), so this is a strict super-set of score()."""
    r = _map_peptide(path)
    if r is None: return None
    pa, tcr, pep_len = r
    if pa is None: return dict(devGIG=None, devDRG=None, tcr=tcr, pep_len=pep_len)
    return dict(devGIG=np.linalg.norm(pa - GIG, axis=1),      # (10,)
                devDRG=np.linalg.norm(pa - DRG, axis=1),      # (10,)
                tcr=tcr, pep_len=10)

def _gen_date(path):
    """PDB generation date (file mtime) — outputs are written once and never modified afterward."""
    if not path: return None
    full = path if os.path.isabs(path) else os.path.join(ROOT, path)
    try:
        return pd.Timestamp(os.path.getmtime(full), unit="s").strftime("%Y-%m-%d")
    except OSError:
        return None

# ---- anchor occupancy (the register readout the RMSD proxy could never express) ----
# The register is a C-terminal anchor OCCUPANCY event: WHICH peptide Cα buries in the MHC F-pocket
# (p10 => GIG/6AM5 register, p9 => DRG/6AMU register). The N-anchor (P2, B-pocket) is shared by BOTH
# registers, so it seats the peptide but does NOT discriminate them. Both pocket centroids are fixed
# landmarks on the rigid MHC groove, derived straight from REF_CA so occupancy is read in the SAME
# frame as the peptide _map_peptide() returns. Validated on the crystals: GIG p10@4.5 Å, DRG p9@5.2 Å,
# each register's non-occupant anchor >=7.7 Å out — a clean gap the RMSD 'crossed' proxy never saw.
FPOCKET = [77, 80, 84, 116, 123, 143]          # HLA-A*02 F-pocket (C-terminal anchor pocket)
BPOCKET = [7, 9, 24, 45, 63, 66, 67, 99]       # HLA-A*02 B-pocket (P2 / N-terminal anchor pocket)
_MOFF = REFSEQ.find(MHC_MOTIF)                 # REFseq[_MOFF] == author residue 2 (SHSMRYFF)
def _refca(n):                                 # MHC author residue number -> REF_CA row (committor frame)
    j = _MOFF + (n - 2); return REF_CA[j] if 0 <= j < len(REF_CA) else None
FPOCKET_CENTROID = np.mean([c for c in (_refca(n) for n in FPOCKET) if c is not None], axis=0)
BPOCKET_CENTROID = np.mean([c for c in (_refca(n) for n in BPOCKET) if c is not None], axis=0)
SEAT_F, SEAT_B = 6.5, 7.0                       # burial-depth cutoffs (Å): native anchors <=6.1, next >=7.7

# ---- threading direction (reverse-threaded population filter) ----
# Both native registers anchor their F-pocket-proximal residue in the C-terminal half (GIG@P10, DRG@P9).
# With nothing to constrain peptide N->C direction, unconditioned RFdiffusion generates two discrete
# topological solutions at roughly equal odds: FORWARD (C-terminal anchor buries, like both natives) and
# REVERSE (N-terminal anchor buries instead — a mirror-image threading of the same backbone shape).
# argmin(F-pocket distance) histogram over the full de-novo corpus (n=1234) is cleanly bimodal with a
# valley at P5/P6 (30/23 designs) vs peaks at P1-3 (541) and P8-10 (503) -> P5/P6 is the split boundary.
# REVERSE designs are a structurally distinct population, not noise around the correct mode: they must be
# filtered out before computing register-recovery statistics, or they silently halve the effective N and
# pull every median toward ~15 Å regardless of how good the forward-threaded half actually is.
THREAD_REV, THREAD_FWD = (1, 4), (7, 10)        # P1-4 = reverse, P5-6 = ambiguous (excluded), P7-10 = forward

def threading(fpos):
    if THREAD_FWD[0] <= fpos <= THREAD_FWD[1]: return "forward"
    if THREAD_REV[0] <= fpos <= THREAD_REV[1]: return "reverse"
    return "ambiguous"

def occupancy(pa):
    """Anchor-occupancy readout for a (10,3) frame-aligned peptide Cα array (as returned by
    _map_peptide). RMSD-independent — reads the coordinate that DEFINES the register:
      fpocket_pos/dist : peptide position nearest the F-pocket centroid + its burial depth (Å)
                         -> 10 = GIG register, 9 = DRG register (register-defining)
      bpocket_pos/dist : same for the B-pocket (N-anchor P2; shared by both registers)
      f_seated/b_seated: burial depth within the seating cutoff
      register         : 'GIG' if F-pocket occupant is p10 & seated, 'DRG' if p9 & seated, else 'off'
      threading        : 'forward' (C-term anchors, like both natives) / 'reverse' (N-term anchors,
                         mirror-threaded) / 'ambiguous' (P5-6) — see THREAD_REV/THREAD_FWD above.
    """
    dF = np.linalg.norm(pa - FPOCKET_CENTROID, axis=1)
    dB = np.linalg.norm(pa - BPOCKET_CENTROID, axis=1)
    fpos, fdist = int(dF.argmin() + 1), float(dF.min())
    bpos, bdist = int(dB.argmin() + 1), float(dB.min())
    fseat, bseat = fdist <= SEAT_F, bdist <= SEAT_B
    reg = "GIG" if (fseat and fpos == 10) else "DRG" if (fseat and fpos == 9) else "off"
    return dict(fpocket_pos=fpos, fpocket_dist=round(fdist, 2), f_seated=bool(fseat),
                bpocket_pos=bpos, bpocket_dist=round(bdist, 2), b_seated=bool(bseat), register=reg,
                threading=threading(fpos))

def score_occ(path):
    """One-pass scorer returning BOTH the RMSD-to-register scalars AND the anchor-occupancy readout
    from a single _map_peptide() call. -> dict or None (None only if MHC unlocatable)."""
    r = _map_peptide(path)
    if r is None: return None
    pa, tcr, pep_len = r
    if pa is None: return dict(toGIG=None, toDRG=None, tcr=tcr, pep_len=pep_len)
    out = dict(toGIG=float(np.sqrt(((pa - GIG) ** 2).sum() / 10)),
               toDRG=float(np.sqrt(((pa - DRG) ** 2).sum() / 10)), tcr=tcr, pep_len=10)
    out.update(occupancy(pa))
    return out

# ---- campaign / cell parsing ----
COND_TOKS = ["mhc_tcr2", "mhc_tcr1", "L2_nterm_t1", "L3_nterm_t2", "L4_expanded", "L5_max",
             "L1_nterm", "mhc", "tcr2", "tcr1", "fixall", "fix8", "fix6", "fix4" ,"fix2", "fix0"]
def parse_meta(path):
    b = os.path.basename(path); src = os.path.basename(os.path.dirname(os.path.dirname(path)))
    pid = "6AMU" if ("6AMU" in b or b.startswith("DRG")) else "6AM5"
    if src == "rfd_denovo30":                             # {pid}_{context}_{cond}_{idx}.pdb
        m = re.match(r"(6AM5|6AMU)_(withTCR|noTCR)_(.+)_\d+\.pdb$", b)
        if m: return dict(source="denovo30", pid=m.group(1), context=m.group(2), cond=m.group(3))
    if src == "rfd_maxcond":                              # {pid}_{level}_{idx}.pdb (level=max|k24|k18..)
        m = re.match(r"(6AM5|6AMU)_(max|k\d+)_\d+\.pdb$", b)
        if m: return dict(source="maxcond", pid=m.group(1), context="withTCR", cond=m.group(2))
    if src == "rfd_xreg":                                 # {seed}_{level}_{idx}.pdb; cross-register on chimera
        m = re.match(r"((?:GIG|DRG)base_(?:GIG|DRG)cterm)_(xg_\w+?)_\d+\.pdb$", b)
        if m:                                             # pid = BASE scaffold crystal; to_other = TARGET register
            base = "6AM5" if m.group(1).startswith("GIG") else "6AMU"
            return dict(source="xreg", pid=base, context=m.group(1), cond=m.group(2))
    # old grind/ladder/promising: {pid}_{cond}_...pdb  (context always withTCR; verified via tcr flag)
    tok = next((t for t in COND_TOKS if f"_{t}_" in "_" + b), "?")
    return dict(source=src, pid=pid, context="withTCR", cond=tok)

def gather(dirs):
    files = []
    for d in dirs:
        files += [f for f in glob.glob(f"{d}/**/*.pdb", recursive=True)
                  if "_split" not in f and "traj" not in f]
    return sorted(set(files))

def main():
    dirs = sys.argv[1:] or [f"{ROOT}/outputs/{x}/pdb" for x in ("grind", "ladder", "promising","rfd_maxcond","fairn", "deep")] + \
                            [f"{ROOT}/outputs/rfd_denovo30/pdb", f"{ROOT}/outputs/rfd_maxcond/pdb"]
    dirs = [d for d in dirs if os.path.isdir(d)]
    files = gather(dirs)
    rows = []
    for f in files:
        sc = score_occ(f)
        if sc is None: continue
        meta = parse_meta(f)
        toG, toD = sc["toGIG"], sc["toDRG"]
        rec = dict(**meta, tcr_present=sc["tcr"], pep_len=sc["pep_len"],
                   toGIG=None if toG is None else round(toG, 3),
                   toDRG=None if toD is None else round(toD, 3), file=f.replace(ROOT + "/", ""))
        if toG is not None:
            cog = toG if meta["pid"] == "6AM5" else toD
            oth = toD if meta["pid"] == "6AM5" else toG
            want = 10 if meta["pid"] == "6AM5" else 9      # this crystal's F-pocket occupant
            othr = 9 if meta["pid"] == "6AM5" else 10      # the opposite register's occupant
            rec.update(to_cognate=round(cog, 3), to_other=round(oth, 3), crossed=bool(oth < CROSS),
                       # ---- occupancy (primary axis) ----
                       fpocket_pos=sc["fpocket_pos"], fpocket_dist=sc["fpocket_dist"],
                       bpocket_pos=sc["bpocket_pos"], bpocket_dist=sc["bpocket_dist"],
                       f_seated=sc["f_seated"], b_seated=sc["b_seated"], register=sc["register"],
                       threading=sc["threading"],
                       # seats its OWN crystal's F-pocket anchor (the real goal); occupancy-based crossing
                       occ_cognate=bool(sc["f_seated"] and sc["fpocket_pos"] == want),
                       occ_crossed=bool(sc["f_seated"] and sc["fpocket_pos"] == othr))
        else:
            rec.update(to_cognate=None, to_other=None, crossed=None, fpocket_pos=None,
                       fpocket_dist=None, bpocket_pos=None, bpocket_dist=None, f_seated=None,
                       b_seated=None, register=None, threading=None, occ_cognate=None, occ_crossed=None)
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/per_design.csv", index=False)
    scored = df[df.toGIG.notna()].copy()
    # summary per (source, pid, context, cond)
    g = scored.groupby(["source", "pid", "context", "cond"])
    summ = g.agg(n=("file", "size"),
                 tcr=("tcr_present", lambda s: "yes" if s.any() else "no"),
                 # occupancy aggregates (primary): how many seat the cognate F-pocket anchor, wrong one, any
                 occ_cognate=("occ_cognate", "sum"), occ_crossed=("occ_crossed", "sum"),
                 f_seated=("f_seated", "sum"), b_seated=("b_seated", "sum"),
                 fpocket_best=("fpocket_dist", "min"),
                 to_cog_med=("to_cognate", "median"), to_cog_best=("to_cognate", "min"),
                 to_oth_med=("to_other", "median"), to_oth_best=("to_other", "min"),
                 crossed=("crossed", "sum")).round(2).reset_index()
    # path to the best peptide in each row: closest-to-OTHER (crossing champion) + closest-to-cognate (best seated)
    best = g.apply(lambda gg: pd.Series({
        "to_oth_best_file": gg.loc[gg["to_other"].idxmin(), "file"],
        "to_cog_best_file": gg.loc[gg["to_cognate"].idxmin(), "file"],
    })).reset_index()
    summ = summ.merge(best, on=["source", "pid", "context", "cond"], how="left")
    summ = summ.sort_values(["source", "pid", "context", "cond"]).reset_index(drop=True)
    summ.to_csv(f"{OUT}/summary.csv", index=False)
    # ---- report ----
    print(f"\nscanned {len(files)} PDBs across {len(dirs)} dir(s); scored {len(scored)} 10-mer designs "
          f"({len(df) - len(scored)} unparsed/non-10mer)")
    print(f"TOTAL designs by source:")
    print(df.groupby("source").size().to_string())
    # ---- PRIMARY AXIS: anchor occupancy (reads the register; the RMSD proxy below never could) ----
    print(f"\n=== ANCHOR OCCUPANCY (primary axis) — n={len(scored)} designs ===")
    print(f"  F-pocket seated (any register, <= {SEAT_F} Å): {int(scored.f_seated.sum())}"
          f"  |  B-pocket/P2 seated (<= {SEAT_B} Å): {int(scored.b_seated.sum())}")
    print(f"  seats its OWN crystal's F-pocket anchor (occ_cognate): {int(scored.occ_cognate.sum())}"
          f"  |  seats the WRONG register (occ_crossed): {int(scored.occ_crossed.sum())}")
    print("  F-pocket occupant position distribution (10=GIG, 9=DRG, else off-register):")
    print("   " + scored.fpocket_pos.value_counts().sort_index().to_string().replace("\n", "\n   "))
    print("\n  THREADING (forward = C-term anchors like both natives; reverse = mirror-threaded; "
          "see THREAD_FWD/THREAD_REV):")
    print("   " + scored.threading.value_counts().to_string().replace("\n", "\n   "))
    print(f"\n  [legacy RMSD proxy] crossings (to_other < {CROSS} Å): {int(scored.crossed.sum())} / {len(scored)}")
    if scored.crossed.sum():
        print(scored[scored.crossed][["source", "pid", "context", "cond", "to_other", "file"]].sort_values(by='to_other').to_string(index=False))
    print("\n=== summary (per source × crystal × context × conditioning) — 'best_oth_pep' = closest-to-OTHER design ===")
    print("\n=== summary (per source × crystal × context × conditioning) — 'best_oth_pep' = closest-to-OTHER design ===")
    disp = summ.copy()
    disp["best_oth_pep"] = disp["to_oth_best_file"].map(os.path.basename)   # full paths are in summary.csv
    disp["best_oth_date"] = disp["to_oth_best_file"].map(_gen_date)         # when that design was generated
    disp = disp.drop(columns=["to_oth_best_file", "to_cog_best_file"])
    with pd.option_context("display.max_rows", None, "display.width", 240, "display.max_colwidth", 60):
        print(disp.sort_values(by=["to_oth_best"]).head(10).to_string(index=False))
    print(f"\nwrote {OUT}/per_design.csv  and  {OUT}/summary.csv "
          f"(summary.csv has full to_oth_best_file + to_cog_best_file paths)")

if __name__ == "__main__":
    main()
