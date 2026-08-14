#!/usr/bin/env python3
"""Stage non-SKEMPI pMHC-TCR structures into the same two-arm layout.

Datasets:
  pmhc25    the pre-existing inputs/pmhc_tcr_dataset structures
  focus6am  the 6AM5 (GIG) and 6AMU (DRG) reference crystals this project is
            built around; 6AMT is excluded because it holds two pMHC copies and
            no TCR, so its no-TCR arm would be identical to its TCR arm

Same two arms and same epitope-only design scope as skempi_prep.py, so the two
datasets are directly comparable:
  full   -- epitope + MHC-side chains + TCR chains
  notcr  -- epitope + MHC-side chains

Unlike the SKEMPI set there is no partner-chain annotation to lean on, and chain
letters are not consistent (3GSN is H/P/L/A/B, not A/B/C/D/E), so roles are
assigned by length band and then disambiguated by contact count. Eight of these
files hold two copies of the complex in the asymmetric unit; picking each role
by "most contacts to the chain already chosen" keeps the copy that actually
contains the annotated epitope chain instead of mixing the two.

Writes inputs/pmhc25/{manifest.csv, arm_full/, arm_notcr/}.
"""
import argparse, os, csv, warnings
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.Polypeptide import three_to_index, index_to_one

warnings.filterwarnings("ignore")
ROOT = "/global/scratch/users/sergiomar10/if-mhc"
DATASETS = {
    "pmhc25":   {"src": f"{ROOT}/inputs/pmhc_tcr_dataset",
                 "dst": f"{ROOT}/inputs/pmhc25", "spec": "dataset.csv"},
    "focus6am": {"src": f"{ROOT}/inputs/focus_6am",
                 "dst": f"{ROOT}/inputs/focus6am", "spec": None,
                 "pdbs": ["6AM5", "6AMU"]},
}

AA3 = {"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU","LYS",
       "MET","PHE","PRO","SER","THR","TRP","TYR","VAL"}
MODRES = {"MSE": "MET", "F2F": "PHE"}
KEEP_ATOMS = {"N", "CA", "C", "O", "CB"}
CONTACT_A = 5.0

# length bands (residues) for class-I pMHC-TCR
MHC_BAND = (190, 300)     # heavy chain; overlaps TCR-beta, resolved by contacts
B2M_BAND = (85, 120)
TCR_BAND = (100, 260)


def resseq(res):
    n = MODRES.get(res.get_resname(), res.get_resname())
    return index_to_one(three_to_index(n)) if n in AA3 else None


def chain_residues(chain):
    return [r for r in chain
            if (r.id[0] == " " or r.get_resname() in MODRES) and resseq(r) is not None]


def normalize_modres(model):
    changed = []
    for chain in model:
        for res in list(chain):
            name = res.get_resname()
            if name not in MODRES:
                continue
            for atom in list(res):
                if atom.get_id() not in KEEP_ATOMS:
                    res.detach_child(atom.get_id())
            res.resname = MODRES[name]
            if res.id[0] != " ":
                res.id = (" ", res.id[1], res.id[2])
            changed.append(f"{chain.id}{res.id[1]}:{name}->{MODRES[name]}")
    return changed


def coords_of(model, cid):
    return np.array([a.coord for a in model[cid].get_atoms() if a.element != "H"])


def n_contacts(a, b):
    """Heavy-atom pairs within CONTACT_A between two coordinate sets."""
    if len(a) == 0 or len(b) == 0:
        return 0
    tot = 0
    for i in range(0, len(a), 2000):                     # chunked to bound memory
        d = np.linalg.norm(a[i:i + 2000, None, :] - b[None, :, :], axis=-1)
        tot += int((d < CONTACT_A).sum())
    return tot


class ArmSelect(Select):
    def __init__(self, chains):
        self.chains = set(chains)

    def accept_model(self, model):
        return model.id == 0

    def accept_chain(self, chain):
        return chain.id in self.chains

    def accept_residue(self, res):
        return res.get_resname() not in ("HOH", "DOD")

    def accept_atom(self, atom):
        return atom.element != "H" and atom.get_altloc() in (" ", "A")


def epitope_chain(model):
    """Shortest polymer chain in the groove-peptide size range."""
    lens = {c.id: len(chain_residues(c)) for c in model}
    cand = {c: L for c, L in lens.items() if 5 <= L <= 25}
    return min(cand, key=lambda c: cand[c]) if cand else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="pmhc25", choices=sorted(DATASETS))
    a = ap.parse_args()
    cfg = DATASETS[a.dataset]
    global SRC, DST
    SRC, DST = cfg["src"], cfg["dst"]
    for d in ("arm_full", "arm_notcr"):
        os.makedirs(f"{DST}/{d}", exist_ok=True)

    if cfg["spec"]:
        ds = pd.read_csv(f"{SRC}/{cfg['spec']}")
    else:
        # no annotation file: detect the epitope chain from the structure
        recs = []
        for pdb in cfg["pdbs"]:
            m = PDBParser(QUIET=True).get_structure(pdb, f"{SRC}/{pdb}.pdb")[0]
            normalize_modres(m)
            pc = epitope_chain(m)
            recs.append({"pdb": pdb, "pep_chain": pc, "valid": True,
                         "peptide": "".join(resseq(r) for r in chain_residues(m[pc]))})
        ds = pd.DataFrame(recs)
    parser, io, rows, skipped = PDBParser(QUIET=True), PDBIO(), [], []

    for _, d in ds.iterrows():
        pdb, pep_ch = str(d["pdb"]).upper(), str(d["pep_chain"])
        path = f"{SRC}/{pdb}.pdb"
        if not os.path.exists(path):
            skipped.append((pdb, "file missing"))
            continue
        model = parser.get_structure(pdb, path)[0]
        rewritten = normalize_modres(model)
        lens = {c.id: len(chain_residues(c)) for c in model}
        lens = {k: v for k, v in lens.items() if v > 0}
        if pep_ch not in lens:
            skipped.append((pdb, f"epitope chain {pep_ch} absent"))
            continue

        pep_res = chain_residues(model[pep_ch])
        pep_seq = "".join(resseq(r) for r in pep_res)
        pc = coords_of(model, pep_ch)

        # MHC heavy: the long chain burying the epitope. TCR-beta falls in the
        # same length band but contacts the epitope far less than the groove does
        cands = [c for c, L in lens.items() if c != pep_ch and MHC_BAND[0] <= L <= MHC_BAND[1]]
        if not cands:
            skipped.append((pdb, "no MHC-heavy candidate"))
            continue
        mhc = max(cands, key=lambda c: n_contacts(pc, coords_of(model, c)))
        mc = coords_of(model, mhc)

        # B2M: the short chain packed against this copy's heavy chain
        b2m_c = [c for c, L in lens.items() if c not in (pep_ch, mhc) and B2M_BAND[0] <= L <= B2M_BAND[1]]
        b2m = max(b2m_c, key=lambda c: n_contacts(mc, coords_of(model, c))) if b2m_c else None

        # TCR: the two remaining chains sitting on this copy's epitope+groove
        taken = {pep_ch, mhc} | ({b2m} if b2m else set())
        tcr_c = [c for c, L in lens.items() if c not in taken and TCR_BAND[0] <= L <= TCR_BAND[1]]
        scored = sorted(((n_contacts(pc, coords_of(model, c)) +
                          n_contacts(mc, coords_of(model, c)), c) for c in tcr_c),
                        reverse=True)
        tcr = sorted(c for s, c in scored[:2] if s > 0)
        if not tcr:
            skipped.append((pdb, "no TCR in contact -- no-TCR arm would be a no-op"))
            continue

        mhc_ch = [mhc] + ([b2m] if b2m else [])
        full = [pep_ch] + mhc_ch + tcr
        notcr = [pep_ch] + mhc_ch
        io.set_structure(model.get_parent())
        io.save(f"{DST}/arm_full/{pdb}.pdb", ArmSelect(full))
        io.save(f"{DST}/arm_notcr/{pdb}.pdb", ArmSelect(notcr))

        rows.append({
            "complex": pdb, "pdb": pdb, "pep_chain": pep_ch,
            "pep_len": len(pep_res), "pep_seq": pep_seq,
            "pep_resids": ";".join(f"{r.id[1]}{r.id[2].strip()}" for r in pep_res),
            "pep_fused_to_mhc": 0,
            "mhc_chains": "".join(mhc_ch), "tcr_chains": "".join(tcr),
            "full_chains": "".join(full), "notcr_chains": "".join(notcr),
            "full_len": sum(lens[c] for c in full),
            "notcr_len": sum(lens[c] for c in notcr),
            "dataset_valid": int(bool(d["valid"])),
            "n_chains_in_file": len(lens),
            "modres_rewritten": ";".join(rewritten),
        })
        print(f"{pdb:6s} pep={pep_ch}({len(pep_res)}) {pep_seq:12s} mhc={''.join(mhc_ch):3s} "
              f"tcr={''.join(tcr):3s}  L: {rows[-1]['full_len']}->{rows[-1]['notcr_len']}"
              f"  valid={bool(d['valid'])}"
              f"{'  [2 copies in file]' if len(lens) > 6 else ''}"
              f"{'  [modres: ' + ','.join(rewritten) + ']' if rewritten else ''}")

        if pep_seq != str(d["peptide"]):
            print(f"       !! epitope differs from dataset.csv: {pep_seq} vs {d['peptide']}")

    with open(f"{DST}/manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nstaged {len(rows)} complexes x 2 arms -> {DST}/manifest.csv")
    for pdb, why in skipped:
        print(f"  skipped {pdb}: {why}")


if __name__ == "__main__":
    main()
