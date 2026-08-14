#!/usr/bin/env python3
"""Stage the SKEMPI v2 TCR/pMHC subset for inverse-folding sampling.

Two arms per complex:
  full   -- peptide + MHC-side chains + TCR chains   (TCR present)
  notcr  -- peptide + MHC-side chains                (TCR deleted)

The designed positions are the epitope only; everything else is fixed context.
Complexes already covered by inputs/pmhc_tcr_dataset are excluded so this run
adds no overlap with the existing campaigns.

Writes:
  inputs/skempi/manifest.csv          one row per complex (roles, epitope, resids)
  inputs/skempi/arm_full/<id>.pdb
  inputs/skempi/arm_notcr/<id>.pdb
  inputs/skempi/skempi_tcr_pmhc_mutations.csv   the filtered ddG records
"""
import os, sys, csv, warnings
import pandas as pd
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.Polypeptide import three_to_index, index_to_one

warnings.filterwarnings("ignore")
ROOT = "/global/scratch/users/sergiomar10/if-mhc"
SK   = f"{ROOT}/inputs/skempi"
RAW  = f"{SK}/pdb_raw"

AA3 = {"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU","LYS",
       "MET","PHE","PRO","SER","THR","TRP","TYR","VAL"}

# Modified residues rewritten to their natural parent. Each of the four models
# reads only backbone atoms, so this changes no geometry -- it exists because
# ProteinMPNN, biotite and Biopython each handle an unrecognised residue
# differently (gap character / dropped / dropped), which would silently give the
# same epitope a different length in each model. Rewriting pins one definition.
#   MSE selenomethionine   -> MET
#   F2F 3,5-difluoro-Phe   -> PHE  (3D3V epitope position 5)
MODRES = {"MSE": "MET", "F2F": "PHE"}
# atoms kept when rewriting a modified residue (all the models read from these)
KEEP_ATOMS = {"N", "CA", "C", "O", "CB"}

# Class-II constructs where the epitope is covalently fused to the MHC-beta
# N-terminus instead of being its own chain: (chain, epitope length).
LINKED = {"3C60": ("D", 13), "4P23": ("D", 13), "4P5T": ("D", 13)}


def resseq(res):
    """1-letter code, with modified residues read as their natural parent."""
    n = MODRES.get(res.get_resname(), res.get_resname())
    if n in AA3:
        return index_to_one(three_to_index(n))
    return None


def chain_residues(chain):
    """Ordered polymer residues (standard AA or a rewritable modified one)."""
    out = []
    for r in chain:
        if r.id[0] == " " or r.get_resname() in MODRES:
            if resseq(r) is not None:
                out.append(r)
    return out


def normalize_modres(model):
    """Rewrite modified residues in place so every parser reads them alike."""
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
            het, num, icode = res.id
            if het != " ":
                # promote HETATM to a polymer residue so every parser reads it as
                # sequence; Biopython's id setter re-indexes the parent for us
                # and leaves child_list order (and so chain order) alone
                res.id = (" ", num, icode)
            changed.append(f"{chain.id}{num}:{name}->{MODRES[name]}")
    return changed


class ArmSelect(Select):
    """Keep the requested chains: protein residues plus non-water heteroatoms.

    Ligands/glycans are retained because LigandMPNN can use them as context;
    ProteinMPNN and ESM-IF filter to backbone atoms and ignore them.
    """

    def __init__(self, chains):
        self.chains = set(chains)

    def accept_model(self, model):
        return model.id == 0

    def accept_chain(self, chain):
        return chain.id in self.chains

    def accept_residue(self, res):
        return res.get_resname() not in ("HOH", "DOD")

    def accept_atom(self, atom):
        if atom.element == "H":
            return False
        alt = atom.get_altloc()
        return alt in (" ", "A")


def main():
    df = pd.read_csv(f"{SK}/skempi_v2.csv", sep=";")
    tcr = df[df["Hold_out_type"] == "TCR/pMHC"].copy()
    tcr["pdb"] = tcr["#Pdb"].str.split("_").str[0].str.upper()

    done = set(pd.read_csv(f"{ROOT}/inputs/pmhc_tcr_dataset/dataset.csv")
               ["pdb"].astype(str).str.upper())
    overlap = sorted(set(tcr["pdb"]) & done)
    tcr = tcr[~tcr["pdb"].isin(done)]
    print(f"excluded {len(overlap)} complexes already in inputs/pmhc_tcr_dataset: {overlap}")

    tcr.to_csv(f"{SK}/skempi_tcr_pmhc_mutations.csv", index=False)
    nmut = tcr.groupby("#Pdb").size().to_dict()

    for arm in ("arm_full", "arm_notcr"):
        os.makedirs(f"{SK}/{arm}", exist_ok=True)

    parser = PDBParser(QUIET=True)
    io = PDBIO()
    rows = []

    for cid in sorted(tcr["#Pdb"].unique()):
        pdb, p1, p2 = cid.split("_")
        pdb = pdb.upper()
        model = parser.get_structure(pdb, f"{RAW}/{pdb}.pdb")[0]
        rewritten = normalize_modres(model)
        listed = list(p1) + list(p2)
        lens = {c: len(chain_residues(model[c])) for c in listed if c in model}

        if pdb in LINKED:
            # epitope fused to the MHC-beta N-terminus
            pep_ch, npep = LINKED[pdb]
            pep_res = chain_residues(model[pep_ch])[:npep]
            fused = 1
        else:
            # epitope is its own short chain; SKEMPI sometimes files it on the
            # TCR side (e.g. 5E9D), so pick it by length across all listed chains
            pep_ch = min(lens, key=lambda c: lens[c])
            if not (5 <= lens[pep_ch] <= 25):
                print(f"  !! {cid}: no epitope chain found (lens={lens}) -- SKIPPED")
                continue
            pep_res = chain_residues(model[pep_ch])
            fused = 0

        pep_seq = "".join(resseq(r) for r in pep_res)
        pep_ids = [f"{r.id[1]}{r.id[2].strip()}" for r in pep_res]

        # TCR = listed partner-2 chains that are not the epitope
        tcr_ch = [c for c in p2 if c in lens and c != pep_ch]
        # MHC side = everything else listed, minus the epitope chain
        mhc_ch = [c for c in listed if c in lens and c != pep_ch and c not in tcr_ch]
        if fused:
            mhc_ch = [c for c in mhc_ch if c != pep_ch]

        full_chains = [pep_ch] + mhc_ch + tcr_ch
        notcr_chains = [pep_ch] + mhc_ch

        io.set_structure(model.get_parent())
        io.save(f"{SK}/arm_full/{cid}.pdb", ArmSelect(full_chains))
        io.save(f"{SK}/arm_notcr/{cid}.pdb", ArmSelect(notcr_chains))

        rows.append({
            "complex": cid, "pdb": pdb,
            "pep_chain": pep_ch, "pep_len": len(pep_res), "pep_seq": pep_seq,
            "pep_resids": ";".join(pep_ids),
            "pep_fused_to_mhc": fused,
            "mhc_chains": "".join(mhc_ch), "tcr_chains": "".join(tcr_ch),
            "full_chains": "".join(full_chains),
            "notcr_chains": "".join(notcr_chains),
            "full_len": sum(lens[c] for c in full_chains),
            "notcr_len": sum(lens[c] for c in notcr_chains),
            "n_skempi_mut": nmut[cid],
            "modres_rewritten": ";".join(rewritten),
        })
        print(f"{cid:14s} pep={pep_ch}({len(pep_res)}) {pep_seq:15s} "
              f"mhc={''.join(mhc_ch):4s} tcr={''.join(tcr_ch):3s} "
              f"L: {rows[-1]['full_len']}->{rows[-1]['notcr_len']}"
              f"{'  [fused epitope]' if fused else ''}"
              f"{'  [modres: ' + ','.join(rewritten) + ']' if rewritten else ''}")

    with open(f"{SK}/manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nstaged {len(rows)} complexes x 2 arms -> {SK}/manifest.csv")
    print(f"kept {len(tcr)} SKEMPI ddG records over {tcr['pdb'].nunique()} PDBs")


if __name__ == "__main__":
    main()
