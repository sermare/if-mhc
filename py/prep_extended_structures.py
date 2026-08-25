#!/usr/bin/env python3
"""Canonicalise the SKEMPI and 6AM complexes into the panel's chain convention.

The panel notebooks address chains positionally: A = MHC heavy (or class II alpha),
B = beta-2 microglobulin (or class II beta), C = peptide, D = TCR alpha, E = TCR beta.
The freshly generated complexes keep their deposited chain letters, which are not
consistent (peptide is C, Q, D, P or J depending on the entry), so they are rewritten
here into the same convention and dropped alongside the panel structures.

Roles come from the design manifest, not from residue-count heuristics: five of the new
complexes are MHC class II, whose alpha/beta chains fall outside the class I length ranges
that py/prep_panel_structures.py keys on, and five carry no beta-2 microglobulin at all
(chain B is then simply absent, matching the context the designs were actually generated in).

Writes inputs/pmhc_tcr_dataset/<PDB>.pdb and appends the matching rows to that directory's
dataset.csv, so every panel notebook picks the new structures up with no path changes.

  /home/ubuntu/miniforge3/bin/python3 py/prep_extended_structures.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

ROOT = Path("/home/ubuntu/if-mhc")
DEST = ROOT / "inputs/pmhc_tcr_dataset"
DATASET = DEST / "dataset.csv"

AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "MSE": "M",
}

SOURCES = [
    ("skempi", ROOT / "designs/skempi/t01/manifest.csv", ROOT / "inputs/skempi/pdb"),
    ("focus6am", ROOT / "designs/focus6am/t01/manifest.csv", ROOT / "inputs/focus_6am"),
]


def parse_modres(spec: str) -> dict[tuple[str, str], str]:
    """'C5:F2F->PHE;...' -> {(chain, resid): standard resname}.

    SKEMPI entries occasionally carry a chemically modified residue (3D3V's F2F at peptide
    position 5). It is deposited as HETATM, so without this it is dropped and the epitope
    comes out a residue short.
    """
    out = {}
    if not isinstance(spec, str) or not spec.strip():
        return out
    for item in spec.split(";"):
        item = item.strip()
        if not item or ":" not in item or "->" not in item:
            continue
        loc, _, rest = item.partition(":")
        _, _, std = rest.partition("->")
        out[(loc[0], loc[1:])] = std.strip()
    return out


def read_model1(path: Path, modres: dict[tuple[str, str], str] | None = None) -> list[str]:
    """ATOM/HETATM lines of the first MODEL only, first altloc, MSE and any manifest-declared
    modified residue promoted to ATOM with its standard residue name."""
    modres = modres or {}
    out, in_model = [], True
    for line in path.read_text().splitlines():
        if line.startswith("MODEL"):
            in_model = out == []
            continue
        if line.startswith("ENDMDL"):
            in_model = False
            continue
        if not in_model:
            continue
        key = (line[21], line[22:27].strip()) if len(line) > 26 else None
        is_mod = line.startswith("HETATM") and key in modres
        if line.startswith("ATOM") or (line.startswith("HETATM") and line[17:20] == "MSE") or is_mod:
            if line[16] not in (" ", "A"):        # keep the first altloc only
                continue
            if is_mod:
                line = line[:17] + f"{modres[key]:>3}" + line[20:]
            if line.startswith("HETATM"):
                line = "ATOM  " + line[6:]
            out.append(line)
    return out


def resolution_of(path: Path) -> float:
    for line in path.read_text().splitlines():
        if line.startswith("REMARK   2 RESOLUTION"):
            try:
                return float(line.split("RESOLUTION.")[1].split("ANGSTROMS")[0].strip())
            except Exception:
                return float("nan")
    return float("nan")


def canonicalise(pdb_lines: list[str], mapping: dict[str, str],
                 pep_src: str, pep_resids: set[str]) -> tuple[list[str], str]:
    """Rewrite chain letters; extract the peptide by (source chain, residue id) into chain C."""
    kept, seq, seen = [], [], set()
    for line in pdb_lines:
        ch = line[21]
        resid = line[22:27].strip()          # resSeq + insertion code
        if ch == pep_src and resid in pep_resids:
            new = "C"
        elif ch in mapping:
            new = mapping[ch]
        else:
            continue
        kept.append(line[:21] + new + line[22:])
        if new == "C" and line[12:16].strip() == "CA":
            key = resid
            if key not in seen:
                seen.add(key)
                seq.append(AA3TO1.get(line[17:20].strip(), "X"))
    return kept, "".join(seq)


def chain_sort_key(line: str) -> tuple:
    return ("ABCDE".index(line[21]),)


def main() -> None:
    if not (DEST / "dataset.csv.orig").exists():
        shutil.copy(DATASET, DEST / "dataset.csv.orig")
        print(f"backed up dataset.csv -> {DEST / 'dataset.csv.orig'}")

    existing = pd.read_csv(DEST / "dataset.csv.orig")
    new_rows, written = [], 0

    for source, man_path, pdb_dir in SOURCES:
        man = pd.read_csv(man_path)
        for _, r in man.iterrows():
            pdb = r["pdb"]
            src = pdb_dir / f"{pdb}.pdb"
            if not src.exists():
                print(f"  MISSING structure {src}")
                continue
            lines = read_model1(src, parse_modres(r.get("modres_rewritten", "")))

            mhc = list(str(r["mhc_chains"]))
            tcr = list(str(r["tcr_chains"]))
            mapping = {}
            if len(mhc) >= 1:
                mapping[mhc[0]] = "A"
            if len(mhc) >= 2:
                mapping[mhc[1]] = "B"
            if len(tcr) >= 1:
                mapping[tcr[0]] = "D"
            if len(tcr) >= 2:
                mapping[tcr[1]] = "E"

            pep_resids = {s.strip() for s in str(r["pep_resids"]).split(";")}
            kept, seq = canonicalise(lines, mapping, str(r["pep_chain"]), pep_resids)

            if seq != r["pep_seq"]:
                print(f"  {pdb}: peptide mismatch, extracted {seq!r} vs manifest {r['pep_seq']!r}"
                      f" -- skipped")
                continue

            kept.sort(key=chain_sort_key)
            # notebook 01 parses TITLE/COMPND for the allele and REMARK 2 for the resolution,
            # so those header records have to survive the rewrite
            header = [l for l in src.read_text().splitlines()
                      if l.startswith(("HEADER", "TITLE", "COMPND", "REMARK   2"))]
            (DEST / f"{pdb}.pdb").write_text("\n".join(header + kept) + "\nEND\n")

            # the notebooks also read the prepped copies (CDR3 extraction, peptide B-factors):
            # full = A-E, mhconly = A,B,C, same as py/prep_panel_structures.py produces
            for arm, keep_chains in [("full", set("ABCDE")), ("mhconly", set("ABC"))]:
                sub = [l for l in kept if l[21] in keep_chains]
                p = ROOT / f"outputs/panel_prep/{pdb}/pdbs/{arm}/{pdb}.pdb"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("\n".join(header + sub) + "\nEND\n")
            written += 1

            chains = sorted({l[21] for l in kept})
            lens = {c: len({l[22:27] for l in kept if l[21] == c}) for c in chains}
            new_rows.append(dict(
                pdb=pdb, valid=True, peptide=seq, pep_len=len(seq), pep_chain="C",
                n_chains=len(chains),
                chain_lens=",".join(f"{c}:{lens[c]}" for c in chains),
                allele=f"source:{source}"))

    add = pd.DataFrame(new_rows)
    add = add[~add.pdb.isin(set(existing.pdb))]
    out = pd.concat([existing, add], ignore_index=True)
    out.to_csv(DATASET, index=False)
    print(f"\nwrote {written} canonicalised structures into {DEST}")
    print(f"dataset.csv: {len(existing)} -> {len(out)} rows (+{len(add)})")
    print(add[["pdb", "peptide", "pep_len", "n_chains", "chain_lens"]].to_string(index=False))


if __name__ == "__main__":
    main()
