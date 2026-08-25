#!/usr/bin/env python3
"""Collect the paper's figures into figures/paper/, named by the number they carry in the manuscript.

The notebooks write into figures/fig_panelN_*/ under names that describe how a figure was made
(`fig_panel3_recovery_raincloud_full_9mer.png`). That is the right name for a build artifact and the
wrong one for a reader holding the manuscript, who knows the same image as "Figure 1". This copies
each one under its manuscript name and writes a manifest.

FIGURES below is the single source of truth: manuscript number -> the notebook output that produces
it. It is kept explicit rather than parsed back out of the .tex so that the script is idempotent --
once the .tex points at figures/paper/, parsing it would make source and destination the same file.

  /home/ubuntu/miniforge3/bin/python3 py/organize_paper_figures.py            # copy + manifest
  /home/ubuntu/miniforge3/bin/python3 py/organize_paper_figures.py --rewrite  # also repoint the .tex
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

ROOT = Path("/home/ubuntu/if-mhc")
TEX = ROOT / "paper/paper_backbone.tex"
DEST = ROOT / "figures/paper"

P1 = "fig_panel1_dataset_presentation"
P2 = "fig_panel2_design_presentation"
P3 = "fig_panel3_recovery_presentation"
P4 = "fig_panel4_replicate_structures"
P5 = "fig_panel5_skempi_validation"
P6 = "fig_panel6_mhcflurry_esmcba_umap"
P7 = "fig_panel7_chemistry"
P8 = "fig_panel8_dmf5_6am_kd"

# manuscript number -> (slug for the filename, notebook output that produces it)
FIGURES: list[tuple[str, str, str]] = [
    ("1",    "per_position_recovery_9mer",        f"{P3}/fig_panel3_recovery_raincloud_full_9mer.png"),
    ("2",    "esmcba_vs_mhcflurry",               f"{P6}/fig_panel6_esmcba_vs_mhcflurry_by_model.png"),
    ("3",    "tcr_context_benefit_heatmap",       f"{P3}/fig_panel3_delta_recovery_heatmap.png"),
    ("4",    "dmf5_two_backbones",                f"{P8}/fig_panel8_dmf5_two_backbones.png"),
    ("S1a",  "peptide_length_distribution",       f"{P1}/fig_panel1_peptide_length_dist.png"),
    ("S1b",  "resolution_distribution",           f"{P1}/fig_panel1_resolution_dist.png"),
    ("S1c",  "mhc_contact_density",               f"{P1}/fig_panel1_mhc_contact_density.png"),
    ("S2",   "total_vs_unique_designs",           f"{P2}/fig_panel2_total_vs_unique_full.png"),
    ("S3",   "design_logos_pooled",               f"{P2}/fig_panel2_design_logos_full.png"),
    ("S4a",  "design_logos_proteinmpnn",          f"{P2}/fig_panel2_design_logo_grid_vanilla.png"),
    ("S4b",  "design_logos_nomhc",                f"{P2}/fig_panel2_design_logo_grid_noMHC.png"),
    ("S4c",  "design_logos_esmif1",               f"{P2}/fig_panel2_design_logo_grid_ESM-IF1.png"),
    ("S4d",  "design_logos_ligandmpnn",           f"{P2}/fig_panel2_design_logo_grid_LigandMPNN.png"),
    ("S5",   "unique_designs_per_model",          f"{P2}/fig_panel2_unique_per_model_meanstd.png"),
    ("S6",   "recovery_and_unique_vs_resolution", f"{P3}/fig_panel3_recovery_vs_resolution_and_unique.png"),
    ("S7",   "replicate_groups_vs_quality",       f"{P4}/fig_panel4_full_panel_recovery_vs_quality.png"),
    ("S8a",  "esmcba_vs_mhcflurry_no_tcr",        f"{P6}/fig_panel6_esmcba_vs_mhcflurry_by_model_mhconly.png"),
    ("S8b",  "resolution_by_model",               f"{P3}/fig_panel3_recovery_vs_resolution_and_unique_by_model.png"),
    ("S9a",  "own_score_shift_pooled",            f"{P6}/fig_panel6_own_score_shift_pooled.png"),
    ("S9b",  "own_score_shift_by_structure",      f"{P6}/fig_panel6_own_score_shift_by_structure.png"),
    ("S10",  "strong_binder_rate_by_model",       f"{P6}/fig_panel6_strong_binder_rate_by_model.png"),
    ("S11",  "umap_six_colorings",                f"{P6}/fig_panel6_umap_2x3.png"),
    ("S12",  "native_anchor_hydrophobicity",      f"{P7}/fig_panel7_native_anchor_hydrophobicity.png"),
    ("S13",  "per_position_recovery_10mer",       f"{P3}/fig_panel3_recovery_raincloud_full_10mer.png"),
    ("S14",  "per_position_recovery_13mer",       f"{P3}/fig_panel3_recovery_raincloud_full_13mer.png"),
    ("S15",  "skempi_ddg_vs_recovery",            f"{P5}/fig_panel5_pooled_skempi_ddg_vs_recovery.png"),
    ("S16",  "dmf5_score_vs_kd",                  f"{P8}/fig_panel8_6am_score_vs_kd.png"),
    ("S17",  "anchor_non_redundant_sets",         f"{P3}/fig_panel3_anchor_dedup_sets.png"),
]

NOTEBOOK = {P1: "01_dataset_presentation", P2: "02_design_presentation",
            P3: "03_recovery_presentation", P4: "04_replicate_structures",
            P5: "05_skempi_validation", P6: "06_mhcflurry_esmcba_umap",
            P7: "07_chemistry", P8: "08_dmf5_6am_kd_scores"}


def dest_name(num: str, slug: str) -> str:
    return f"fig{num}_{slug}.png"


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    expected = {dest_name(n, s) for n, s, _ in FIGURES}

    copied, missing = 0, []
    for num, slug, src_rel in FIGURES:
        src = ROOT / "figures" / src_rel
        if not src.exists():
            missing.append(src_rel); continue
        shutil.copy2(src, DEST / dest_name(num, slug))
        copied += 1
    # only remove images that are no longer part of the manuscript
    for f in DEST.glob("*.png"):
        if f.name not in expected:
            f.unlink(); print(f"  removed stale {f.name}")
    print(f"copied {copied} of {len(FIGURES)} figures into {DEST}")
    for m in missing:
        print(f"  MISSING {m}")

    rows = ["| Figure | File | Produced by | Notebook |", "|---|---|---|---|"]
    for num, slug, src_rel in FIGURES:
        d = src_rel.split("/")[0]
        rows.append(f"| {num} | `{dest_name(num, slug)}` | `figures/{src_rel}` | "
                    f"`notebooks/panel/{NOTEBOOK[d]}.ipynb` |")
    (DEST / "README.md").write_text(
        "# Paper figures\n\n"
        "Every figure in `paper/paper_backbone.tex`, named by the number it carries in the\n"
        "manuscript. **These are copies.** The originals are written by the panel notebooks into the\n"
        "`figures/fig_panelN_*/` directories; this folder is regenerated by\n"
        "`py/organize_paper_figures.py`. Editing an image here would be overwritten on the next run --\n"
        "change the notebook that produces it, re-run that notebook, then re-run the script.\n\n"
        + "\n".join(rows) + "\n")
    print(f"wrote {DEST / 'README.md'}")

    # the .tex must reference exactly the figures managed here, no more and no fewer
    tex = TEX.read_text()
    refs = set(re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", tex))
    stale = {r for r in refs if not r.startswith("figures/paper/")}
    unmanaged = {Path(r).name for r in refs if r.startswith("figures/paper/")} - expected
    if unmanaged:
        print(f"  WARNING: .tex references {len(unmanaged)} file(s) this script does not manage: "
              + ", ".join(sorted(unmanaged)))
    if stale and "--rewrite" not in sys.argv:
        print(f"  NOTE: {len(stale)} \\includegraphics still point outside figures/paper/; "
              f"re-run with --rewrite")

    if "--rewrite" in sys.argv:
        for num, slug, src_rel in FIGURES:
            tex = tex.replace("{figures/" + src_rel + "}",
                              "{figures/paper/" + dest_name(num, slug) + "}")
        TEX.write_text(tex)
        n = tex.count("figures/paper/")
        print(f"{TEX.name} now references {n} figures under figures/paper/")


if __name__ == "__main__":
    main()
