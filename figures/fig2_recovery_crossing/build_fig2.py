#!/usr/bin/env python3
"""Assemble Fig. 2 as a bare 2x2 tile of the rendered PyMOL structure panels.

  LEFT column  = RECOVERY       (6AMU `max` design vs its own DRG register, and vs non-target GIG)
  RIGHT column = CROSS-REACTIVE (6AM5 `L3_nterm_t2` design vs its own GIG, and vs the DRG it crosses into)

Deliberately NO baked column headers / colour key / captions: when the whole grid is scaled into a journal
or one-page-abstract column, any text baked into the image becomes illegible. The column headers and the
colour key are therefore added in LaTeX (at true document font size) around this image -- see paper.tex /
iscb_onepager.tex. Colour is by peptide identity: DRG teal, GIG orange, design magenta, MHC groove grey.
"""
import os
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIG = os.path.join(ROOT, "figures")

RECOVERY_OWN   = f"{FIG}/recovery_denovo_max/6AMU_max_recovery_el30_cog.png"
RECOVERY_OTHER = f"{FIG}/recovery_denovo_max/6AMU_max_recovery_no_cog.png"
CROSSING_OWN   = f"{FIG}/crossing_L3/6AM5_L3_crossing_no_oth.png"
CROSSING_OTHER = f"{FIG}/crossing_L3/6AM5_L3_crossing_el30_oth.png"
OUT = f"{FIG}/fig2_recovery_crossing/fig2_recovery_crossing.png"

# row-major order -> left column = recovery, right column = cross-reactive
PANELS = [RECOVERY_OWN, CROSSING_OWN,      # top row
          RECOVERY_OTHER, CROSSING_OTHER]  # bottom row

ims = [Image.open(p).convert("RGB") for p in PANELS]
cw, ch = ims[0].width, ims[0].height
canvas = Image.new("RGB", (cw * 2, ch * 2), (255, 255, 255))
for i, im in enumerate(ims):
    r, c = divmod(i, 2)
    canvas.paste(im, (c * cw, r * ch))
canvas.save(OUT)
print("wrote", OUT)
