#!/usr/bin/env python3
"""Assemble Fig. 2 (recovery vs crossing) from the rendered PyMOL panels.

Scored against the MD-calibrated 310 K native envelope (physiological; DRG band <=1.48 A, block-bootstrap
95% CI 1.36-1.58 A; plus F-pocket occupant identity + burial depth).

TOP ROW  - a de-novo RECOVERY: 6AMU `max` design, Ca-RMSD 1.07 A to its native DRG register, correct
           F-pocket occupant (p9) and burial depth -> passes all three criteria.
BOTTOM ROW - a de-novo CROSSING: 6AM5 `L3_nterm_t2` design, 1.44 A to the non-native DRG register, seating
           the DRG anchor (p9) -- inside the 1.48 A band, i.e. a genuine register crossing.
"""
import os
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIG = os.path.join(ROOT, "figures")

RECOVERY_OWN   = f"{FIG}/recovery_denovo_max/6AMU_max_recovery_el30_cog.png"
RECOVERY_OTHER = f"{FIG}/recovery_denovo_max/6AMU_max_recovery_no_cog.png"
CROSSING_OWN   = f"{FIG}/crossing_L3/6AM5_L3_crossing_no_oth.png"
CROSSING_OTHER = f"{FIG}/crossing_L3/6AM5_L3_crossing_el30_oth.png"
OUT = f"{FIG}/fig2_recovery_crossing/fig2_recovery_crossing.png"

PANELS = [
    (RECOVERY_OWN,   "de-novo RECOVERY  -  recovers native DRG (1.07 A)  [inside the 1.48 A 310K band]"),
    (RECOVERY_OTHER, "de-novo RECOVERY  -  misses the non-target GIG register (2.75 A)"),
    (CROSSING_OWN,   "de-novo CROSSING  -  departs its own GIG register (3.02 A)"),
    (CROSSING_OTHER, "de-novo CROSSING  -  crosses into DRG (1.44 A)  [inside the 1.48 A 310K band]"),
]


def load_font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            pass
    return ImageFont.load_default()


f = load_font(34)
ims = [Image.open(p).convert("RGB") for p, _ in PANELS]
cw, ch = ims[0].width, ims[0].height
pad_top = 48
ncols = nrows = 2
canvas = Image.new("RGB", (cw * ncols, (ch + pad_top) * nrows), (255, 255, 255))
d = ImageDraw.Draw(canvas)
for i, (im, (_, label)) in enumerate(zip(ims, PANELS)):
    r, c = divmod(i, ncols)
    x, y = c * cw, r * (ch + pad_top)
    canvas.paste(im, (x, y + pad_top))
    d.text((x + 10, y + 6), label, fill=(0, 0, 0), font=f)

canvas.save(OUT)
print("wrote", OUT)
