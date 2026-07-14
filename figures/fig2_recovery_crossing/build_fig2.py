#!/usr/bin/env python3
"""Assemble Fig. 2 (recovery vs crossing) from the rendered PyMOL panels.

Updated to the completed 12,786-design corpus, scored against the MD-calibrated 300 K native
envelope (own-register bands: GIG <=1.07 A, DRG <=1.14 A; plus F-pocket occupant + burial depth).

TOP ROW  - the first (and only) de-novo RECOVERY: 6AMU `max` design, Ca-RMSD 1.07 A to its native
           DRG register, correct F-pocket occupant (p9), burial depth 5.81 A -> passes all three
           criteria. 1 of 8,456 de-novo designs.
BOTTOM ROW - the best CROSSING candidate: 6AM5 `L3_nterm_t2` design, 1.44 A to the non-native DRG
           register. It seats the DRG anchor (p9) but sits OUTSIDE the 1.14 A band -- so it is a
           near-miss, not a crossing. No crossing was observed in any of the 12,786 designs.
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
    (RECOVERY_OWN,   "RECOVERY  -  recovers native DRG (1.07 A)  [inside 1.14 A band]"),
    (RECOVERY_OTHER, "RECOVERY  -  misses non-native GIG (2.75 A)"),
    (CROSSING_OWN,   "CROSSING candidate  -  misses own GIG (3.02 A)"),
    (CROSSING_OTHER, "CROSSING candidate  -  closest to DRG (1.44 A)  [OUTSIDE band: no crossing]"),
]


def load_font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            pass
    return ImageFont.load_default()


f = load_font(15)
ims = [Image.open(p).convert("RGB") for p, _ in PANELS]
cw, ch = ims[0].width, ims[0].height
pad_top = 24
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
