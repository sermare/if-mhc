from PIL import Image, ImageDraw, ImageFont

RECOVERY_OWN = "/home/ubuntu/if-mhc/figures/L5_0796681_2/6AMU_L5_max_w1_0796681_1_el30_cog.png"
RECOVERY_OTHER = "/home/ubuntu/if-mhc/figures/L5_0796681_2/6AMU_L5_max_w1_0796681_1_no_cog.png"
CROSSING_OWN = "/home/ubuntu/if-mhc/figures/k18_44/6AM5_k18_44_no_oth.png"
CROSSING_OTHER = "/home/ubuntu/if-mhc/figures/k18_44/6AM5_k18_44_el30_oth.png"
OUT = "/home/ubuntu/if-mhc/figures/fig2_recovery_crossing/fig2_recovery_crossing.png"

PANELS = [
    (RECOVERY_OWN, "Recovers target DRG (1.73 Å)"),
    (RECOVERY_OTHER, "Misses non-target GIG (3.06 Å)"),
    (CROSSING_OWN, "Misses own cognate GIG (2.93 Å)"),
    (CROSSING_OTHER, "Crosses into DRG (1.85 Å)"),
]


def load_font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            pass
    return ImageFont.load_default()


f = load_font(13)
ims = [Image.open(p).convert("RGB") for p, _ in PANELS]
cw, ch = ims[0].width, ims[0].height
pad_top = 20
ncols = 2
nrows = 2
canvas = Image.new("RGB", (cw * ncols, (ch + pad_top) * nrows), (255, 255, 255))
d = ImageDraw.Draw(canvas)
for i, (im, (_, label)) in enumerate(zip(ims, PANELS)):
    r, c = divmod(i, ncols)
    x, y = c * cw, r * (ch + pad_top)
    canvas.paste(im, (x, y + pad_top))
    d.text((x + 10, y + 5), label, fill=(0, 0, 0), font=f)

canvas.save(OUT)
print("wrote", OUT)
