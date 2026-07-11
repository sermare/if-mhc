import os
from pymol import cmd, util

PEP_DIR = "/home/ubuntu/if-mhc/outputs/aligned/fig4_pep"
NATIVE_DRG = "/home/ubuntu/if-mhc/outputs/aligned/6AMU_native_aligned.pdb"
OUT_DIR = "/home/ubuntu/if-mhc/figures/fig4_conditioning"
MHC_CHAIN, PEP_CHAIN = "A", "P"

CONDS = [
    ("L3_nterm_t2", "C5", "5 contacts"),
    ("L4_expanded", "C9", "9 contacts"),
    ("L5_max", "C12", "12 contacts"),
    ("k18", "C18", "18 contacts"),
    ("k24", "C24", "24 contacts"),
    ("max", "C37", "37 contacts"),
]

base_view_el30 = (
   -0.380353183,    0.681584120,    0.625119567,
    0.339069843,   -0.526087046,    0.779913008,
    0.860443652,    0.508601606,   -0.031005755,
    0.000000000,    0.000000000, -102.892776489,
  -34.647323608,   20.778272629,    2.131773949,
   81.121444702,  124.664108276,  -20.000000000
)


def render_panel(cond, contact_label):
    """Same visual language as Figure 2: native (target, DRG) = orange, design = magenta,
    cartoon + sticks, faint translucent MHC groove."""
    cmd.reinitialize()
    cmd.load(NATIVE_DRG, "native_drg")
    cmd.hide("everything")
    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 1)
    cmd.set("antialias", 2)
    cmd.set("cartoon_side_chain_helper", 1)

    cmd.create("mhc_surf", "native_drg and chain " + MHC_CHAIN)
    cmd.color("grey80", "mhc_surf")
    cmd.set("surface_color", "grey90", "mhc_surf")

    cmd.set("transparency_mode", 1)
    cmd.show("cartoon", "mhc_surf")
    cmd.set("cartoon_transparency", 0.7, "mhc_surf")
    cmd.show("surface", "mhc_surf")
    cmd.set("transparency", 0.95, "mhc_surf")
    cmd.set("two_sided_lighting", 1)

    cmd.show("cartoon", "native_drg and chain " + PEP_CHAIN)
    cmd.show("sticks", "native_drg and chain " + PEP_CHAIN)
    cmd.color("orange", "native_drg and chain " + PEP_CHAIN)
    util.cnc("native_drg and chain " + PEP_CHAIN)

    cmd.load(f"{PEP_DIR}/{cond}_pep.pdb", "design")
    cmd.show("cartoon", "design")
    cmd.show("sticks", "design")
    cmd.color("magenta", "design")
    util.cnc("design")

    cmd.set_view(base_view_el30)
    cmd.zoom(f"(design or (native_drg and chain {PEP_CHAIN}))", buffer=5)
    cmd.set("ray_trace_mode", 1)

    png = f"{OUT_DIR}/_panel_{cond}.png"
    cmd.ray(800, 650)
    cmd.png(png, dpi=150, ray=0)
    print("wrote", png)
    return png


pngs = [render_panel(cond, label) for cond, _, label in CONDS]


def load_font(sz):
    from PIL import ImageFont
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def stitch(pngs, labels, legend_entries, out, ncols=3):
    from PIL import Image, ImageDraw
    ims = [Image.open(p).convert("RGB") for p in pngs]
    f = load_font(13)
    pad_top = 20
    cw, ch = ims[0].width, ims[0].height
    nrows = (len(ims) + ncols - 1) // ncols
    canvas = Image.new("RGB", (cw * ncols, (ch + pad_top) * nrows + 60), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    for i, (im, lbl) in enumerate(zip(ims, labels)):
        r, c = divmod(i, ncols)
        x, y = c * cw, r * (ch + pad_top)
        canvas.paste(im, (x, y + pad_top))
        d.text((x + 10, y + 4), lbl, fill=(0, 0, 0), font=f)
    # legend along the bottom
    pad, sw, gap, lh = 10, 22, 8, 27
    y0 = (ch + pad_top) * nrows + 10
    x0 = 12
    for kind, rgb, label in legend_entries:
        d.rectangle([x0, y0, x0 + sw, y0 + sw], fill=rgb, outline=(0, 0, 0))
        tw = int(d.textlength(label, font=f))
        d.text((x0 + sw + gap, y0 + 1), label, fill=(0, 0, 0), font=f)
        x0 += sw + gap + tw + 30
    canvas.save(out)
    print("wrote", out)


labels = [f"{name} ({lbl})" for _, name, lbl in CONDS]
stitch(pngs, labels,
       [("swatch", (255, 165, 0), "native DRG (target)"),
        ("swatch", (255, 0, 255), "best design for that conditioning")],
       f"{OUT_DIR}/fig4_conditioning.png", ncols=3)
