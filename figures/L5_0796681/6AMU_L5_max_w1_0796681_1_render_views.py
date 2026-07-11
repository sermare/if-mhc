COGNATE_ALIGNED = r"/home/ubuntu/if-mhc/outputs/aligned/6AMU_native_aligned.pdb"
OTHER_ALIGNED   = r"/home/ubuntu/if-mhc/outputs/aligned/6AM5_native_aligned.pdb"
DESIGN_ALIGNED  = r"/home/ubuntu/if-mhc/outputs/aligned/6AMU_L5_max_w1_0796681_1_aligned.pdb"
OUT_DIR         = r"/home/ubuntu/if-mhc/figures/L5_0796681"
BASE            = r"6AMU_L5_max_w1_0796681_1"
COGNATE_NAME    = r"DRG"
OTHER_NAME      = r"GIG"
COGNATE_PID     = r"6AMU"
OTHER_PID       = r"6AM5"

import os, json
from pymol import cmd, util

os.makedirs(OUT_DIR, exist_ok=True)

MHC_CHAIN      = "A"
PEP_CHAIN      = "P"
HC_SURF_TRANSP = 0.72
HC_CART_TRANSP = 0.7

cmd.reinitialize()
cmd.load(COGNATE_ALIGNED, "native_cog")
cmd.load(OTHER_ALIGNED,   "native_oth")
cmd.load(DESIGN_ALIGNED,  "design")

cmd.hide("everything")
cmd.bg_color("white")
cmd.set("ray_opaque_background", 1)
cmd.set("antialias", 2)
cmd.set("cartoon_transparency", 0.0)
cmd.set("cartoon_side_chain_helper", 1)
cmd.set("label_size", 18)
cmd.set("label_color", "black")
cmd.set("label_outline_color", "white")
cmd.set("float_labels", 1)

cmd.create("mhc_surf", "native_cog and chain " + MHC_CHAIN)
cmd.show("cartoon", "mhc_surf")
cmd.color("grey80", "mhc_surf")
cmd.set("surface_color", "grey90", "mhc_surf")

cmd.label("first (design and chain %s and name CA)" % PEP_CHAIN, '"N-term"')
cmd.label("last (design and chain %s and name CA)" % PEP_CHAIN, '"C-term"')
cmd.pseudoatom("fpocket_lbl",
               selection="(native_cog and chain %s) within 6 of (last (native_cog and chain %s and name CA))"
                         % (MHC_CHAIN, PEP_CHAIN),
               label="F-pocket")
cmd.hide("everything", "fpocket_lbl")
cmd.show("label", "fpocket_lbl")
cmd.set("label_color", "grey20", "fpocket_lbl")

W, H, DPI = 900, 720, 150
cmd.set("ray_trace_mode", 1)


def load_font(sz):
    from PIL import ImageFont
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def annotate_legend(png, entries):
    """Simple swatch-only legend, no title/filename line."""
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return
    im = Image.open(png).convert("RGB"); d = ImageDraw.Draw(im)
    f = load_font(19)

    def tlen(s, fo):
        try:
            return int(d.textlength(s, font=fo))
        except Exception:
            return int(fo.getbbox(s)[2])

    pad, sw, gap, lh = 10, 22, 8, 27
    tw = max(tlen(l, f) for _, _, l in entries)
    bw = pad + sw + gap + tw + pad
    bh = pad + lh * len(entries) + pad
    x0, y0 = 12, im.height - bh - 12
    d.rectangle([x0, y0, x0 + bw, y0 + bh], fill=(255, 255, 255), outline=(120, 120, 120))
    yy = y0 + pad
    for kind, rgb, label in entries:
        if kind == "swatch":
            d.rectangle([x0 + pad, yy, x0 + pad + sw, yy + sw], fill=rgb, outline=(0, 0, 0))
        d.text((x0 + pad + sw + gap, yy + 1), label, fill=(0, 0, 0), font=f)
        yy += lh
    im.save(png)


def do_single_view(tag, desc, legend_entries):
    png = os.path.join(OUT_DIR, "%s_%s.png" % (BASE, tag))
    cmd.ray(W, H); cmd.png(png, dpi=DPI, ray=0)
    annotate_legend(png, legend_entries)
    print("wrote " + png)


base_view_el30 = (
   -0.380353183,    0.681584120,    0.625119567,
    0.339069843,   -0.526087046,    0.779913008,
    0.860443652,    0.508601606,   -0.031005755,
    0.000000000,    0.000000000, -102.892776489,
  -34.647323608,   20.778272629,    2.131773949,
   81.121444702,  124.664108276,  -20.000000000
)


def figure_cog():
    """Cognate (own) native + design + faint groove -- THE recovery panel."""
    cmd.enable("native_cog"); cmd.disable("native_oth"); cmd.enable("design")
    cmd.show("cartoon", "mhc_surf"); cmd.show("surface", "mhc_surf")
    cmd.set("cartoon_transparency", HC_CART_TRANSP, "mhc_surf")
    cmd.set("transparency", 0.95, "mhc_surf")
    for obj, col in [("native_cog", "teal"), ("design", "magenta")]:
        s = obj + " and chain " + PEP_CHAIN
        cmd.show("cartoon", s); cmd.show("sticks", s); cmd.color(col, s); util.cnc(s)
    cmd.set_view(base_view_el30)
    entries = [("swatch", (240, 240, 240), "MHC groove (%s) [faint]" % COGNATE_PID),
               ("swatch", (0, 191, 191), "native %s (own register)" % COGNATE_NAME),
               ("swatch", (255, 0, 255), "recovering design")]
    do_single_view("el30_cog", "el30/az000 (recovery: design vs own native)", entries)


def figure_all3():
    """All three: own native, other native, design -- for context."""
    cmd.enable("native_cog"); cmd.enable("native_oth"); cmd.enable("design")
    cmd.show("cartoon", "mhc_surf")
    cmd.set("cartoon_transparency", HC_CART_TRANSP, "mhc_surf")
    cmd.show("surface", "mhc_surf")
    cmd.set("transparency", 0.95, "mhc_surf")
    cmd.set("two_sided_lighting", 1)
    for obj, col in [("native_cog", "teal"), ("native_oth", "orange"), ("design", "magenta")]:
        s = obj + " and chain " + PEP_CHAIN
        cmd.show("cartoon", s); cmd.show("sticks", s); cmd.color(col, s); util.cnc(s)
    cmd.set_view(base_view_el30)
    entries = [("swatch", (240, 240, 240), "MHC groove (%s) [faint]" % COGNATE_PID),
               ("swatch", (0, 191, 191), "native %s (own)" % COGNATE_NAME),
               ("swatch", (255, 128, 0), "native %s (other)" % OTHER_NAME),
               ("swatch", (255, 0, 255), "recovering design")]
    do_single_view("el30_all3", "el30/az000 (all three)", entries)


figure_cog()
figure_all3()
