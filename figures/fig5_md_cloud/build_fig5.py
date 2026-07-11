from pymol import cmd, util

NATIVE_DRG = "/home/ubuntu/if-mhc/outputs/aligned/6AMU_native_aligned.pdb"
DESIGN_ALIGNED = "/home/ubuntu/if-mhc/outputs/aligned/6AM5_k18_44_aligned.pdb"
CLOSEST_FRAME = "/home/ubuntu/if-mhc/outputs/aligned/closest_drg_frame_pep.pdb"
OUT_DIR = "/home/ubuntu/if-mhc/figures/fig5_md_cloud"
MHC_CHAIN, PEP_CHAIN = "A", "P"

cmd.reinitialize()
cmd.load(NATIVE_DRG, "native_drg")
cmd.load(DESIGN_ALIGNED, "design")
cmd.load(CLOSEST_FRAME, "md_frame")
cmd.hide("everything")
cmd.bg_color("white")
cmd.set("ray_opaque_background", 1)
cmd.set("antialias", 2)

cmd.create("mhc_surf", "native_drg and chain " + MHC_CHAIN)
cmd.color("grey80", "mhc_surf")
cmd.set("surface_color", "grey90", "mhc_surf")

cmd.set("transparency_mode", 1)
cmd.show("cartoon", "mhc_surf")
cmd.set("cartoon_transparency", 0.7, "mhc_surf")
cmd.show("surface", "mhc_surf")
cmd.set("transparency", 0.95, "mhc_surf")
cmd.set("two_sided_lighting", 1)

# native DRG structure, faint, for reference only -- same orange convention as fig2/fig4
cmd.show("cartoon", "native_drg and chain " + PEP_CHAIN)
cmd.show("sticks", "native_drg and chain " + PEP_CHAIN)
cmd.color("orange", "native_drg and chain " + PEP_CHAIN)
util.cnc("native_drg and chain " + PEP_CHAIN)
cmd.set("cartoon_transparency", 0.4, "native_drg")
cmd.set("stick_transparency", 0.4, "native_drg")

# the design
cmd.show("cartoon", "design and chain " + PEP_CHAIN)
cmd.show("sticks", "design and chain " + PEP_CHAIN)
cmd.color("magenta", "design and chain " + PEP_CHAIN)
util.cnc("design and chain " + PEP_CHAIN)

# the single closest real MD snapshot (Ca-only trace, so tube not sticks)
cmd.show("cartoon", "md_frame")
cmd.set("cartoon_trace_atoms", 1, "md_frame")
cmd.set("cartoon_tube_radius", 0.35, "md_frame")
cmd.color("gold", "md_frame")

base_view_el30 = (
   -0.380353183,    0.681584120,    0.625119567,
    0.339069843,   -0.526087046,    0.779913008,
    0.860443652,    0.508601606,   -0.031005755,
    0.000000000,    0.000000000, -102.892776489,
  -34.647323608,   20.778272629,    2.131773949,
   81.121444702,  124.664108276,  -20.000000000
)
cmd.set_view(base_view_el30)
cmd.zoom(f"((design and chain {PEP_CHAIN}) or md_frame or (native_drg and chain {PEP_CHAIN}))", buffer=3)
cmd.set("ray_trace_mode", 1)

png = f"{OUT_DIR}/fig5_closest_frame.png"
cmd.ray(900, 650)
cmd.png(png, dpi=150, ray=0)


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
    from PIL import Image, ImageDraw
    im = Image.open(png).convert("RGB"); d = ImageDraw.Draw(im)
    f = load_font(19)
    pad, sw, gap, lh = 10, 22, 8, 27
    tw = max(int(d.textlength(l, font=f)) for _, _, l in entries)
    bw = pad + sw + gap + tw + pad
    bh = pad + lh * len(entries) + pad
    x0, y0 = 12, im.height - bh - 12
    d.rectangle([x0, y0, x0 + bw, y0 + bh], fill=(255, 255, 255), outline=(120, 120, 120))
    yy = y0 + pad
    for kind, rgb, label in entries:
        d.rectangle([x0 + pad, yy, x0 + pad + sw, yy + sw], fill=rgb, outline=(0, 0, 0))
        d.text((x0 + pad + sw + gap, yy + 1), label, fill=(0, 0, 0), font=f)
        yy += lh
    im.save(png)


annotate_legend(png, [
    ("swatch", (255, 0, 255), "closest crossing design"),
    ("swatch", (255, 215, 0), "closest single MD frame (t=41.4ns)"),
    ("swatch", (255, 165, 0), "native DRG [faint]"),
    ("swatch", (230, 230, 230), "MHC groove [faint]"),
])
print("wrote", png)
