import os, glob
from pymol import cmd, util

ALIGN_DIR = "/home/ubuntu/if-mhc/outputs/aligned/fig3_pep"  # peptide-only (designs have no side chains anyway)
NATIVE = "/home/ubuntu/if-mhc/outputs/aligned/6AM5_native_aligned.pdb"
OUT_DIR = "/home/ubuntu/if-mhc/figures/fig3_threading"
MHC_CHAIN, PEP_CHAIN = "A", "P"
N_SHOW = 5  # keep it simple -- few traces per panel, not a crowd

base_view_el30 = (
   -0.380353183,    0.681584120,    0.625119567,
    0.339069843,   -0.526087046,    0.779913008,
    0.860443652,    0.508601606,   -0.031005755,
    0.000000000,    0.000000000, -102.892776489,
  -34.647323608,   20.778272629,    2.131773949,
   81.121444702,  124.664108276,  -20.000000000
)


def render_panel(pattern, design_color, tag):
    """Same visual language as Figure 2: native = teal, design(s) = a single hue, cartoon + sticks,
    faint translucent MHC groove. No black, no marker spheres -- just backbone + side chains."""
    cmd.reinitialize()
    cmd.load(NATIVE, "native")
    cmd.hide("everything")
    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 1)
    cmd.set("antialias", 2)
    cmd.set("cartoon_side_chain_helper", 1)

    cmd.create("mhc_surf", "native and chain " + MHC_CHAIN)
    cmd.color("grey80", "mhc_surf")
    cmd.set("surface_color", "grey90", "mhc_surf")

    cmd.set("transparency_mode", 1)
    cmd.show("cartoon", "mhc_surf")
    cmd.set("cartoon_transparency", 0.7, "mhc_surf")
    cmd.show("surface", "mhc_surf")
    cmd.set("transparency", 0.95, "mhc_surf")
    cmd.set("two_sided_lighting", 1)

    cmd.show("cartoon", "native and chain " + PEP_CHAIN)
    cmd.show("sticks", "native and chain " + PEP_CHAIN)
    cmd.color("teal", "native and chain " + PEP_CHAIN)
    util.cnc("native and chain " + PEP_CHAIN)
    cmd.label("first (native and chain %s and name CA)" % PEP_CHAIN, '"N-term"')
    cmd.label("last (native and chain %s and name CA)" % PEP_CHAIN, '"C-term"')
    cmd.set("label_size", 16)
    cmd.set("label_color", "black")
    cmd.set("label_outline_color", "white")
    cmd.set("float_labels", 1)

    n = 0
    for f in sorted(glob.glob(f"{ALIGN_DIR}/{pattern}"))[:N_SHOW]:
        name = os.path.splitext(os.path.basename(f))[0]
        cmd.load(f, name)
        cmd.show("cartoon", name)
        cmd.show("sticks", name)
        cmd.color(design_color, name)
        util.cnc(name)
        n += 1

    cmd.set_view(base_view_el30)
    design_objs = [o for o in cmd.get_object_list() if o.startswith(tag)]
    pep_sel = "(" + " or ".join(design_objs) + f" or (native and chain {PEP_CHAIN}))"
    cmd.zoom(pep_sel, buffer=5)
    cmd.set("ray_trace_mode", 1)

    png = f"{OUT_DIR}/_panel_{tag}threaded.png"
    cmd.ray(900, 720)
    cmd.png(png, dpi=150, ray=0)
    print("wrote", png, "n=", n)
    return png


png_fwd = render_panel("fwd_*_pep.pdb", "skyblue", "fwd")
png_rev = render_panel("rev_*_pep.pdb", "salmon", "rev")


def load_font(sz):
    from PIL import ImageFont
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def stitch(pngs, labels, legend_entries, out):
    from PIL import Image, ImageDraw
    ims = [Image.open(p).convert("RGB") for p in pngs]
    f = load_font(13)
    d0 = ImageDraw.Draw(ims[0])
    pad_top = 22
    w = sum(im.width for im in ims)
    h = max(im.height for im in ims) + pad_top
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    x = 0
    for im, lbl in zip(ims, labels):
        canvas.paste(im, (x, pad_top))
        d = ImageDraw.Draw(canvas)
        d.text((x + 12, 6), lbl, fill=(0, 0, 0), font=f)
        x += im.width
    d = ImageDraw.Draw(canvas)
    pad, sw, gap, lh = 10, 22, 8, 27
    tw = max(int(d.textlength(l, font=load_font(19))) for _, _, l in legend_entries)
    bw = pad + sw + gap + tw + pad
    bh = pad + lh * len(legend_entries) + pad
    x0, y0 = 12, h - bh - 12
    d.rectangle([x0, y0, x0 + bw, y0 + bh], fill=(255, 255, 255), outline=(120, 120, 120))
    yy = y0 + pad
    for kind, rgb, label in legend_entries:
        d.rectangle([x0 + pad, yy, x0 + pad + sw, yy + sw], fill=rgb, outline=(0, 0, 0))
        d.text((x0 + pad + sw + gap, yy + 1), label, fill=(0, 0, 0), font=load_font(19))
        yy += lh
    canvas.save(out)
    print("wrote", out)


stitch([png_fwd, png_rev], ["Threads forward (P7-P10 anchor)", "Threads reverse (P1-P4 anchor)"],
       [("swatch", (0, 128, 128), "native GIG"),
        ("swatch", (135, 206, 235), "forward-threaded designs"),
        ("swatch", (250, 128, 114), "reverse-threaded designs")],
       f"{OUT_DIR}/fig3_threading.png")
