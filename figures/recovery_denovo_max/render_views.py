COGNATE_ALIGNED = r"/global/scratch/users/sergiomar10/if-mhc/outputs/aligned/6AMU_native_aligned.pdb"
OTHER_ALIGNED   = r"/global/scratch/users/sergiomar10/if-mhc/outputs/aligned/6AM5_native_aligned.pdb"
DESIGN_ALIGNED  = r"/global/scratch/users/sergiomar10/if-mhc/outputs/aligned/6AMU_max_recovery_aligned.pdb"
OUT_DIR         = r"/global/scratch/users/sergiomar10/if-mhc/figures/recovery_denovo_max"
BASE            = r"6AMU_max_recovery"
COGNATE_NAME    = r"DRG"
OTHER_NAME      = r"GIG"
COGNATE_PID     = r"6AMU"
OTHER_PID       = r"6AM5"

import os, json
from pymol import cmd, util

os.makedirs(OUT_DIR, exist_ok=True)

# ---- tweakables ----
MHC_CHAIN      = "A"
PEP_CHAIN      = "P"
HC_SURF_TRANSP = 0.72     # heavy-chain surface transparency (was 0.65 -> a touch fainter)
HC_CART_TRANSP = 0.7     # heavy-chain cartoon transparency for el30 view (very faint)

# ---- scene ----
cmd.reinitialize()
cmd.load(COGNATE_ALIGNED, "native_cog")
cmd.load(OTHER_ALIGNED,   "native_oth")
cmd.load(DESIGN_ALIGNED,  "design")
for obj in ("native_cog", "native_oth", "design"):
    print("chains[%s] = %s" % (obj, ",".join(cmd.get_chains(obj))))

cmd.hide("everything")
cmd.bg_color("white")
cmd.set("ray_opaque_background", 1)
cmd.set("antialias", 2)
cmd.set("cartoon_transparency", 0.0)
cmd.set("cartoon_side_chain_helper", 1)
# labels: float in front, white-outlined for legibility
cmd.set("label_size", 18)
cmd.set("label_color", "black")
cmd.set("label_outline_color", "white")
cmd.set("float_labels", 1)
# faint H-bond dashes
cmd.set("dash_gap", 0.45)
cmd.set("dash_length", 0.35)

# MHC groove (chain A) as its own object; cartoon always, surface toggled per figure
cmd.create("mhc_surf", "native_cog and chain " + MHC_CHAIN)
cmd.show("cartoon", "mhc_surf")
cmd.color("grey80", "mhc_surf")
cmd.set("surface_color", "grey90", "mhc_surf")

# which native object carries the DRG peptide
DRG_NAME = OTHER_NAME if OTHER_NAME == "DRG" else COGNATE_NAME
drg_obj  = "native_oth" if OTHER_NAME == "DRG" else "native_cog"

# ---- annotations: N-term / C-term (on design) + F-pocket ----
cmd.label("first (design and chain %s and name CA)" % PEP_CHAIN, '"N-term"')
cmd.label("last (design and chain %s and name CA)" % PEP_CHAIN, '"C-term"')
cmd.pseudoatom("fpocket_lbl",
               selection="(native_cog and chain %s) within 6 of (last (native_cog and chain %s and name CA))"
                         % (MHC_CHAIN, PEP_CHAIN),
               label="F-pocket")
cmd.hide("everything", "fpocket_lbl")
cmd.show("label", "fpocket_lbl")
cmd.set("label_color", "grey20", "fpocket_lbl")

# ---- DRG -> MHC H-bonds (created once, enabled only when needed) ----
cmd.distance("hb_drg", drg_obj + " and chain " + PEP_CHAIN,
             drg_obj + " and chain " + MHC_CHAIN, mode=2)
cmd.hide("labels", "hb_drg")
cmd.set("dash_color", "grey50", "hb_drg")
cmd.set("dash_width", 1.4, "hb_drg")
cmd.disable("hb_drg")

pep_sel = "((native_cog and chain %s) or (native_oth and chain %s) or (design and chain %s))" \
          % (PEP_CHAIN, PEP_CHAIN, PEP_CHAIN)

W, H, DPI = 900, 720, 150
cmd.set("ray_trace_mode", 1)


def fmt_set_view(v):
    rows = [v[0:3], v[3:6], v[6:9], v[9:12], v[12:15], v[15:18]]
    out = ["set_view (\\"]
    for i, r in enumerate(rows):
        out.append("   " + ", ".join("%14.9f" % x for x in r) + (",\\" if i < 5 else " )"))
    return "\n".join(out)


def load_font(sz):
    from PIL import ImageFont
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def annotate_legend(png, entries, title):
    """Bake a small colour legend + title into the bottom-left of a PNG (in place)."""
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return
    im = Image.open(png).convert("RGB"); d = ImageDraw.Draw(im)
    f, ft = load_font(19), load_font(17)

    def tlen(s, fo):
        try:
            return int(d.textlength(s, font=fo))
        except Exception:
            return int(fo.getbbox(s)[2])

    pad, sw, gap, lh = 10, 22, 8, 27
    tw = max([tlen(l, f) for _, _, l in entries] + [tlen(title, ft)])
    bw = pad + sw + gap + tw + pad
    bh = pad + lh * (len(entries) + 1) + pad
    x0, y0 = 12, im.height - bh - 12
    d.rectangle([x0, y0, x0 + bw, y0 + bh], fill=(255, 255, 255), outline=(120, 120, 120))
    d.text((x0 + pad, y0 + pad), title, fill=(0, 0, 0), font=ft)
    yy = y0 + pad + lh
    for kind, rgb, label in entries:
        if kind == "dash":
            cy = yy + sw // 2
            for k in range(0, sw, 8):
                d.line([x0 + pad + k, cy, x0 + pad + k + 4, cy], fill=rgb, width=3)
        elif kind == "swatch":
            d.rectangle([x0 + pad, yy, x0 + pad + sw, yy + sw], fill=rgb, outline=(0, 0, 0))
        # kind == "text" -> no glyph
        d.text((x0 + pad + sw + gap, yy + 1), label, fill=(0, 0, 0), font=f)
        yy += lh
    im.save(png)


def do_single_view(tag, name, turns, desc, legend_entries):
    png = os.path.join(OUT_DIR, "%s_%s.png" % (BASE, tag))
    # View is already set by cmd.set_view(base_view_flipped) in the calling function
    v = tuple(cmd.get_view())
    cmd.ray(W, H); cmd.png(png, dpi=DPI, ray=0)
    annotate_legend(png, legend_entries, "%s  [%s]" % (BASE, tag))
    tsum = "el45/az180 (flipped 180°)"
    print("\n[%s] %s (%s)" % (tag, desc, tsum)); print(fmt_set_view(v))
    
    txt = os.path.join(OUT_DIR, "%s_%s_views.txt" % (BASE, tag))
    with open(txt, "w") as fh:
        fh.write("[%s] %s (%s)\n" % (name, desc, tsum))
        fh.write(fmt_set_view(v) + "\n")
    
    with open(os.path.join(OUT_DIR, "%s_%s_views.json" % (BASE, tag)), "w") as fh:
        json.dump({"name": name, "desc": desc, "view_type": tsum, "view": list(v), "png": png}, fh, indent=2)
    
    print("wrote params -> " + txt)


# ============================ base view (el45/az180) ============================
# Exact view from pep/el45_az180
base_view = (
    0.380353153,    0.780226111,    0.496566862,
   -0.339069933,    0.617176712,   -0.710017383,
   -0.860443652,    0.101686463,    0.499296367,
    0.000000000,    0.000000000, -102.892776489,
  -34.647323608,   20.778272629,    2.131773949,
   81.121444702,  124.664108276,  -20.000000000
)

# Generate flipped view (180° rotation around Z-axis to swap N-term and C-term positions)
cmd.set_view(base_view)
cmd.turn("z", 180)
base_view_flipped = tuple(cmd.get_view())

# el30 / az000 view (from pep/el30_az000)
base_view_el30 = (
   -0.380353183,    0.681584120,    0.625119567,
    0.339069843,   -0.526087046,    0.779913008,
    0.860443652,    0.508601606,   -0.031005755,
    0.000000000,    0.000000000, -102.892776489,
  -34.647323608,   20.778272629,    2.131773949,
   81.121444702,  124.664108276,  -20.000000000
)


# ============================ FIGURE A (no cognate/cyan) ============================
def figure_A():
    cmd.enable("native_cog"); cmd.enable("native_oth"); cmd.enable("design")
    cmd.show("cartoon", "mhc_surf"); cmd.show("surface", "mhc_surf")
    cmd.set("transparency", HC_SURF_TRANSP, "mhc_surf")
    
    # Hide cognate (cyan), show other (orange) and design (magenta)
    cmd.hide("everything", "native_cog")
    for obj, col in [("native_oth", "orange"), ("design", "magenta")]:
        s = obj + " and chain " + PEP_CHAIN
        cmd.show("cartoon", s); cmd.show("sticks", s); cmd.color(col, s); util.cnc(s)
    
    cmd.set_view(base_view_flipped)
    entries = [("swatch", (229, 229, 229), "MHC groove (%s)" % COGNATE_PID),
               ("swatch", (255, 128, 0), "other %s" % OTHER_NAME),
               ("swatch", (255, 0, 255), "design"),
               ("text",   None, "labels: N-term / C-term / F-pocket")]
    do_single_view("no_cog", "el45_az180", [], "el45/az180 (no %s)" % COGNATE_NAME, entries)


# ============================ FIGURE B (no other/orange) ============================
def figure_B():
    cmd.enable("native_cog"); cmd.enable("native_oth"); cmd.enable("design")
    cmd.show("cartoon", "mhc_surf"); cmd.show("surface", "mhc_surf")
    cmd.set("transparency", HC_SURF_TRANSP, "mhc_surf")
    
    # Hide other (orange), show cognate (cyan) and design (magenta)
    cmd.hide("everything", "native_oth")
    for obj, col in [("native_cog", "teal"), ("design", "magenta")]:
        s = obj + " and chain " + PEP_CHAIN
        cmd.show("cartoon", s); cmd.show("sticks", s); cmd.color(col, s); util.cnc(s)
    
    cmd.set_view(base_view_flipped)
    entries = [("swatch", (229, 229, 229), "MHC groove (%s)" % COGNATE_PID),
               ("swatch", (0, 191, 191), "cognate %s" % COGNATE_NAME),
               ("swatch", (255, 0, 255), "design"),
               ("text",   None, "labels: N-term / C-term / F-pocket")]
    do_single_view("no_oth", "el45_az180", [], "el45/az180 (no %s)" % OTHER_NAME, entries)


figure_A()
figure_B()


# ============================ FIGURE C (el30/az000 - all three peptides) ============================
def figure_C():
    cmd.enable("native_cog"); cmd.enable("native_oth"); cmd.enable("design")
    cmd.set("transparency_mode", 1) 
    # Note: In newer versions of PyMOL, cmd.set("transparency_mode", 3) 
    # often looks much better for real-time viewing in the viewport.

    # 2. Your existing code
    cmd.show("cartoon", "mhc_surf")
    cmd.set("cartoon_transparency", HC_CART_TRANSP, "mhc_surf")

    cmd.show("surface", "mhc_surf")
    cmd.set("transparency", 0.95, "mhc_surf")

    # 3. (Optional) Prevent the inside of the surface from casting heavy dark shadows on the cartoon
    cmd.set("two_sided_lighting", 1)

    
    # Show all three peptides
    for obj, col in [("native_cog", "teal"), ("native_oth", "orange"), ("design", "magenta")]:
        s = obj + " and chain " + PEP_CHAIN
        cmd.show("cartoon", s); cmd.show("sticks", s); cmd.color(col, s); util.cnc(s)
    
    cmd.set_view(base_view_el30)
    entries = [("swatch", (240, 240, 240), "MHC groove (%s) [faint]" % COGNATE_PID),
               ("swatch", (0, 191, 191), "cognate %s" % COGNATE_NAME),
               ("swatch", (255, 128, 0), "other %s" % OTHER_NAME),
               ("swatch", (255, 0, 255), "design"),
               ("text",   None, "labels: N-term / C-term / F-pocket")]
    do_single_view("el30_all3", "el30_az000", [], "el30/az000 (all 3 peptides)", entries)


# ============================ FIGURE D (el30/az000 - only other) ============================
def figure_D():
    cmd.disable("native_cog"); cmd.enable("native_oth"); cmd.enable("design")
    cmd.show("cartoon", "mhc_surf");
    cmd.set("cartoon_transparency", HC_CART_TRANSP, "mhc_surf")
    cmd.show("surface", "mhc_surf")
    cmd.set("transparency", 0.95, "mhc_surf")  # very faint - 85% transparent
    
    # Show only other and design
    for obj, col in [("native_oth", "orange"), ("design", "magenta")]:
        s = obj + " and chain " + PEP_CHAIN
        cmd.show("cartoon", s); cmd.show("sticks", s); cmd.color(col, s); util.cnc(s)
    
    cmd.set_view(base_view_el30)
    entries = [("swatch", (240, 240, 240), "MHC groove (%s) [faint]" % COGNATE_PID),
               ("swatch", (255, 128, 0), "other %s" % OTHER_NAME),
               ("swatch", (255, 0, 255), "design"),
               ("text",   None, "labels: N-term / C-term / F-pocket")]
    do_single_view("el30_oth", "el30_az000", [], "el30/az000 (only other)", entries)


# ============================ FIGURE E (el30/az000 - only cognate) ============================
def figure_E():
    cmd.enable("native_cog"); cmd.disable("native_oth"); cmd.enable("design")
    cmd.show("cartoon", "mhc_surf"); cmd.show("surface", "mhc_surf")
    cmd.set("cartoon_transparency", HC_CART_TRANSP, "mhc_surf")
    cmd.set("transparency", 0.95, "mhc_surf")  # very faint - 85% transparent
    
    # Show only cognate and design
    for obj, col in [("native_cog", "teal"), ("design", "magenta")]:
        s = obj + " and chain " + PEP_CHAIN
        cmd.show("cartoon", s); cmd.show("sticks", s); cmd.color(col, s); util.cnc(s)
    
    cmd.set_view(base_view_el30)
    entries = [("swatch", (240, 240, 240), "MHC groove (%s) [faint]" % COGNATE_PID),
               ("swatch", (0, 191, 191), "cognate %s" % COGNATE_NAME),
               ("swatch", (255, 0, 255), "design"),
               ("text",   None, "labels: N-term / C-term / F-pocket")]
    do_single_view("el30_cog", "el30_az000", [], "el30/az000 (only cognate)", entries)


figure_C()
figure_D()
figure_E()
