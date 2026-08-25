#!/usr/bin/env python3
"""Top-down view into the 6AM groove with both crystallized peptides overlaid, coloured by
Kyte-Doolittle hydrophobicity.

6AM5 (SMLGIGIVPV) and 6AMU (MMWDRGLGMM) are the same DMF5 receptor on the same HLA-A*02:01 groove,
solved with two different peptides. Superposing them on the MHC heavy chain alone -- never on the
peptides -- puts both epitopes in one frame, so any difference in where a side chain sits is a real
difference between the two structures rather than an artifact of how they were aligned.

The camera is placed on the axis running from the MHC groove centroid to the TCR centroid, looking
back down it, which is the direction a receptor approaches from and therefore the face of the peptide
a receptor reads. The peptide N-to-C direction is laid along screen x.

Colour is per-residue Kyte-Doolittle hydrophobicity on one shared ramp for both peptides, so the two
are directly comparable; the two structures are told apart by representation, not by colour.

  /home/ubuntu/miniforge3/envs/pymolviz/bin/python py/render_6am_peptide_hydrophobicity.py
  /home/ubuntu/miniforge3/bin/python3 py/render_6am_peptide_hydrophobicity.py --annotate-only
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/ubuntu/if-mhc")
OUT_DIR = ROOT / "figures/fig_panel8_dmf5_6am_kd"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Kyte-Doolittle, the same scale notebook 07 and the paper's anchor analysis use
KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5, "G": -0.4,
      "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8,
      "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}
AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
       "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
       "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}

ANNOTATE_ONLY = "--annotate-only" in sys.argv
if not ANNOTATE_ONLY:
    import pymol
    pymol.finish_launching(["pymol", "-qc"])
    from pymol import cmd


BLUE, WHITE, RED = np.array([0.13, 0.33, 0.75]), np.array([0.96, 0.96, 0.96]), np.array([0.78, 0.08, 0.10])


def kd_color(v: float) -> tuple[float, float, float]:
    """Hydrophilic -> blue, neutral -> white, hydrophobic -> red."""
    f = (np.clip(v, -4.5, 4.5) + 4.5) / 9.0
    return tuple((BLUE + (WHITE - BLUE) * (f / 0.5)) if f < 0.5
                 else (WHITE + (RED - WHITE) * ((f - 0.5) / 0.5)))


def colour_peptide(obj: str) -> list[tuple[int, str, float]]:
    """Colour each peptide residue by its Kyte-Doolittle value; return the per-residue table."""
    space = {"rows": []}
    cmd.iterate(f"{obj} and chain C and name CA", "rows.append((resi, resn))", space=space)
    table = []
    for resi, resn in space["rows"]:
        aa = AA3.get(resn, "X")
        v = KD.get(aa, 0.0)
        name = f"kd_{obj}_{resi}"
        cmd.set_color(name, list(kd_color(v)))
        cmd.color(name, f"{obj} and chain C and resi {resi}")
        table.append((int(resi), aa, v))
    return sorted(table)


ROLL_DEG = 275.0          # requested in-plane rotation of the whole view


def build_scene() -> None:
    """Load both complexes, superpose on the MHC, and set the top-down camera."""
    cmd.reinitialize()
    cmd.set("assembly", "")
    for pdb in ["6AM5", "6AMU"]:
        cmd.load(ROOT / f"inputs/pmhc_tcr_dataset/{pdb}.pdb", pdb)

    # superpose on the MHC heavy chain only, so peptide differences are not absorbed by the fit
    rms = cmd.align("6AMU and chain A and name CA", "6AM5 and chain A and name CA",
                    cycles=0, transform=1)[0]
    print(f"MHC heavy-chain superposition RMSD: {rms:.2f} A")

    groove = np.array(cmd.centerofmass("6AM5 and chain A"))
    tcr = np.array(cmd.centerofmass("6AM5 and chain D+E"))
    pep = cmd.get_coords("6AM5 and chain C and name CA")
    z = tcr - groove; z /= np.linalg.norm(z)             # camera looks back down this axis
    x = pep[-1] - pep[0]                                  # peptide N -> C along screen x
    x = x - np.dot(x, z) * z; x /= np.linalg.norm(x)
    R = np.vstack([x, np.cross(z, x), z])

    centre = np.array(cmd.centerofmass("6AM5 and chain C"))
    # set_view takes the rotation column-major, so the row-stacked camera basis is transposed here
    cmd.set_view(list(R.T.flatten()) + [0.0, 0.0, -120.0] + list(centre) + [40.0, 200.0, -20.0])

    cmd.hide("everything")
    # only the peptide-binding domain of the heavy chain. The alpha3 domain and beta2m sit below the
    # groove and, seen down this axis, would project across the peptide.
    GROOVE = "chain A and resi 1-180"
    for obj in ["6AM5", "6AMU"]:
        cmd.show("cartoon", f"{obj} and {GROOVE}")
        cmd.color("grey85", f"{obj} and {GROOVE}")
        # peptide sits on top of the groove, on the KD ramp
        cmd.show("sticks", f"{obj} and chain C")
        print(f"{obj}: " + "  ".join(f"P{i}{a}={v:+.1f}" for i, a, v in colour_peptide(obj)))

    # NB: transparent cartoon disappears entirely under ray_trace_mode 1/3, so the groove is
    # drawn opaque and pushed pale instead of being made see-through
    cmd.set("cartoon_transparency", 0.0)
    cmd.set("cartoon_fancy_helices", 1)
    cmd.set("stick_radius", 0.22)
    cmd.set("two_sided_lighting", 1)
    # keep the peptide mesh light: thin lines, coarse tessellation, tight probe so the envelope
    # hugs the side chains instead of ballooning over them

    # Hydrogen bonds from the epitope to the groove, as N/O-to-N/O heavy-atom contacts within
    # 3.4 A. PyMOL's mode=2 polar-contact finder needs explicit hydrogens to judge donor geometry,
    # and these crystal files carry none, so it reported only 2 contacts per complex -- far fewer
    # than a 10-mer in a class I groove actually makes. The heavy-atom criterion does not depend
    # on hydrogens being present.
    POLAR = "(name N* or name O*)"
    for obj in ["6AM5", "6AMU"]:
        name = f"hb_{obj}"
        n = cmd.distance(name, f"{obj} and chain C and {POLAR}",
                         f"{obj} and {GROOVE} and {POLAR}", cutoff=3.4, mode=0)
        cmd.set("dash_color", "grey20", name)
        cmd.set("dash_width", 2.6, name)
        cmd.set("dash_gap", 0.30, name)
        cmd.set("dash_radius", 0.05, name)
        cmd.hide("labels", name)

        # a dash that ends on an undrawn atom looks clipped, so the MHC residues it lands on are
        # shown as thin sticks and the bond visibly terminates on something
        partners = f"partners_{obj}"
        cmd.select(partners, f"byres ({obj} and {GROOVE} and {POLAR} within 3.4 of "
                             f"({obj} and chain C and {POLAR}))")
        cmd.show("sticks", f"{partners} and sidechain")
        cmd.show("sticks", f"{partners} and name N+CA+C+O")
        cmd.color("grey60", partners)
        cmd.set("stick_radius", 0.09, partners)
        pairs = cmd.find_pairs(f"{obj} and chain C and {POLAR}",
                               f"{obj} and {GROOVE} and {POLAR}", cutoff=3.4)
        print(f"{obj}: {len(pairs)} peptide-MHC polar contacts <=3.4 A across "
              f"{cmd.count_atoms(f'{partners} and name CA')} MHC residues")
        cmd.deselect()

    HELICES = {"a1": "resi 57-84", "a2": "resi 138-176"}

    # one panel per structure, side by side
    cmd.set("grid_mode", 1)
    for slot, obj in [(1, "6AM5"), (2, "6AMU")]:
        cmd.set("grid_slot", slot, obj)
        cmd.set("grid_slot", slot, f"hb_{obj}")


    # frame the whole groove, not the inside of it
    cmd.zoom("6AM5 and chain C", buffer=20.0)   # room for the labels above and below
    cmd.turn("z", ROLL_DEG)

    # Labels are placed only now, because where "below" is depends on the final camera. get_view
    # returns the rotation column-major, so transposing it gives the camera basis as rows; screen
    # down is then -y_cam, and each label is pushed out along it until it clears the groove.
    # Offsets are measured from the peptide centroid, which is what the camera is centred on, so
    # a label placed this way stays inside the frame. Measuring from the groove centroid instead
    # pushed both label sets outside the rendered area.
    R_cam = np.array(cmd.get_view()[:9]).reshape(3, 3).T
    down, across = -R_cam[1], R_cam[0]
    DROP = 26.0
    for obj in ["6AM5", "6AMU"]:
        pc = np.array(cmd.centerofmass(f"{obj} and chain C"))
        for tag, sel in HELICES.items():
            hc = np.array(cmd.centerofmass(f"{obj} and chain A and {sel}"))
            lateral = float(np.dot(hc - pc, across)) * across   # keep it under its own helix
            ps = f"lbl_{tag}_{obj}"
            cmd.pseudoatom(ps, pos=list(pc + lateral + down * DROP),
                           label=("\u03b11" if tag == "a1" else "\u03b12"))
            cmd.set("label_size", 30, ps)
            cmd.set("label_color", "grey20", ps)
            cmd.show("label", ps)
            cmd.set("grid_slot", 1 if obj == "6AM5" else 2, ps)


def label_sequences() -> None:
    """Write each panel's peptide sequence above its groove, in the same 3D frame as the render."""
    R_cam = np.array(cmd.get_view()[:9]).reshape(3, 3).T
    up = R_cam[1]
    for obj in ["6AM5", "6AMU"]:
        seq = "".join(AA3.get(r, "X") for r in
                      [x[1] for x in sorted(_pep_residues(obj))])
        pc = np.array(cmd.centerofmass(f"{obj} and chain C"))
        ps = f"seq_{obj}"
        cmd.pseudoatom(ps, pos=list(pc + up * 26.0), label=seq)   # sequence only, no PDB id
        cmd.set("label_size", 30, ps)
        cmd.set("label_color", "black", ps)
        cmd.show("label", ps)
        cmd.set("grid_slot", 1 if obj == "6AM5" else 2, ps)
        print(f"{obj}: sequence label {seq}")


def _pep_residues(obj: str) -> list[tuple[int, str]]:
    space = {"rows": []}
    cmd.iterate(f"{obj} and chain C and name CA", "rows.append((int(resi), resn))", space=space)
    return space["rows"]


def tighten(path: Path, gap_frac: float = 0.03) -> None:
    """Pull the two grid panels together.

    PyMOL's grid_mode sizes each cell from the canvas, so with two slots in a wide image each
    structure sits centred in its own half with a lot of white either side and no setting to close
    it. The two panels are separated here on the actual empty gutter between them rather than at the
    midline: with shadows on, the rendered content is not centred in its cell, and cutting at the
    midline sliced straight through the second structure.
    """
    import numpy as _np
    from PIL import Image

    im = Image.open(path).convert("RGB")
    a = _np.asarray(im)
    # 238 rather than 250: soft shadow gradients sit just under pure white and would otherwise
    # count as ink, hiding the real gutter and leaving only the gap before a label to split on
    ink = (a < 238).any(axis=2)
    col_has_ink = ink.any(axis=0)

    # widest run of fully empty columns, searched in the middle half of the image
    w = len(col_has_ink)
    best, run_start, best_run = None, None, 0
    for x in range(w):                          # widest empty run anywhere, not just mid-image
        if not col_has_ink[x]:
            run_start = x if run_start is None else run_start
            if x - run_start + 1 > best_run:
                best_run, best = x - run_start + 1, (run_start, x)
        else:
            run_start = None
    # A real inter-panel gutter is wide. With shadows on, shading tints the whole background and
    # the widest empty run collapses to a sliver -- which on this figure was the gap between the
    # second structure and its sequence label, so splitting there sheared the panel apart. Refuse
    # to crop unless the gap is convincingly a gutter.
    MIN_GUTTER = 0.05
    if best is None or best_run < MIN_GUTTER * w:
        print(f"  no clear gutter (widest empty run {best_run}px < {MIN_GUTTER:.0%} of {w}px); "
              f"leaving panel spacing unchanged")
        return
    cut = (best[0] + best[1]) // 2

    halves = []
    for box in [(0, 0, cut, a.shape[0]), (cut, 0, w, a.shape[0])]:
        part = im.crop(box)
        sub = ink[:, box[0]:box[2]]
        rows, cols = _np.where(sub)
        if len(rows) == 0:
            continue
        halves.append(part.crop((cols.min(), rows.min(), cols.max() + 1, rows.max() + 1)))

    gap = int(max(x.width for x in halves) * gap_frac)
    out_w = sum(x.width for x in halves) + gap * (len(halves) - 1)
    out_h = max(x.height for x in halves)
    canvas = Image.new("RGB", (out_w, out_h), (255, 255, 255))
    x0 = 0
    for part in halves:
        canvas.paste(part, (x0, (out_h - part.height) // 2))
        x0 += part.width + gap
    canvas.save(path)
    print(f"  tightened -> {out_w}x{out_h} (gutter at x={cut}, {best_run}px empty, gap {gap}px)")


def render(shadows: bool) -> Path:
    """Raytrace the two-panel view; black contours come from ray_trace_mode 1."""
    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 1)
    cmd.set("antialias", 2)
    # ray_trace_mode 3 is the cel-shaded/poster mode: quantised flat fill plus a black contour.
    # Mode 2 is outline-ONLY -- it discards colour completely, which is not what "cartoony" means
    # here. Killing specular and reflection and raising ambient takes the plastic sheen off, which
    # is what makes it read as matte rather than glossy-with-an-outline.
    cmd.set("ray_trace_mode", 3)
    cmd.set("ray_trace_color", "black")
    cmd.set("ray_trace_gain", 0.25)
    cmd.set("specular", 0.0)
    cmd.set("reflect", 0.0)
    cmd.set("ambient", 0.45)
    cmd.set("direct", 0.55)
    cmd.set("spec_reflect", 0.0)
    cmd.set("ray_shadows", 1 if shadows else 0)
    tag = "matte_shadows" if shadows else "matte_noshadows"
    out = OUT_DIR / f"fig_panel8_6am_peptide_hydrophobicity_2panel_{tag}.png"
    cmd.png(str(out), width=2600, height=1400, dpi=300, ray=1)
    print(f"wrote {out}")
    try:
        tighten(out)
    except ImportError:
        print("  (PIL unavailable here; run with --tighten in the base env to close the gap)")
    return out


def main() -> None:
    build_scene()
    label_sequences()
    for shadows in [True, False]:
        render(shadows)
    cmd.save(str(OUT_DIR / "6am_superposed_on_mhc.pse"))


def annotate() -> None:
    """Composite a hydrophobicity colourbar and a structure key onto the raytraced render.

    Kept separate from the PyMOL pass because text baked by PyMOL rasterises at whatever the render
    resolution happens to be; drawn here it stays crisp and the scale can carry real units. The
    raytraced PNG is trimmed of its white border first, otherwise the structure floats in the middle
    of a mostly empty frame.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as _np
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.lines import Line2D
    from PIL import Image, ImageChops

    src = OUT_DIR / "fig_panel8_6am_peptide_hydrophobicity_top.png"
    im = Image.open(src).convert("RGB")
    bg = Image.new("RGB", im.size, (255, 255, 255))
    bbox = ImageChops.difference(im, bg).getbbox()
    if bbox:                                    # keep a small margin around the trimmed content
        pad = 24
        bbox = (max(0, bbox[0] - pad), max(0, bbox[1] - pad),
                min(im.width, bbox[2] + pad), min(im.height, bbox[3] + pad))
        im = im.crop(bbox)
    arr = _np.asarray(im)
    h, w = arr.shape[:2]

    strip = 0.19                                # fraction of the canvas reserved for annotation
    fig = plt.figure(figsize=(w / 200, (h / 200) / (1 - strip)))
    ax = fig.add_axes([0.0, strip, 1.0, 1.0 - strip]); ax.imshow(arr); ax.axis("off")

    cmap = LinearSegmentedColormap.from_list(
        "kd", [(0.20, 0.40, 0.68), (0.92, 0.92, 0.90), (0.85, 0.55, 0.05)])
    cax = fig.add_axes([0.37, strip * 0.60, 0.26, strip * 0.10])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(-4.5, 4.5), cmap=cmap),
                      cax=cax, orientation="horizontal")
    cb.set_ticks([-4.5, 0, 4.5])
    cb.set_ticklabels(["$-$4.5\nhydrophilic", "0", "+4.5\nhydrophobic"])
    cb.ax.tick_params(labelsize=8, length=2)
    cax.set_title("Kyte-Doolittle hydrophobicity", fontsize=9, pad=4)

    key = [Line2D([0], [0], color="0.30", lw=5.0, label="6AM5  SMLGIGIVPV"),
           Line2D([0], [0], color="0.55", lw=2.2, alpha=0.55, label="6AMU  MMWDRGLGMM")]
    fig.legend(handles=key, loc="center left", bbox_to_anchor=(0.015, strip * 0.5),
               frameon=False, fontsize=9, handlelength=2.4, labelspacing=0.7,
               title="crystallised peptide", title_fontsize=9)
    fig.text(0.985, strip * 0.52,
             "top-down view along the TCR approach axis,\n"
             "superposed on the MHC heavy chain\n(C$\\alpha$ RMSD 0.81 $\\AA$)",
             ha="right", va="center", fontsize=8, color="0.40", linespacing=1.6)

    out = OUT_DIR / "fig_panel8_6am_peptide_hydrophobicity_top_annotated.png"
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"wrote {out}  ({w}x{h} render, trimmed)")


if __name__ == "__main__":
    # pass 1 (pymolviz env, has pymol):   render the raytraced view
    # pass 2 (base env, has matplotlib):  composite the colourbar and key
    if not ANNOTATE_ONLY:
        main()
    else:
        annotate()
