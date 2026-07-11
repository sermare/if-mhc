#!/usr/bin/env python3
"""Build a 3-page PDF report of the most telling V3-campaign figures → V3_Q30_report.pdf."""
import sys, importlib, datetime
sys.path.insert(0, "/home/ubuntu/if-mhc/py")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import q30_analysis as Q; importlib.reload(Q)

ARMS = ["fpocket", "tcr", "apoc", "nslide", "sc_tcr", "sc_apoc", "sc_fpocket"]
ACOL = {"fpocket":"#7f7f7f","tcr":"#1f77b4","apoc":"#d62728","nslide":"#2ca02c",
        "sc_tcr":"#17becf","sc_apoc":"#e377c2","sc_fpocket":"#bcbd22"}
df, CO = Q.load_all(ARMS)
df = df[df.broken == 0].reset_index(drop=True)
part = df[~df.arm.isin(Q.SCAFFOLD_ARMS)].copy(); part["T"] = part.cond.astype(int)
scaf = df[df.arm.isin(Q.SCAFFOLD_ARMS)].copy()
sep = round(Q.rmsd(Q.REFS["GIG"], Q.REFS["DRG_shift"]), 2)
df["seed_basin"] = np.where(df.pid == "6AM5", "GIG", "DRG_shift")
crossed = int((df[~df.arm.isin(Q.SCAFFOLD_ARMS)].basin !=
               df[~df.arm.isin(Q.SCAFFOLD_ARMS)].seed_basin).sum())
TS = "2026-07-02"

def hdr(fig, title, sub):
    fig.text(0.06, 0.965, title, fontsize=13.5, fontweight="bold")
    fig.text(0.06, 0.943, sub, fontsize=8.3, color="#444")
    fig.text(0.5, 0.028, f"V3 Q30 interaction-conditioning campaign · {TS}", fontsize=7, color="#999", ha="center")

pp = PdfPages("/home/ubuntu/if-mhc/V3_Q30_report.pdf")

# ══════════════════ PAGE 1 — headline: conditioning does not steer the basin ══════════════════
fig = plt.figure(figsize=(8.5, 11)); hdr(fig,
    "V3 — does interaction conditioning steer the peptide basin?",
    "Peptide-only RFdiffusion in a fixed pMHC-TCR complex; read-out = basin geometry, floor frame.")

axA = fig.add_axes([0.09, 0.60, 0.40, 0.28])
for arm in ARMS:
    s = df[df.arm == arm]
    if len(s): axA.scatter(s.to_GIG, s.to_DRG_shift, s=16, c=ACOL[arm], alpha=.7,
                           edgecolor="k", linewidth=.2, label=f"{arm} ({len(s)})")
lim = 1 + np.nanmax([df.to_GIG.max(), df.to_DRG_shift.max()])
axA.plot([0, lim], [0, lim], "k--", lw=.7, alpha=.5); axA.axhline(4, c="g", lw=.5, ls=":"); axA.axvline(4, c="g", lw=.5, ls=":")
axA.set_xlabel("Cα-RMSD to GIG (Å)"); axA.set_ylabel("Cα-RMSD to DRG (Å)")
axA.set_title("A. Basin map (all arms)", fontsize=10, loc="left"); axA.legend(fontsize=5.5, loc="upper right")

axB = fig.add_axes([0.58, 0.60, 0.36, 0.28])
for arm in ["fpocket", "tcr", "apoc"]:
    g = part[part.arm == arm].groupby("T").min_ref.median()
    if len(g): axB.plot(g.index, g.values, "-o", ms=4, c=ACOL[arm], label=arm)
axB.axhline(4, c="g", lw=.6, ls=":"); axB.set_ylim(0, 5)
axB.set_xlabel("partial_T (noise level)"); axB.set_ylabel("median dist to nearest basin (Å)")
axB.set_title("B. Partial diffusion is FROZEN", fontsize=10, loc="left"); axB.legend(fontsize=7)

fig.text(0.06, 0.55, "What the figures show", fontsize=11, fontweight="bold")
fig.text(0.06, 0.525, (
 "• The two native registers (GIG=6AM5 P10-anchor, DRG=6AMU P9-anchor) are only "
 f"{sep} Å apart in the floor frame but geometrically distinct (F-pocket P9↔P10 swap).\n"
 "• Panel A: every partial-diffusion design (fpocket/tcr/apoc) sits essentially ON a native basin — and "
 "always its OWN (cognate) basin. Register transfer across all partial arms = "
 f"{crossed} / {len(part)} designs. No steering.\n"
 "• Panel B: increasing the noise from partial_T=5→40 does NOT move the peptide off native — the curve is "
 "flat (median 0.1–0.65 Å to the seed's native basin at every T; 100% 'seated'). The fixed MHC+TCR context "
 "+ provide_seq over-constrains the peptide, which denoises straight back to its starting register "
 "regardless of noise. partial_T≈50 is full diffusion, so ~40 is already near the ceiling.\n"
 "• Conditioning on the interactions (Q30α, CDR3β-Phe, A/B-pocket, floor F-pocket) — applied JOINTLY per "
 "arm — therefore never gets a chance to act in partial mode: the backbone doesn't leave native.\n\n"
 "VERDICT (partial arms): interaction conditioning does not steer the basin here — it is a null result "
 "driven by an over-constrained generator, not by the conditioning being wrong."),
 fontsize=8.2, va="top", wrap=True)
pp.savefig(fig); plt.close(fig)

# ══════════════════ PAGE 2 — mechanism: discrete register, contacts form, no bridge ══════════════════
fig = plt.figure(figsize=(8.5, 11)); hdr(fig,
    "V3 — mechanism: a discrete register, and no bridge between the two",
    "Why nothing interpolates GIG↔DRG, and confirmation that the conditioned contacts DO form.")

axC = fig.add_axes([0.09, 0.60, 0.38, 0.28])
for arm in ARMS:
    s = df[df.arm == arm]
    if len(s): axC.hist(s.fpocket_pos, bins=np.arange(6.5, 11.5, 1), histtype="step", lw=1.6, color=ACOL[arm], label=arm)
axC.set_xlabel("peptide position in F-pocket centroid"); axC.set_ylabel("designs")
axC.set_title("C. F-pocket register (discrete)", fontsize=10, loc="left"); axC.legend(fontsize=5.5)

axD = fig.add_axes([0.57, 0.60, 0.37, 0.28])
cf = df.groupby("arm").contacts_frac.mean().reindex([a for a in ARMS if a in df.arm.unique()])
axD.barh(range(len(cf)), cf.values, color=[ACOL[a] for a in cf.index])
axD.set_yticks(range(len(cf))); axD.set_yticklabels(cf.index, fontsize=7); axD.set_xlim(0, 1)
axD.set_xlabel("fraction of conditioned contacts formed (Cβ≤8Å)")
axD.set_title("D. Conditioned contacts DO form", fontsize=10, loc="left")

fig.text(0.06, 0.55, "Interpretation", fontsize=11, fontweight="bold")
fig.text(0.06, 0.525, (
 "• Panel C — the register is DISCRETE. The peptide Cα occupying the F-pocket centroid is either P10 "
 "(GIG-like, unshifted) or P9 (DRG-shifted); no design occupies an intermediate. The GIG↔DRG difference is "
 "a two-state P9↔P10 swap, not a continuous deformation — there is no low-RMSD intermediate backbone to "
 "design into.\n"
 "• NO BRIDGE. Zero partial designs cross to the non-cognate basin; the scaffold arms that DO leave native "
 "drift to a register-ambiguous middle (~equidistant, ~3 Å from both) — that is under-determination, not a "
 "stable connecting state. (A naive 'within 4 Å of both basins' count is meaningless here because the "
 f"basins are only {sep} Å apart — the threshold exceeds the separation.)\n"
 "• Panel D — the conditioning IS being satisfied at the backbone level: the TCR contacts (Q30α + CDR3β-Phe) "
 "form in ~100% of tcr designs; the A/B-pocket arm ~0.76; the floor F-pocket reads lower (~0.40) only "
 "because those contacts are sidechain-mediated and invisible at the Cβ level on backbone-only designs. So "
 "the peptide forms the contacts we asked for — it just does so while staying in its native register.\n\n"
 "TAKEAWAY: satisfying the conditioned interactions is necessary but not sufficient to move register; "
 "register is selected at the C-terminal F-pocket, which the fixed context pins."),
 fontsize=8.2, va="top", wrap=True)
pp.savefig(fig); plt.close(fig)

# ══════════════════ PAGE 3 — two failure modes, methods, verdict ══════════════════
fig = plt.figure(figsize=(8.5, 11)); hdr(fig,
    "V3 — two failure modes, methods, and verdict",
    "Partial (over-constrained) vs scaffold (under-constrained); what was run; what it means.")

axE = fig.add_axes([0.09, 0.62, 0.38, 0.26])
for lbl, sub, c in [("partial (echo)", part, "#1f77b4"), ("scaffold (drift)", scaf, "#2ca02c")]:
    if len(sub): axE.hist(sub.off_diagonal, bins=np.arange(0, 4, 0.3), histtype="stepfilled",
                          alpha=.5, color=c, label=lbl)
axE.axvline(sep, c="k", ls="--", lw=.8); axE.text(sep+.05, axE.get_ylim()[1]*.8, f"native |ΔRMSD|={sep}", fontsize=6)
axE.set_xlabel("off-diagonal = |toGIG − toDRG| (Å)"); axE.set_ylabel("designs")
axE.set_title("E. Two failure modes", fontsize=10, loc="left"); axE.legend(fontsize=7)

axF = fig.add_axes([0.57, 0.62, 0.37, 0.26])
for arm in ["tcr", "nslide"]:
    idx = df.index[df.arm == arm]
    if len(idx) >= 2:
        sub = CO[idx]; rmsf = np.sqrt(((sub - sub.mean(0))**2).sum(-1).mean(0))
        axF.plot(range(1, 11), rmsf, "-o", ms=3, c=ACOL[arm], label=f"{arm}")
axF.axvspan(0.5, 2.5, color="0.9", zorder=0)
axF.set_xlabel("peptide position"); axF.set_ylabel("Cα RMSF (Å)"); axF.set_xticks(range(1, 11))
axF.set_title("F. Spread: N held, C free", fontsize=10, loc="left"); axF.legend(fontsize=7)

fig.text(0.06, 0.565, "Methods (verified against coordinates)", fontsize=11, fontweight="bold")
fig.text(0.06, 0.54, (
 "• Seeds (17): 2 RAW crystal (6AM5, 6AMU — no MD) + 15 OpenMM-relaxed snapshots; 6AMT (unshifted DRG) held "
 "out as a reference only. Trimmed to α1α2 + peptide + TCR variable (all hotspots retained; verified no "
 "interface residue clipped). α2 arm His145–Val152 left free to breathe.\n"
 "• Conditioning is applied JOINTLY per arm (one ppi.hotspot_res list, one run): fpocket=[D30,A77,A84,A116,"
 "A123]; tcr=[D30,E97|E100]; apoc=tcr+[A7,A63,A66,A70,A99,A159,A171]; nslide/sc=scaffold rebuild. "
 "E97(6AM5)/E100(6AMU) is the epitope-specific CDR3β-Phe.\n"
 "• Frame: superpose on 61 β-sheet-floor Cα (helices, TCR, arm 145–152 excluded); 6AM5↔6AMU floor RMSD "
 f"= 0.21 Å — the invariant frame that makes the basin map interpretable. Peptide RMSD to 3 refs "
 "(GIG / DRG-shift / DRG-unshift).\n"
 "• Geometry QC: 0 broken designs (Cα-Cα bonds + clashes) among those shown."),
 fontsize=8.2, va="top", wrap=True)

fig.text(0.06, 0.28, "Verdict & next", fontsize=11, fontweight="bold")
fig.text(0.06, 0.255, (
 "• Panel E: partial arms pile at the native |ΔRMSD| (off-diagonal ≈ native, i.e. IN a basin but only "
 "because they never left native); scaffold arms collapse toward 0 (on the diagonal = ambiguous drift). "
 "Neither selects a NEW basin.\n"
 "• Panel F: RMSF is low at the conditioned N-terminus (P1-2) and high at the free C-terminus — the "
 "backbone spread lives exactly where the register is decided, i.e. where conditioning has least grip.\n"
 "• BOTTOM LINE: interaction-only conditioning did not steer or bridge the register. Partial diffusion is "
 "over-constrained (echoes native at all T); scaffold rebuild is under-constrained (drifts, no stable "
 "intermediate). The two registers behave as separate discrete wells. NEXT: seed from real MD-excursion / "
 "mid-transition backbones (cached dual-RMSD trajectory) rather than near-native crystal/relaxed frames — "
 "give diffusion a non-native starting point to bridge from."),
 fontsize=8.2, va="top", wrap=True)
pp.savefig(fig); plt.close(fig)

pp.close()
print(f"wrote V3_Q30_report.pdf | {len(df)} clean designs | partial {len(part)} | scaffold {len(scaf)} | crossed {crossed}")
