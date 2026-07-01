Q30-conditioned peptide-design campaign (Q30α / CDR1α)

 Context
                                                                                                                                                                                                                                                                                    Prior campaigns showed the noMHC ProteinMPNN model recovers the cognate peptide core only on
   near-native backbones and never recovers the MHC anchors, and that no RFdiffusion conditioning
   captured either epitope. This new campaign tests a single, mechanistically-motivated TCR contact
   — the conserved Gln30 of CDR1α (chain D, residue 30 = GLN in both 6AM5 and 6AMU), the only
 conserved TCR–peptide contact — as the organizing constraint, and asks whether the register-defining                                                                                                                                                                                             second Q30 contact (peptide p3 for GIG / p4 for DRG) can be driven while the shared
 N-terminal anchor p2 is held.

   Two reconciled framings from the clarifications:
   - Track A (direct): design the peptide against the full MHC+TCR complex as fixed context
   (Q30 present but not isolated), with the table's per-position FIX/FREE scheme, on the existing
 native + relaxed backbones.
 - Track B (Q30-hotspot): express "condition only on Q30" as an RFdiffusion partial-diffusion
 hotspot (ppi.hotspot_res=[D30], sole hotspot) that re-diffuses the peptide toward the Q30
   contact, then MPNN-designs the freed core with the anchor + register-contact positions held.

   Model: noMHC proteinmpnn_nomhc for all sequence design (MHC stays in the PDB as context but the
   MHC-blind model effectively down-weights it). Sampling temp 0.3.
