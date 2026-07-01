╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
0 + F-pocket dual-hotspot basin campaign

ntext

e conserved Gln30 of CDR1α (chain D res 30 = GLN in both crystals) is the only conserved
R–peptide contact and pins the peptide N-terminus (p2/p3). But the GIG↔DRG basin is selected at
e C-terminus (p9–p10 F-pocket swap). A single N-terminal hotspot is underdetermined by
nstruction: constraining Q30 alone leaves the C-terminus free — the exact drift that put the prior
   RFdiffusion runs on the toGIG≈toDRG diagonal. So Q30 must be paired with a C-terminal F-pocket
 hotspot to pin both ends and let a peptide select a basin.                                                                                                                                                                                                                                        
 Decisions from the user's geometric correction:
 - Dual hotspot (both epitopes, same set): ppi.hotspot_res=[D30, A77, A80, A116, A143] —
   Q30 (N-term) + floor F-pocket (C-term). Floor residues only; the His145–Val152 α2 arm is left free
   to breathe (per the table: locking it kills DRG).
   - Route everything through RFdiffusion. Drop the direct-MPNN-on-relaxed track (pure MPNN can't move
 a backbone between basins, so it can't test the hypothesis).
 - Score on basin geometry, not recovery — RMSD to the GIG/DRG basins on the existing map
 (toGIG/toDRG from core_load). Success = diffused backbones coming off the diagonal toward a
 defined basin (min(toGIG,toDRG) small), vs the prior sole-conditioning diagonal.
 - Design on relaxed/diffused backbones, not the rigid crystal. The crystal is the diffusion seed
 only. Seed DRG from 6AMU-bound (+ its 6AMU_snap* relaxed frames); exclude 6AMT. Seed GIG
 from 6AM5 + 6AM5_snap*.
