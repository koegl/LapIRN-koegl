# Findings

Current evidence and decisions for the JMI paper.

Checkbox meaning:

- [x] Current and already represented in the manuscript, or explicitly resolved as an omission.
- [ ] Still outstanding or dependent on new results.

Unless stated otherwise, values are from the corrected official scorer on the test set: 28 patients, 38 registration pairs, and 26 patients with lesions in the moving scan.

## Paper story

- [x] **Problem.** Follow-up-to-baseline PSMA PET/CT registration should align anatomy while limiting registration-induced changes in metabolic tumor volume (MTV) and total lesion glycolysis (TLG).
- [x] **Primary question.** Determine how PET-preservation and bone-rigidity losses trade anatomical alignment for lower PET change and whether the losses should act during network training, instance optimization (IO), or both.
- [x] **Selected method.** “Ours” is the network trained without the PET-preservation or rigidity losses, followed by IO with both groups of losses.
- [x] **Selection basis.** The method was selected on the validation set by minimizing $|\Delta\mathrm{MTV}|+|\Delta\mathrm{TLG}|$ among candidates within two Dice points of the most accurate candidate.
- [x] **Main result.** Compared with the same network before IO, the selected method gained 2.9 Dice points without a detected difference in scan-level MTV or TLG change.
- [x] **Placement result.** Applying both loss groups only during IO gained 1.0 Dice point over applying them during training and IO; the data do not establish equivalent preservation between these placements.
- [x] **Time-dependent interpretation.** Loss-guided IO is appropriate when predicted masks and about 2.5 minutes per pair are available; without IO, training with the losses lowers mean PET change at a cost in alignment, with a detected reduction only for MTV.
- [x] **Limits of preservation.** Residual scan-level change is associated with incomplete predicted-mask coverage, cancellation between lesions, and mismatch between the soft optimized quantities and nearest-neighbor evaluation.
- [x] **Per-lesion result.** Near-zero scan-level MTV change does not imply preservation of individual lesions; even with provided lesion masks, the median per-lesion change remained 13.5%.
- [x] **Rigidity result.** The per-bone rigid fit preserved within-bone distances best, at a cost in bone Dice; lower distortion does not establish correct bone placement.
- [x] **Scope of the claim.** The study measures registration-induced changes and label-based alignment, not downstream clinical benefit or clinical outcomes.

## Effect and placement of the losses

- [x] **Network output, PET losses.** Training with the PET losses lowered $|\Delta\mathrm{MTV}|$ from 10.3% to 8.2% (significant) and $|\Delta\mathrm{TLG}|$ from 12.5% to 9.8% (not significant), at a cost of 1.9 Dice points.
- [x] **IO, PET losses.** With PET losses in training and IO, MTV and TLG changes were 8.2% and 10.3%, compared with 11.3% and 14.5% without the losses; the preservation differences were not significant after correction.
- [x] **Rigidity alone.** Rigidity significantly reduced network-output MTV change from 10.3% to 8.9% and IO TLG change from 14.5% to 10.6% relative to the corresponding unconstrained variants.
- [x] **Both loss groups.** Combining the PET and rigidity losses produced mean PET changes close to the PET-only configuration and the lowest Dice among the four training configurations.
- [x] **IO-only versus training and IO.** With both groups, IO-only minus training-and-IO was +1.0 Dice points (95% CI 0.4 to 1.6), +0.8 MTV points (−0.4 to 2.0), and +1.0 TLG points (−0.3 to 2.6).
- [x] **No equivalence claim.** None of the preservation differences between IO-only and training-and-IO was significant after Holm correction, but the intervals do not establish equivalence.
- [x] **IO without preservation losses.** IO without the losses raised mean MTV change from 10.3% to 11.3% and TLG change from 12.5% to 14.5% relative to network output.

## Comparison with baselines

- [x] **Current draft values.** Ours reaches 78.3% CT Dice and 6.89 mm HD95, compared with NiftyReg at 71.8% and 7.88 mm and ConvexAdam at 60.3% and 9.32 mm (re-tuned with the selection rule: NiftyReg be0.0002_sx7, ConvexAdam lambda2; ranking-chosen results kept as B3_ranking / B4_ranking).
- [x] **Current preservation comparison.** No difference in MTV or TLG change was detected between ours (8.7% and 11.3%), NiftyReg (10.7% and 12.6%), and ConvexAdam (8.5% and 9.3%).
- [x] **IO needs the network.** IO directly after affine pre-registration reached 56.6% Dice and did not replace the learned registration within the same iteration budget.
- [x] **Ranking interpretation.** IO without the losses ranked first, ours second, and losses in training and IO third, with overlapping bootstrap rank intervals; accuracy largely determined the ranking.
- [x] **Runtime.** The network requires about 11 seconds per pair, ConvexAdam about 11 seconds, NiftyReg 128 seconds (A6000, val pairs), and IO adds about 2.5 minutes.
- [x] **Qualitative case.** Case 0146 is presented as an alignment example selected for its tenfold tumor-volume change, not as evidence of PET preservation; ours changed MTV by +2.6% and TLG by +6.6%, versus −0.6% and +2.3% with unconstrained IO.
- [x] **Replace the baseline values after re-tuning.** Update the abstract, Results, tables, Discussion if needed, and all derived comparisons once the final NiftyReg and ConvexAdam settings are available.

Sources: `evaluation/test/`, `journal_results/paper/table_baselines.py`, and `overleaf_jmi/tables/baselines.tex`.

## Accuracy–preservation trade-off and mask quality

- [x] **Status of the sweep.** The low-weight sweep is a post-hoc test-set sensitivity analysis and was not used to replace the validation-selected factor $\times0.1$.
- [x] **Tiny PET weight.** At $\times10^{-4}$, evaluated MTV change was 9.3%, the same as at $\times0.1$, with 1.6 more Dice points; TLG change was 12.9% rather than 11.8%.
- [x] **Validation-grid limitation.** The original validation grid contained no nonzero PET factor below $\times0.1$.
- [x] **Selection consequence.** The test-set sweep cannot justify selecting a smaller factor; that would require an expanded validation sweep and independent confirmation.
- [x] **Predicted masks.** At the tuned factor, the loss reduced MTV change on the predicted lesion masks to about 0.8%, while 9.3% remained on the provided lesions.
- [x] **Provided-mask diagnostic.** Using provided lesion masks during IO lowered scan-level MTV change from 9.3% to 0.1% and TLG change from 11.8% to 4.5%, while Dice rose from 78.6% to 79.1%.
- [x] **Higher weights.** Increasing the provided-mask PET factor from $\times0.1$ to $\times1$ changed MTV from 0.1% to 0.2% and TLG from 4.5% to 4.0%, so larger weights did not resolve the remaining uptake change.
- [x] **Diagnostic status.** Provided lesion masks are unavailable in the intended test-time setting and are used only to identify limitations of predicted-mask IO.

Sources: `evaluation/test/W*.csv`, `evaluation/test/G1.csv`, `evaluation/test/G2.csv`, and `journal_results/paper/figure_tradeoff.py`.

## Lesion coverage and recall-biased masks

- [x] **Coverage groups.** Of 1435 lesions, 214 were missed by the predicted mask, 155 were partly covered, and 1066 were covered; these groups contained 6%, 6%, and 88% of lesion volume.
- [x] **Covered lesions.** PET IO reduced change in covered lesions from 9.1% to 6.0% for MTV and from 11.0% to 8.4% for TLG.
- [x] **Missed-lesion contribution.** Missed lesions contributed 5.2 of the remaining 9.3 MTV points and 6.2 of 11.8 TLG points, despite accounting for only 6% of lesion volume.
- [x] **Covered-lesion contribution.** Covered lesions still contributed 3.3 MTV points and 5.4 TLG points after PET IO.
- [x] **Patient weighting.** In the third of patients with the lowest lesion volume, missed lesions made up 33% of lesion volume and caused 72% of the missed-lesion contribution.
- [x] **Recall-biased mask.** On moving test scans with lesions, the recall-biased mask increased mean lesion detection from 74% to 91% and reduced voxel-level precision from 0.71 to 0.22.
- [x] **Recall-biased-mask result.** Compared with the nnU-Net mask, MTV change was 10.3% rather than 9.3%, TLG change 12.5% rather than 11.8%, Dice 79.3% rather than 78.6%, and per-lesion median 13.5% rather than 13.7%; none of the scan-level differences was significant.
- [x] **Interpretation.** The tested recall-biased mask did not improve measured preservation, but the experiment does not show that recall gains are ineffective if precision can be retained.

Sources: `journal_results/paper/figure_detection.py`, `table_detection_groups.py`, `table_detection_contributions.py`, and the corrected stage-10 recall-mask evaluation.

## Per-lesion change and per-lesion loss terms

- [x] **Denominators.** The moving scans contain 1435 lesions; the size/location analysis excludes 108 components below five voxels and analyzes 1327 lesions from 26 patients.
- [x] **Small lesions.** Small lesions change most because one or two voxels are already a large relative change; their medians are correspondingly quantized.
- [x] **IO without losses.** IO without preservation losses increased per-lesion distortion, especially for small bone-associated lesions.
- [x] **Rigidity by location.** Rigidity primarily reduced change in bone-associated lesions, consistent with its construction.
- [x] **PET losses by location.** PET losses acted in both bone-associated and soft-tissue lesions.
- [x] **Scan totals hide lesion changes.** Growth and shrinkage of different lesions can cancel in scan-level MTV and TLG.
- [x] **Provided masks do not solve per-lesion change.** With provided masks, scan-level MTV change was 0.1%, while median per-lesion change remained 13.5%.
- [x] **Per-lesion terms.** The test-time experiment constrains the eight largest predicted components with at least 20 voxels using component-wise MTV, TLG, and mean-Jacobian terms.
- [x] **Validation selection.** Validation selected ten times the base per-lesion weight and reduced median per-lesion MTV change from 12.5% to 11.3%.
- [x] **Negative test result.** On the test set, median per-lesion MTV change moved only from 13.7% to 13.3%, improved for 5 of 26 patients, and was higher for small lesions (20.0% versus 16.7%).
- [x] **Scan-level effect.** With per-lesion terms, mean scan-level MTV remained 9.3%, TLG changed from 11.8% to 12.0%, and Dice rose by 0.6 points.
- [x] **Provided-mask per-lesion diagnostic.** At the base weight, the per-lesion terms left the median unchanged at 13.5% when provided masks were used.

Source: `journal_results/paper/figure_lesions.py`, `lesions_summary.txt`, and the per-lesion-term stage-5-to-stage-7 evaluations.

## Contribution of individual PET terms

- [x] **Analysis status.** The leave-one-out comparison is a diagnostic test-set analysis and was not used to select a reduced configuration.
- [x] **Mean-Jacobian term.** Removing $\mathcal{L}_{\overline{J}}$ increased MTV change from 9.3% to 10.4% and TLG change from 11.8% to 13.5%; both differences were significant.
- [x] **Voxel-wise Jacobian term.** Removing $\mathcal{L}_{J}$ significantly increased TLG change to 12.3%, while MTV remained 9.3%.
- [x] **Direct MTV term.** Removing $\mathcal{L}_{\mathrm{MTV}}$ increased Dice from 78.6% to 79.3% and reduced TLG change to 10.7%; both differences were significant, while its MTV difference was not.
- [x] **Direct TLG term.** Removing $\mathcal{L}_{\mathrm{TLG}}$ significantly reduced TLG change to 9.8%; Dice and MTV differences were not significant.
- [x] **Bounded interpretation.** The test-set result does not support a claim that all four PET terms improve the evaluated endpoints.
- [x] **Possible mechanism.** The direct terms operate on soft, linearly warped masks and PET, whereas evaluation uses nearest-neighbor warping; this is a plausible but unproven explanation.
- [x] **Required follow-up.** Selecting a reduced loss or a closer evaluation surrogate would require validation tuning and independent confirmation.

Source: `overleaf_jmi/tables/loo.tex` and the corresponding corrected test-set outputs.

## Bone rigidity

- [x] **Stencil leakage.** Across all 560 labeled challenge scans, 21.2% of finite-difference stencil reads left the bone of the center voxel and therefore also constrained surrounding tissue or neighboring bones.
- [x] **Effect of erosion.** Erosion removed stencil leakage but retained only 35.5% of bone voxels and removed 23% of bones entirely.
- [x] **Ribs.** Erosion retained 2.4% of rib voxels and removed 58% of ribs entirely.
- [x] **Validation selection.** Within the two-Dice-point tolerance, validation selected $\times3$ for Staring and the grid maximum $\times100$ for Staring with erosion and the per-bone rigid fit.
- [x] **Within-bone distortion.** On the test set, the per-bone rigid fit reduced short-distance distortion from 13.4% to 5.0% and long-distance distortion from 4.4% to 1.8%.
- [x] **Comparison with Staring.** The rigid fit preserved within-bone distances better than either Staring variant; Staring with erosion constrained the bones least.
- [x] **Alignment cost.** The rigid fit and Staring reduced CT Dice by 1.6 and 2.0 points and bone Dice by about four points; soft-tissue Dice was retained.
- [x] **Folding.** Staring reduced NDV inside bones, whereas the rigid fit did not significantly change it.
- [x] **PET effect on top of PET losses.** None of the three rigidity penalties produced a detected additional improvement in MTV or TLG preservation when added to the PET losses.
- [x] **Rigidity alone.** Rigidity alone improved particular PET endpoints in the main ablation, so the manuscript does not claim that rigidity never helps PET preservation.
- [x] **Interpretive limit.** Lower within-bone distance distortion establishes better preservation of bone shape, but not correct bone placement.

Source: `journal_results/erosion_leakage.py`, `journal_results/erosion_leakage/`, the corrected whole-bone rigidity sweep, and `overleaf_jmi/tables/rigidity_full.tex`.

## Deformation regularity

- [x] **NiftyReg transfer.** The validation-selected NiftyReg configuration exceeded the 0.005% NDV threshold on 8 of 38 validation pairs and 13 of 38 test pairs, reaching up to 0.51% NDV; it folded significantly more than ours.
- [x] **Reason for selection.** The selection rule weighs Dice against MTV/TLG but not folding, so it chose a NiftyReg setting that folds above the threshold.
- [x] **Selected method.** The selected method exceeded the NDV threshold on 2 of 38 test pairs (76 and 125 ppm).
- [x] **Training plus IO.** The variant with both loss groups in training and IO exceeded the threshold on 1 of 38 test pairs (262 ppm), the maximum among our reported variants.
- [x] **Aggregation.** NDV means and standard deviations are calculated from patient-level averages, whereas threshold exceedances count individual registration pairs.

Source: `evaluation/test/B3.csv` and `journal_results/baselines/niftyreg/eval/val/`.

## Resolved omissions and outstanding work

- [x] **Early stopping remains omitted.** Selecting the T1 checkpoint by tumor preservation cost 3.0 Dice points and reduced MTV and TLG change by only 0.4 and 0.9 points, neither significant. Without the training curves, this experiment is disconnected from the current story.
- [x] **Old violin plot remains omitted.** The patient-level baseline violins did not add clear information beyond the table because the distributions overlap strongly.
- [x] **Historical pre-correction results are excluded.** The old validation mask experiment, old align-corners results, interior-only rigidity metric, and pre-correction recall-mask claim must not be used as manuscript evidence.
- [ ] **Finalize re-tuned baseline results.** Replace all temporary baseline values and remove the abstract TODO.
- [ ] **Run the final whole-paper consistency pass.** Recheck the abstract, main text, tables, figures, captions, appendices, and this file after every final result is frozen.

Historical sources retained only for provenance: `journal_results/io_pet_check.py`, `io_pet_check/summary.csv`, `journal_results/recall_masks.py`, `recall_masks/summary.csv`, and the old `rigidity_sweep_interior_metric/` outputs.
