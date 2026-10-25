# Paper editing checklist for the JMI extension

Re-reviewed 2026-10-25 after the corrected `align_corners` evaluations, using `main.tex` and its included manuscript, tables, figures, appendices, `../findings.md`, and the six supplied JMI examples (`024502`, `024503`, `034001`, `034002`, `044002`, `044003`). The current PDF is 57 double-spaced pages including references, appendices, and lists of figures and tables. The examples show presentation patterns, not current submission rules. Keep the `#NN` identifiers when discussing items. Reopened boxes mark earlier edits that need review against the new evidence.

**Recommended story.** Longitudinal PSMA PET/CT registration should improve anatomical correspondence without changing measurements of the *same* follow-up lesions. The corrected results support a bounded claim: loss-guided IO gains about three CT Dice points over network output without a detected increase in scan-level MTV/TLG change; IO without those losses increases both changes. This is an operating-point finding, conditional on predicted lesion masks and per-pair runtime. The remaining change is informative: predicted masks miss lesions, whole-scan losses permit opposing changes in individual lesions to cancel, and the soft direct MTV/TLG terms do not track the nearest-neighbor evaluation well. Per-bone rigidity is a separate anatomical result; it does not show a PET benefit. Do not claim equivalence of preservation or demonstrated clinical benefit.

**JMI fit.** The supplied papers use a structured Purpose / Approach / Results / Conclusions abstract and lead from an imaging problem to a gap, method, quantitative evidence, practical meaning, and limitations. This draft has those components, but many ablations, tuning details, and appendix-scale results obscure the main decision. Give the clinical measurement problem and recommended operating point priority; use the ablations and rigidity analysis to explain its limits. Check live special-issue rules separately at submission (#29).

## Story and claims — highest priority

1. [x] **#01 — Reopen the thesis and through-line after the scorer correction.** The earlier story was implemented, but the abstract's “without increasing” is stronger than the paired intervals establish. “Lesion masks and per-lesion evaluation limit further preservation” conflates a limitation with a way to reveal it. State one bounded thesis across Abstract, Introduction, Results, Discussion, and Conclusion: loss-guided IO improves alignment while holding *measured scan-level* PET change near network output; individual-lesion change remains. Decide whether the negative per-lesion and soft-loss results merit abstract space (`main.tex`).

2. [ ] **#02 — Define rigidity as a distinct anatomical result.** Full validation and test results are now present: per-bone rigid fit lowers short-distance distortion 13.4% to 5.0% and long-distance distortion 4.4% to 1.8%, at a bone-Dice cost and without detected PET benefit. Replace the old “performance pending” framing. Explain why rigidity belongs in the paper and avoid inferring correct bone placement solely from lower within-bone distortion (`results.tex`, `discussion.tex`).

3. [ ] **#03 — Make the Results sequence answer a few connected questions.** Preserve Q1 (do losses change the trade-off?) → Q2 (training or IO?), then connect the selected method to baselines and its limits. Per-lesion analysis currently precedes the mask-coverage and supplied-mask diagnostic that explain it; leave-one-out, ranking, rigidity, and weight sweeps add competing headlines. Consider mask coverage → per-lesion ceiling → failed per-lesion intervention, with leave-one-out explaining the surrogate mismatch. The Results opening says “all results are on the test set,” although rigidity sweeps and per-lesion tuning include validation results (`results.tex`).

4. [ ] **#04 — Make the clinical use case concrete without claiming benefit.** Explain why registration-induced change in the same follow-up scan could matter for lesion tracking or response assessment. If no defensible consequential-error threshold exists, say so. This study measures registration-induced change and label-based alignment, not downstream decisions or outcomes (`introduction.tex`, `discussion.tex`).

5. [x] **#05 — Distinguish the extension from the Learn2Reg paper.** The draft explains the additions and qualifies that its held-out split belonged to earlier challenge development. Keep that qualification if the experiment inventory changes (`introduction.tex`, `methods.tex`, `discussion.tex`).

6. [ ] **#06 — Recheck training-versus-IO and IO-versus-network claims.** Q2 reports IO-only minus training-and-IO differences of +1.0 Dice [0.4, 1.6], +0.8 MTV points [-0.4, 2.0], and +1.0 TLG points [-0.3, 2.6]. IO-versus-network PET intervals also allow small benefits and harms. Describe unresolved PET differences, not equivalent preservation. Make every abstract/conclusion sentence identify its comparison and separate significant Dice gains from PET results (`main.tex`, `results.tex`, `discussion.tex`).

7. [ ] **#07 — Separate two preservation limits.** With the corrected scorer, missed lesions account for much of the residual *scan-level* change (about 5.2/9.3 MTV and 6.2/11.8 TLG points in the decomposition). Supplied masks lower scan-level MTV change to 0.1%, yet median per-lesion change remains 13.5%; tested per-lesion terms change the test median only 13.7% to 13.3% and help 5/26 patients. The old checked item used obsolete numbers. Distinguish missed-lesion coverage, cancellation between lesions, and surrogate mismatch; better masks alone do not solve individual-lesion distortion (`results.tex`, `discussion.tex`).

## Design, methods, and interpretation

8. [ ] **#08 — Resolve PSMA biomarker terminology with nuclear-medicine coauthors.** MTV and especially “total lesion glycolysis” are challenge names, while this study measures PSMA uptake rather than FDG glucose metabolism. Explain that at first use and consider a clinically accurate parallel term while keeping challenge labels for reproducibility (`main.tex`, `methods.tex`).

9. [x] **#09 — Define preservation in one place.** Methods gives the challenge context, formulas, and same-moving-scan before/after error; Experiments refers back rather than redefining MTV/TLG. Preserve that division (`methods.tex`).

10. [ ] **#10 — Close the corrected-scorer and zero-field account.** Confirm every figure/table uses the corrected corner-aligned, nearest-neighbor scorer. Distinguish direct full-resolution evaluation here from the challenge's half-resolution submitted fields. Explain why resampling a zero field can give nonzero PET change while unresampled “Initial” is zero. Recheck the old 2.9% TLG control before calling it a general resampling floor (`methods.tex`, `tables/baselines.tex`).

11. [x] **#11 — Explain the predicted-label dependency.** The draft distinguishes predicted masks used by IO from supplied masks used for evaluation. Preserve this near the main result; lesion Dice is not the same as detection or coverage (`methods.tex`, `results.tex`).

12. [ ] **#12 — Re-audit statistics after the rerun.** Reconcile adjusted p-values, Holm families, confidence intervals, sample sizes (28 CT patients; 26 PET patients), and significance marks across text, captions, tables, and appendices. Say when a result is descriptive or based on validation tuning. The apparent PET reduction versus unconstrained IO is mostly not significant after correction; “no significant difference” does not establish equivalence (`methods.tex`, `results.tex`).

13. [ ] **#13 — Correct folding summaries everywhere.** The old checked claim that all our variants stayed below the 0.005% NDV threshold is false. The selected method exceeds it on two test pairs (76 and 125 ppm); the train-plus-IO variant on one (262 ppm), and NiftyReg on eight of 38. Distinguish averages from pair-level threshold exceedances in paper and `findings.md` (`results.tex`, `tables/baselines.tex`).

14. [ ] **#14 — Revisit baseline fairness and method selection.** Our method and IO settings use a 2-Dice-point/PET-error rule; NiftyReg and ConvexAdam currently use the challenge ranking. The old checked item incorrectly said all used the same rank. Baseline re-tuning is marked pending in the abstract. Once complete, document grids, selection rules, label access, affine preprocessing, and test evaluation consistently. Explain why rigidity weight was fixed when the stated selection rule would set it to zero (`methods.tex`, `main.tex`).

15. [x] **#15 — State transform and loss conventions.** The draft gives the fixed-to-moving sampling direction, Jacobian meaning, soft loss masks, and hard nearest-neighbor evaluation. Current IO code also uses nearest-neighbor PET and masks to choose its best iterate. Retain this precision (`methods.tex`, `Code/instance_opt.py`).

16. [x] **#16 — Fix the bone-distance metric description.** `methods.tex` says point pairs come from an “eroded mask,” but `Code/bone_metrics.py` passes the whole bone mask to `distance_distortion`; erosion is measured separately. Correct this concrete mismatch and check rigidity captions/appendices. Keep bone-associated lesions distinct from bone-rigidity metrics.

17. [ ] **#17 — State the practical cost of each option.** Network output takes about 11 s per pair and IO adds about 2.5 min on the stated GPU. In one concise paragraph or figure, connect runtime, Dice, scan-level PET change, and need for predicted lesion masks. The right operating point depends on per-pair time budget (`results.tex`, `discussion.tex`).

## Language, figures, and presentation

18. [ ] **#18 — Rebuild the abstract from final comparisons.** Name the configuration and denominator for each number; keep comparison sets coherent and qualify PET uncertainty. Baseline numbers are flagged for re-tuning, “similar MTV and TLG change” lacks scale, and “without increasing” is too strong. Compress secondary per-lesion/rigidity details if needed after fixing the main claim (`main.tex`).

19. [ ] **#19 — Reassess the qualitative example.** The old caption/run mismatch is fixed. The pair shows strong CT alignment, but PET change is larger for the selected method (+2.6% MTV, +6.6% TLG) than unconstrained IO (-0.6%, +2.3%). The text acknowledges this. Keep it explicitly as an alignment example or show a same-patient method contrast; do not imply that this case demonstrates PET benefit (`results.tex`, `figures/qualitative.tex`).

20. [ ] **#20 — Make captions self-contained.** Give cohorts/lesion denominators, axes, units, method definitions, uncertainty, and abbreviations. Distinguish group-relative change from contribution to whole-scan error in detection graphics, and predicted-mask losses from supplied-mask evaluation in trade-off graphics. Check final-PDF legibility and non-color cues (`figures/`, `tables/`).

21. [x] **#21 — Keep lesion denominators explicit.** Twenty-six patients have 1435 moving-scan components; size/location analysis excludes 108 below five voxels, leaving 1327. Scan-level and coverage analyses retain all 1435. Recheck only if final inputs change (`methods.tex`, `results.tex`).

22. [ ] **#22 — Reduce table density.** The baseline table has 11 columns and a long caption; ablation and rigidity tables carry secondary endpoints. Consider a compact decision-facing main table and full ablations, HD95, NDV, and rankings in appendices. Keep uncertainty and sample sizes visible (`tables/`).

23. [ ] **#23 — Shorten technical detail in the main narrative.** The loss weights, tuning grids, training schedule, ranking formula, and rigidity sweeps matter for reproduction but interrupt the clinical argument. Move secondary detail to appendices if allowed; keep enough for the primary comparison (`methods.tex`).

24. [ ] **#24 — Tighten language and calibrate inference.** Qualify “preserves,” “without increasing,” “best,” and “more plausible” by endpoint. Lower within-bone distortion establishes shape preservation, not correct bone placement; the sentence claiming it indicates more plausible bone motion overreaches. “A closer surrogate ... would likely preserve more” is a hypothesis, not a tested result. Standardize follow-up/moving, supplied/ground-truth, and bone-associated wording (`discussion.tex`).

25. [ ] **#25 — Make Discussion interpret rather than replay Results.** Open with the bounded IO decision, explain the three preservation limits and clinical relevance, then treat rigidity as a separate anatomical finding. Finish with one qualified recommendation and need for downstream validation. The current conclusion has five result sentences plus a denominator sentence that belongs in Methods (`discussion.tex`).

## Completion and JMI-facing checks

26. [ ] **#26 — Finish outstanding evidence and freeze the operating point.** Rigidity and per-lesion tests are now present. Finish baseline re-tuning, decide whether the chosen IO weight remains defensible after the tiny-weight sweep, and verify affected figures, tables, appendices, and qualitative panels. Include the early-stopping check only with its motivating training curves. Label post-hoc test observations as such; use validation evidence for any new recommendation.

27. [ ] **#27 — Finish disclosures and availability.** Replace code-link and funding TODOs; specify release scope and data access, and verify AI-use/disclosure wording against active policy (`main.tex`).

28. [ ] **#28 — Remove drafting artefacts and inspect layout.** Resolve every `\todo`, inspect resized tables and crowded figures, and decide whether lists of figures/tables belong in submission. Current PDF is 57 pages including appendices, references, and those lists. Use current submission instructions for spacing and numbering (`main.tex`).

29. [ ] **#29 — Check live JMI special-issue instructions before submission.** Verify abstract/section/appendix limits, supplementary material, disclosures, availability statements, and final layout. Published examples alone cannot establish these rules.

30. [ ] **#30 — Run a whole-paper consistency pass after results are frozen.** Check abstract, main text, figures, tables, captions, references, and *all appendices* for matching numbers, denominators, intervals, significance, method names, weights, and scorer/transform conventions. Compile and inspect the PDF.

31. [ ] **#31 — Cut repetition and long sentences.** Give each paragraph one job, remove Results restatements in Discussion, and simplify multi-clause Methods and captions. Use one LaTeX sentence per source line when editing, without reformatting unrelated text (`methods.tex`, `results.tex`, `discussion.tex`).

32. [ ] **#32 — Explain the PET-weight sweep and selection tension.** At ×10⁻⁴, the sweep reports roughly the same supplied-mask MTV change as validation-chosen ×0.1, with 1.6 more Dice points; TLG is 1.1 points worse. The tuning grid skipped weights between zero and ×0.1. Decide whether an expanded *validation* sweep is needed before recommending ×0.1, or present the tiny-weight result as post-hoc sensitivity analysis. Explain the 2-Dice-point tolerance and fixed rigidity weight as value choices (`methods.tex`, `results.tex`, `discussion.tex`).

33. [ ] **#33 — Clarify the direct PET-term negative result.** In leave-one-out IO, removing the soft MTV or TLG term improves the official hard TLG endpoint; mean-Jacobian loss carries most measured preservation. This challenges a blanket claim that every preservation term helps. State significant comparisons, distinguish evidence from the plausible soft-versus-nearest explanation, and say whether four terms remain for a prespecified analysis or should be retuned on validation (`results.tex`, `discussion.tex`, `tables/loo.tex`).

34. [ ] **#34 — Label the per-lesion intervention as a negative test.** Terms on the eight largest predicted components improved validation median volume change but had little test benefit (13.7% to 13.3%; 5/26 patients better) and worsened small lesions. State eligibility and weight selection; distinguish the diagnostic supplied-mask experiment from deployable predicted-mask IO (`methods.tex`, `results.tex`, `discussion.tex`).

35. [ ] **#35 — Reconcile `findings.md` with the corrected paper.** Its leading story uses corrected values, but later checked sections retain pre-fix results, zero-field interpretations, and an old rigidity sweep. Mark obsolete analyses clearly or replace them so checked boxes cannot be mistaken for current evidence. Cross-check manuscript claims against corrected outputs rather than old checked notes.
