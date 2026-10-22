# Journal plan: remaining work (temporary)

One step at a time, any order: implement → user checks → tick. Done already:
interim steps 1–10, experiment 4 (B1–B6 on test, table, distribution plot,
qualitative figure).

## Experiment 1: PET and rigidity ablations
- [x] 1. **Networks on test:** T1, T2, T3 (T4 done).
- [x] 2. **T1-tum:** evaluate T1's `best_tumour` checkpoint on test (downloaded).
- [x] 3. **IO runs on test:** I1–I3, I5–I7 (I4 done).
- [x] 3b. **Affine + IO (I8) and choice of "Ours":** I1–I8 on val, ranked → I1; baseline table shows both Ours (I1, val-chosen) and Ours (full) (I4, pre-declared); I8 on test (keep for paper).
- [x] 4. **Leave-one-out on test:** L1–L4 (reference I5).
- [x] 5. **Q1 table + Q2 figure:** `table_ablations.py` (T1–T4 | I1–I4, marks vs. neither), `figure_q2.py` (trained + IO vs. IO only, Dice vs. |MTV| / |TLG|; effects for the text in `q2_effects.txt`). I8 moved to the baseline table; T1-tum in findings (text only).
- [x] 6. **Per-lesion results:** |MTV|/|TLG| by size bin and bone association, signed + upper tail (table or figure).
- [x] 7. **Leave-one-out table:** `table_loo.py` (I5 vs. L1–L4), in `results.tex`.
- [ ] 7.1. Describe the leave-one-out results in the paper (L_MTV costs Dice and adds little; L_MTV-avg and L_jac carry the preservation; L_TLG no significant effect).

## Experiment 2: rigidity comparison (IO on T2)
- [x] 8. **Weight selection (decided: curve + Dice tolerance):** calibrate a reference weight per method (initial rigidity loss = Kabsch at 0.25, few val pairs); val grid ×{0.1, 0.3, 1, 3, 10}; choose the largest weight within 1 Dice point of R1; val curve Dice vs. bone distortion.
- [ ] 9. **Weight sweeps on val:** R2 Staring, R3 Staring + erosion, R4 Kabsch. Running on janus (tmux `rigidity_sweep`, 15 runs, ~2.2 h each); stays on janus.
- [ ] 10. **Test runs:** R1 (= I2), R2–R4 at the chosen weights (cluster: `submit_eval_array.sh`).
- [ ] 11. **Rigidity table + figure:** bone/soft Dice and HD95, distance distortion, NDV in bone / band / body, leakage, erosion retention; alignment vs. distortion plot.
- [ ] 11.1. Describe the rigidity results in the paper.

## Experiment 3: accuracy–preservation trade-off (IO on T1)
- [x] 12. **W grid (decided):** T1 + PET IO, × {0, 0.01, 0.03, 0.1, 0.3, 1} of the config PET weights; × 0 = I1, × 0.1 = I5 (tuned operating point, no val runs); new: W0.01, W0.03, W0.3, W1 (evaluate.py).
- [x] 13. **W runs on test:** W0.01, W0.03, W0.3, W1.
- [ ] 13.1. **Cluster runs on test** (array 41881677, general pool): W0 (×0 with PET terms logged), W0.0001, W0.0003, W0.001, W0.003, and IO with GT lesion masks G1 (×0.1), G2 (×1). Then sync back: `rsync -av fryderyk.koegl@hpc-submit02.scidom.de:/lustre/groups/iml/data/PET_CT_bone/test_set_for_eval/evaluation/test/ evaluation/test/`.
- [ ] 14.1. **Trade-off figure:** `figure_tradeoff.py` done (Dice, |MTV|, |TLG| on predicted vs. GT labels, network output dashed); rerun once 13.1 is synced.
- [ ] 14.1.1. Describe the trade-off results (incl. G1/G2: does IO with GT masks reach ~1 %?).
- [x] 14.2. **Detected vs. missed lesions:** `figure_detection.py` (in `results.tex`); tables `table_detection_groups.py` / `table_detection_contributions.py` (keep or drop).
- [ ] 14.2.1. Results text written; add the G1/G2 breakdown (does the covered-lesion contribution of 2.6 points disappear with GT masks?) and resolve the `\todo`.

## Paper
- [x] 15. **Methods:** evaluation protocol (patient-level aggregation, significance ranking, bootstrap), baselines + tuning grids, IO settings (rigidity fixed at 0.25), checkpoint selection, predicted-label provenance, patient exposure disclosure.
- [ ] 16. **Results:** experiments 1–4 text. Done: Q1, Q2, per-lesion, baselines, detected vs. missed. Open: leave-one-out (7.1), rigidity (11.1), trade-off (14.1.1), description of the qualitative case (maybe choose another `CASE` in `figure_qualitative.py`; 0054 has extensive disease).
- [ ] 17. **Discussion / limitations:** TLG resampling floor, PET IO limited by missed lesions of the predicted masks (and mask inaccuracy, pending G1/G2), NiftyReg folding, ranking dominated by accuracy, Staring leakage; conclusion.
- [ ] 18. **Supplement:** final-checkpoint results, component scores, signed MTV/TLG, IO tuning tables (maybe the detection tables).
- [ ] 21. **Abstract and contribution list** (intro `\todo`), once all results are in.
- [ ] 22. **Pipeline figure:** decide whether `figures/overview.png` needs the rigidity term / IO-only variant.
- [ ] 23. **Front matter:** funding of E.C. (acknowledgments `\todo`); agree the author order (L. Förner 5th) with all authors.
- [ ] 24. **Code and weights release:** clean repo `psmareg-jmi` (training, IO, evaluation, split, trained weights); add the link to the code/data availability section and the submission portal.
- [ ] 25. **Cover letter:** fill in the rigidity bullet, check all findings bullets against the final results, shorten to one page.
- [ ] 26. **Before submission:** remove `\todo` macro and line numbers if required, check figures/tables are copied from `journal_results/paper/out/`, compile, author approval. Deadline: Oct 30, 2026.

## Optional
- [ ] 19. **Gradient analysis:** loss-pair cosine matrix, IO trajectories.
- [ ] 20. **External cohort** (only if it doesn't delay submission).
