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
- [x] 9. **Weight sweeps on val (first run):** 19 runs (janus + cluster); grid extended to ×30, ×100 for Staring + erosion and Kabsch, capped at ×100. Chosen: Staring ×1, Staring + erosion ×100, Kabsch ×100 (`evaluate.RIGIDITY_WEIGHTS`). Old results: `rigidity_sweep_interior_metric/`.
- [ ] 9.1. **Rerun sweep with whole-bone metrics** (cluster array, val I2 + 19 runs; sync loop in tmux `sync_rigidity` writes `curve.csv` / `chosen.json`): distance distortion on whole bones instead of eroded interiors; new `staring_energy_bone` (+ its three terms). Check that the chosen weights still hold (Kabsch ×100 is close to the threshold).
- [ ] 10. **Test runs:** R1 (= I2, needed for the new metrics), R2–R4 at the chosen weights (cluster array, queued; synced by the same loop).
- [ ] 11. **Rigidity figures + table:** drafted on old val data:
  - `figure_rigidity.py` (2 × 2: CT / bone Dice vs. short / long distortion, val curve; label offsets `OFFSETS_A`–`D`),
  - `figure_rigidity_measures.py` (cross-evaluation on all rigidity measures, relative to no rigidity),
  - `table_rigidity.py` (+ `rigidity_pairwise.txt` for the text; PDF preview),
  - `figure_rigidity_preservation.py` (|MTV| / |TLG| vs. weight).
  When 9.1 / 10 are synced: set `SWEEP_DIR` to `rigidity_sweep` and `rigidity.SOURCE = "test"`, rerun, decide table vs. plot.
- [ ] 11.1. Describe the rigidity results in the paper (results `\todo`, discussion `\todo`, abstract `\todo`, contribution 4).

## Experiment 3: accuracy–preservation trade-off (IO on T1)
- [x] 12. **W grid (decided):** T1 + PET IO, × {0, 0.01, 0.03, 0.1, 0.3, 1} of the config PET weights; × 0 = I1, × 0.1 = I5 (tuned operating point, no val runs); new: W0.01, W0.03, W0.3, W1 (evaluate.py).
- [x] 13. **W runs on test:** W0.01, W0.03, W0.3, W1.
- [x] 13.1. **Cluster runs on test:** W0, W0.0001, W0.0003, W0.001, W0.003, G1, G2 (IO with GT lesion masks); synced.
- [x] 14.1. **Trade-off figure:** `figure_tradeoff.py` (10 scales, predicted vs. GT labels, G1/G2 as diamonds).
- [x] 14.1.1. Trade-off results written (tiny weights suffice; G1/G2: masks are the limit).
- [x] 14.2. **Detected vs. missed lesions:** `figure_detection.py` (in `results.tex`); tables `table_detection_groups.py` / `table_detection_contributions.py` (supplement candidates).
- [x] 14.2.1. Results text incl. G1/G2 breakdown.

## Paper
- [x] 15. **Methods:** evaluation protocol (patient-level aggregation, significance ranking, bootstrap), baselines + tuning grids, IO settings (rigidity fixed at 0.25), checkpoint selection, predicted-label provenance, patient exposure disclosure.
- [ ] 16. **Results:** done: Q1, Q2, per-lesion, leave-one-out, baselines, qualitative (0146, full method), trade-off, detected vs. missed. Open: rigidity (11.1).
- [ ] 17. **Discussion / conclusion:** written (user restructured into subsections); open: rigidity subsection.
- [ ] 18. **Appendix** (in `main.tex` after the conclusion, as in JMI papers; `appendix.tex`): A IO tuning (`table_io_tuning.py`), B signed MTV/TLG (`table_signed.py`), C detection tables; open: D full rigidity table once 10 is in.
- [ ] 21. **Abstract and contribution list:** written (abstract 235 / 250 words); open: rigidity sentence in approach / results, check contribution 4.
- [ ] 22. **Pipeline figure:** decide whether `figures/overview.png` needs the rigidity term / IO-only variant (and whether it has a white border).
- [ ] 23. **Front matter:** funding of E.C. (acknowledgments `\todo`); agree the author order (L. Förner 5th) with all authors.
- [ ] 24. **Code and weights release:** clean repo `psmareg-jmi` (training, IO, evaluation, split, trained weights); add the link to the code/data availability section and the submission portal.
- [ ] 25. **Cover letter:** fill in the rigidity bullet, check all findings bullets against the final results (add: tiny PET weights suffice, GT masks -> 2 % MTV), shorten to one page.
- [ ] 26. **Before submission:** remove `\todo` macro and line numbers if required, check figures/tables are copied from `journal_results/paper/out/`, compile, author approval. Deadline: Oct 30, 2026.

## Optional
- [ ] 19. **Gradient analysis:** loss-pair cosine matrix, IO trajectories.
- [ ] 20. **External cohort** (only if it doesn't delay submission).
