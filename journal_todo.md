# Journal plan: remaining work (temporary)

One step at a time, any order: implement → user checks → tick. Done already:
interim steps 1–10, experiment 4 (B1–B6 on test, table, distribution plot,
qualitative figure).

## Experiment 1: PET and rigidity ablations
- [x] 1. **Networks on test:** T1, T2, T3 (T4 done).
- [x] 2. **T1-tum:** evaluate T1's `best_tumour` checkpoint on test (downloaded).
- [x] 3. **IO runs on test:** I1–I3, I5–I7 (I4 done).
- [x] 3b. **Affine + IO (I8) and choice of "Ours":** I1–I8 on val, ranked → I1; baseline table shows both Ours (I1, val-chosen) and Ours (full) (I4, pre-declared); I8 on test (keep for paper).
- [ ] 4. **Leave-one-out on test:** L1–L4 (reference I5).
- [x] 5. **Q1 table + Q2 figure:** `table_ablations.py` (T1–T4 | I1–I4, marks vs. neither), `figure_q2.py` (trained + IO vs. IO only, Dice vs. |MTV| / |TLG|; effects for the text in `q2_effects.txt`). I8 moved to the baseline table; T1-tum in findings (text only).
- [x] 6. **Per-lesion results:** |MTV|/|TLG| by size bin and bone association, signed + upper tail (table or figure).
- [ ] 7. **Leave-one-out table:** `table_loo.py` (I5 vs. L1–L4), run once step 4 is done.

## Experiment 2: rigidity comparison (IO on T2)
- [x] 8. **Weight selection (decided: curve + Dice tolerance):** calibrate a reference weight per method (initial rigidity loss = Kabsch at 0.25, few val pairs); val grid ×{0.1, 0.3, 1, 3, 10}; choose the largest weight within 1 Dice point of R1; val curve Dice vs. bone distortion.
- [ ] 9. **Weight sweeps on val:** R2 Staring, R3 Staring + erosion, R4 Kabsch.
- [ ] 10. **Test runs:** R1 (= I2), R2–R4 at the chosen weights.
- [ ] 11. **Rigidity table + figure:** bone/soft Dice and HD95, distance distortion, NDV in bone / band / body, leakage, erosion retention; alignment vs. distortion plot.

## Experiment 3: accuracy–preservation trade-off (IO on T1)
- [x] 12. **W grid (decided):** T1 + PET IO, × {0, 0.01, 0.03, 0.1, 0.3, 1} of the config PET weights; × 0 = I1, × 0.1 = I5 (tuned operating point, no val runs); new: W0.01, W0.03, W0.3, W1 (evaluate.py).
- [ ] 13. **W runs on test:** queued in tmux `tradeoff_test` (after step 4, before step 9).
- [ ] 14. **Trade-off figure:** Dice vs. |MTV|/|TLG|, NDV alongside, operating point marked.

## Paper
- [x] 15. **Methods:** evaluation protocol (patient-level aggregation, significance ranking, bootstrap), baselines + tuning grids, IO settings (rigidity fixed at 0.25), checkpoint selection, predicted-label provenance, patient exposure disclosure.
- [ ] 16. **Results:** experiments 1–4 text, replacing the placeholder tables/figures.
- [ ] 17. **Discussion / limitations:** TLG resampling floor, predicted-label limit of PET IO, NiftyReg folding, ranking dominated by accuracy, Staring leakage.
- [ ] 18. **Supplement:** final-checkpoint results, component scores, signed MTV/TLG, IO tuning tables.

## Optional
- [ ] 19. **Gradient analysis:** loss-pair cosine matrix, IO trajectories.
- [ ] 20. **External cohort** (only if it doesn't delay submission).
