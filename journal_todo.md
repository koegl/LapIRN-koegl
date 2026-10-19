# Journal plan: remaining work (temporary)

One step at a time, any order: implement → user checks → tick. Done already:
interim steps 1–10, experiment 4 (B1–B6 on test, table, distribution plot,
qualitative figure).

## Experiment 1: PET and rigidity ablations
- [x] 1. **Networks on test:** T1, T2, T3 (T4 done).
- [x] 2. **T1-tum:** evaluate T1's `best_tumour` checkpoint on test (downloaded).
- [ ] 3. **IO runs on test:** I1–I3, I5–I7 (I4 done).
- [ ] 4. **Leave-one-out on test:** L1–L4 (reference I5).
- [ ] 5. **Ablation table:** network (T1–T4, T1-tum) and IO (I1–I7). No ranking: planned contrasts (Q1: I2–I4 vs. I1; Q2: I5–I7 vs. I2–I4), paired difference of patient means with 95 % bootstrap CI, Wilcoxon + Holm within the table.
- [ ] 6. **Per-lesion results:** |MTV|/|TLG| by size bin and bone association, signed + upper tail (table or figure).
- [ ] 7. **Leave-one-out table/figure:** contribution of each PET term in IO.

## Experiment 2: rigidity comparison (IO on T2)
- [ ] 8. **Decide the weight selection:** the challenge ranking never rewards rigidity (tuned weight → 0), so pick per method e.g. the largest weight within a Dice tolerance, or report the whole curve.
- [ ] 9. **Weight sweeps on val:** R2 Staring, R3 Staring + erosion, R4 Kabsch.
- [ ] 10. **Test runs:** R1 (= I2), R2–R4 at the chosen weights.
- [ ] 11. **Rigidity table + figure:** bone/soft Dice and HD95, distance distortion, NDV in bone / band / body, leakage, erosion retention; alignment vs. distortion plot.

## Experiment 3: accuracy–preservation trade-off (IO on T1)
- [ ] 12. **Revise the W grid** around the tuned PET weight (×0.1–×4 is flat), e.g. ×{0, 0.01, 0.03, 0.1, 0.3, 1} of the current value.
- [ ] 13. **W runs:** val (operating point) and test (curve).
- [ ] 14. **Trade-off figure:** Dice vs. |MTV|/|TLG|, NDV alongside, operating point marked.

## Paper
- [ ] 15. **Methods:** evaluation protocol (patient-level aggregation, significance ranking, bootstrap), baselines + tuning grids, IO settings (rigidity fixed at 0.25), checkpoint selection, predicted-label provenance, patient exposure disclosure.
- [ ] 16. **Results:** experiments 1–4 text, replacing the placeholder tables/figures.
- [ ] 17. **Discussion / limitations:** TLG resampling floor, predicted-label limit of PET IO, NiftyReg folding, ranking dominated by accuracy, Staring leakage.
- [ ] 18. **Supplement:** final-checkpoint results, component scores, signed MTV/TLG, IO tuning tables.

## Optional
- [ ] 19. **Gradient analysis:** loss-pair cosine matrix, IO trajectories.
- [ ] 20. **External cohort** (only if it doesn't delay submission).
