# Redo with the official scorer

New results: `evaluation_official_scorer/` (old ones stay in `evaluation/`).
Paper outputs: back up `journal_results/paper/out/` to `out_old_scorer/`, switch the
scripts' input folder, regenerate, copy to `overleaf_jmi/`.

## Runs
- [x] Stage 1: IO lr x iterations -> lr 0.025 / it 40
- [x] Stage 2: IO PET scale -> x0.1
- [x] Stage 3: test runs, I1-I8 val, rigidity sweep -> R2 Staring 1.98, R3 eroded 103.2, R4 Kabsch 25
- [ ] Stage 4: R2-R4 on test
- [x] Stage 5: P1 / P2 per-lesion terms (fixed weights 8/8/15) -> no effect (terms < 1 % of the loss)
- [ ] Stage 6: val sweep of the per-lesion weight (P1 x1 / x10 / x100) -> `select_stage.py perlesion` -> chosen scale on test
- [x] Choice of "Ours" (I1-I8 val, new rule) -> I7
- [ ] (Optional) re-tune NiftyReg / ConvexAdam with the new rule

## Methods
- [x] Evaluation metrics: official scorer conventions (mask and PET warped nearest, align_corners=True, official NDV mask, exact field conversion)
- [x] Remove the TLG resampling floor (methods, results, discussion, findings)
- [x] Tuning and model selection
  - [x] new selection rule (lowest |MTV| + |TLG| within 2 Dice points of the most accurate)
  - [x] new IO settings (lr 0.025, 40 iterations, PET x0.1)
  - [x] choice of "Ours" -> I7 (paragraph "Choice of the full method")
- [x] Rigidity weight selection: 2-point tolerance, grid x0.1-x100 for all penalties
- [x] Weights table (`tab:weights`): IO row

## Results
- [x] Q1: effect of the losses
  - [x] table (`table_ablations.py`)
  - [x] text
- [x] Q2: losses in training or only in IO
  - [x] figure (`figure_q2.py`, now all eleven variants: Q1 + Q2)
  - [x] text
- [ ] Per-lesion volume change
  - [x] figure (`figure_lesions.py`)
  - [x] TLG figure (appendix B, same script)
  - [x] text
  - [ ] outlook: P1 / P2 numbers
- [x] Leave-one-out
  - [x] table (`table_loo.py`)
  - [x] text
- [x] Comparison with baselines
  - [x] table (`table_baselines.py`; "Ours" = I7)
  - [x] distribution figure (`plot_baselines_distribution.py`)
  - [x] text
- [x] Qualitative example
  - [x] figure (`figure_qualitative.py`, new I4 field from lustre)
  - [x] text
- [x] Accuracy-preservation trade-off
  - [x] figure (`figure_tradeoff.py`)
  - [x] text
- [x] Missed and detected lesions
  - [x] figure (`figure_detection.py`)
  - [x] text
- [ ] Rigidity penalties (after stages 3 and 4)
  - [ ] 4-panel sweep figure (`figure_rigidity.py`)
  - [ ] cross-evaluation figure (`figure_rigidity_measures.py`)
  - [ ] preservation vs. weight figure (`figure_rigidity_preservation.py`)
  - [ ] table (`table_rigidity.py`) + pairwise tests
  - [ ] text

## Appendix
- [x] A: Q1 table (`table_ablations.py`, moved from the results; figure_q2 now shows Q1 + Q2)
- [x] B: per-lesion TLG figure
- [x] C: IO tuning table (`table_io_tuning.py`, new stages and rule)
- [x] D: signed MTV / TLG (`table_signed.py`)
- [x] E: detection tables (`table_detection_groups.py`, `table_detection_contributions.py`)
- [ ] F: full rigidity table

## Rest of the manuscript
- [ ] Abstract (numbers, rigidity sentence)
- [ ] Introduction: contribution list
- [ ] Discussion
  - [x] numbers
  - [x] weighting paragraph (tuning grid statement)
  - [x] evaluation paragraph (TLG floor)
  - [ ] rigidity subsection
- [ ] Conclusion
- [ ] Cover letter bullets
- [ ] findings.md
