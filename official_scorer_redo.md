# Redo with the official scorer

New results: `evaluation_official_scorer/` (old ones stay in `evaluation/`).
Paper outputs: back up `journal_results/paper/out/` to `out_old_scorer/`, switch the
scripts' input folder, regenerate, copy to `overleaf_jmi/`.

## Runs
- [x] Stage 1: IO lr x iterations -> lr 0.025 / it 40
- [x] Stage 2: IO PET scale -> x0.1
- [ ] Stage 3: test runs, I1-I8 val, rigidity sweep (3 Kabsch runs left)
- [ ] Stage 4: R2-R4 on test
- [x] Choice of "Ours" (I1-I8 val, new rule) -> I7
- [ ] (Optional) re-tune NiftyReg / ConvexAdam with the new rule

## Methods
- [ ] Evaluation metrics: official scorer conventions (mask and PET warped nearest, align_corners=True, official NDV mask, exact field conversion)
- [ ] Remove the TLG resampling floor (methods, results, discussion, findings)
- [ ] Tuning and model selection
  - [ ] new selection rule (lowest |MTV| + |TLG| within 2 Dice points of the most accurate)
  - [ ] new IO settings (lr 0.025, 40 iterations, PET x0.1)
  - [ ] choice of "Ours" -> I7 (paragraph "Choice of the full method")
- [ ] Rigidity weight selection: 2-point tolerance, grid x0.1-x100 for all penalties
- [ ] Weights table (`tab:weights`): IO row

## Results
- [ ] Q1: effect of the losses
  - [ ] table (`table_ablations.py`)
  - [ ] text
- [ ] Q2: losses in training or only in IO
  - [ ] figure (`figure_q2.py`)
  - [ ] text
- [ ] Per-lesion volume change
  - [ ] figure (`figure_lesions.py`)
  - [ ] text
- [ ] Leave-one-out
  - [ ] table (`table_loo.py`)
  - [ ] text
- [ ] Comparison with baselines
  - [ ] table (`table_baselines.py`; "Ours" = I7)
  - [ ] distribution figure (`plot_baselines_distribution.py`)
  - [ ] text
- [ ] Qualitative example
  - [ ] figure (`figure_qualitative.py`, new I4 field from lustre)
  - [ ] text
- [ ] Accuracy-preservation trade-off
  - [ ] figure (`figure_tradeoff.py`)
  - [ ] text
- [ ] Missed and detected lesions
  - [ ] figure (`figure_detection.py`)
  - [ ] text
- [ ] Rigidity penalties (after stages 3 and 4)
  - [ ] 4-panel sweep figure (`figure_rigidity.py`)
  - [ ] cross-evaluation figure (`figure_rigidity_measures.py`)
  - [ ] preservation vs. weight figure (`figure_rigidity_preservation.py`)
  - [ ] table (`table_rigidity.py`) + pairwise tests
  - [ ] text

## Appendix
- [ ] A: IO tuning table (`table_io_tuning.py`, new stages and rule)
- [ ] B: signed MTV / TLG (`table_signed.py`)
- [ ] C: detection tables (`table_detection_groups.py`, `table_detection_contributions.py`)
- [ ] D: full rigidity table

## Rest of the manuscript
- [ ] Abstract (numbers, rigidity sentence)
- [ ] Introduction: contribution list
- [ ] Discussion
  - [ ] numbers
  - [ ] weighting paragraph (tuning grid statement)
  - [ ] evaluation paragraph (TLG floor)
  - [ ] rigidity subsection
- [ ] Conclusion
- [ ] Cover letter bullets
- [ ] findings.md
