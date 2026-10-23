# Findings

Running log of results that matter for the paper. Each entry states its data
basis; **preliminary** = not yet confirmed on the full dataset.

## [ ] Paper story (order of results and discussion)

**Status:** agreed outline; numbers from the test set (28 patients, 38 pairs).

1. **Problem:** follow-up→baseline PSMA PET/CT registration should align the
   anatomy without changing lesion volume / uptake (MTV / TLG); bones should
   stay rigid.
2. **Method:** LapIRN network + instance optimisation (IO) with PET-preservation
   and bone-rigidity losses. Full method = losses in training and IO (I4),
   defined in advance. The val selection (challenge ranking over I1–I8) chose
   I1 (no losses): the ranking is dominated by accuracy, biomarker differences
   are rarely significant (methods section).
3. **Q1 — do the losses work?** (`ablations.tex`, network | network + IO):
   PET and rigidity lower |ΔMTV| / |ΔTLG| significantly in both blocks, at a
   known accuracy cost (PET about −2 to −3 Dice points, rigidity ≤ 0.5).
   Early stopping was tested as a possible alternative (T1-tum; entry below),
   but is not part of the present manuscript story without the training curves.
4. **Q2 — where should the losses act?** (`q2.pdf`): losses only in IO (I7)
   vs. in training + IO (I4): Dice +1.7 points (significant). The paired
   preservation differences were not significant, but equivalence was not
   established (MTV +0.8 points, 95% CI −1.0 to 2.7; TLG +0.2,
   −1.9 to 2.4). IO-only placement is more accurate when per-pair refinement
   is feasible; without IO (14× faster: ~11 s vs. ~157 s per pair), training
   losses reduce the registration-induced PET changes.
5. **Leave-one-out** (`loo.tex`): what each PET term contributes in IO.
6. **Baselines** (`baselines.tex`, distribution plot, qualitative figure):
   training / IO columns state where the losses act. Every variant of ours is
   significantly more accurate than NiftyReg / ConvexAdam / Affine, without
   folding; ranks: IO only (I7) 1, basic IO (I1) and full (I4) 2. Affine + IO
   (no network) reaches only 52.7 % Dice: the network is essential.
7. **Rigidity comparison (exp. 2)** and **trade-off (exp. 3)**: follow.
8. **Discussion:**
   - losses in IO only vs. in training: use case (time budget) decides;
   - ranking weakness: methods that barely deform score high on biomarkers
     (Affine + IO rank 5 with 53 % Dice);
   - TLG resampling floor; predicted lesion labels limit PET IO;
   - NiftyReg folding on test; Staring stencil leakage (entries below).

## [ ] Staring stencils leak; erosion leaves most of the skeleton unconstrained

**Status:** confirmed — all 560 labelled challenge scans (358 patients: train,
val, test and single-session), ground-truth CT labels. Identical across splits.

- The Staring rigidity terms read the 18 neighbours within one voxel (6 face
  neighbours for det(J) / orthonormality, plus 12 edge neighbours for the mixed
  second derivatives of the affine term).
- On the full bone mask, **21.2% ± 1.2%** of these stencil reads fall outside
  the centre voxel's own bone (soft tissue or a neighbouring bone), so the term
  also constrains surrounding tissue (per scan, range 16–24%).
- Eroding each bone by exactly this footprint removes the leakage (0% on every
  scan), but keeps only **35.5% ± 2.2%** of bone voxels, and **23%** of bones
  lose their interior entirely.
- By bone group (voxels kept / bones that vanish):
  - ribs: **2.4%** / **58%**
  - vertebrae: 22% / 0.7%
  - other bones: 39% / 0%
- Implication: Staring + erosion leaves most of the skeleton, and the rib cage
  almost entirely, unconstrained; Kabsch needs no erosion and has no stencil
  leakage by construction.

Source: `journal_results/erosion_leakage.py` → `journal_results/erosion_leakage/`
(`per_scan.csv`, `per_bone.csv`, `summary.txt`).

## [x] The scorer's warp changes TLG even for the identity transform

**Status:** confirmed — all test pairs with lesions (30 of 38 pairs), our copy
of the scorer's transformer (`evaluate.SpatialTransformer`, from voxelmorph).
To check against the official scorer code.

- The transformer normalises voxel positions with (n − 1) but samples with
  `align_corners=False`, so voxel i is read at i·n/(n − 1) − 0.5: a zero field
  shifts the image by up to half a voxel (−0.5 at one end, +0.5 at the other).
- With an all-zero field, the linearly resampled PET changes TLG by
  **2.9% ± 1.2%** (range 1.0–6.9%), always downwards (signed −2.9%: the linear
  interpolation smooths the lesion peaks); MTV (nearest-neighbour mask) stays
  at 0%. Val pair 0071 01→00 gave 2.5%.
- Handling: the "initial" row (B1) is scored without any resampling (MTV / TLG
  change exactly 0); all other methods keep the scorer's warp. The floor is
  stated in the paper as a resampling effect common to all methods.

Source: `evaluate.py` runs B1, B1-floor → `evaluation/test/`.

## [x] NiftyReg folds more on test than on val

**Status:** confirmed — val-chosen config be 0.0002, sx −9 (38 val / 38 test pairs).

- %NDV mean 0.016 (val) vs. **0.093** (test); the median is 0 on both.
- On test, three pairs fold strongly (0.8–1.3%; cases 0338, 0166, 0170).
- Cause: the ranking picks the least-regularised config on val (the
  accuracy scores separate configs; NDV only barely). Report as is.

Source: `evaluation/test/B3.csv`, `journal_results/baselines/niftyreg/eval/val/`.

## [ ] IO PET terms work on their mask; predicted lesion labels limit them

**Status:** confirmed on val (38 pairs, 26 patients), T4 + IO (lr 0.05, 20 it),
scored against ground-truth lesions; p = patient-level one-sided Wilcoxon vs. PET off.

| IO lesion labels | MTV bias | TLG bias | CT Dice |
|---|---|---|---|
| PET off | 0.103 | 0.118 | 0.800 |
| nnU-Net (×1 = ×4) | 0.063 (p 0.35) | 0.050 (p 0.001) | 0.782 |
| ground truth (×1 = ×4) | 0.035 (p 0.016) | 0.047 (p 0.002) | 0.786 |
| recall-biased mask (×1 = ×4) | 0.156 | 0.166 | 0.787 |

- PET weight ×1 and ×4 give identical results: the terms are satisfied on the
  mask IO sees, so the remaining MTV error comes from label errors, not weights.
- The recall-biased mask (SUV > 3 minus dilated physiological organs, ∪ nnU-Net;
  challenge winner's approach) makes it worse: preserving false-positive
  regions shrinks the true lesions (signed MTV −6 %). Its lesion detection is
  0.89 vs. 0.78 for nnU-Net, but precision 0.18 vs. 0.62.
- IO tuning (round 2) settles on PET ×0.1; ×0.1 to ×4 are flat.

Source: `journal_results/io_pet_check.py` → `io_pet_check/summary.csv`;
`journal_results/recall_masks.py` → `recall_masks/summary.csv`.

## [ ] Early stopping on tumour preservation does not replace the PET losses

**Status:** confirmed — test set (28 patients, 38 pairs); currently omitted from
the manuscript. Reconsider only if the training curves are shown to explain why
early stopping was tested.

- T1-tum (T1, checkpoint selected on tumour preservation instead of accuracy)
  vs. T1: CT Dice −3.0 points (significant), |ΔMTV| −0.4 and |ΔTLG| −0.9
  points (both not significant).
- For comparison, training with the PET losses (T2 vs. T1): Dice −1.9,
  |ΔMTV| −3.0 points (both significant).
- So stopping early costs more accuracy and buys almost no preservation.

Source: `evaluation/test/T1.csv`, `T1-tum.csv`, `T2.csv`;
`journal_results/paper/table_ablations.py` (paired Wilcoxon, Holm).

## [x] Per-lesion volume change: draft results text

**Status:** draft for the results section — test set, 1,327 lesions (≥ 5 voxels)
of 26 patients; figure `journal_results/paper/out/figures/lesions.pdf`
(median |ΔMTV| per lesion, network | network + IO, soft tissue / bone ×
small / medium / large; patient-cluster bootstrap CIs).

- **Size: the smaller the lesion, the more its volume changes.** E.g. bone
  lesions with network + IO and no losses typically change by 33 % (small),
  17 % (medium) and 9 % (large). A small lesion has only a few voxels
  (< 0.5 ml is fewer than about 20 voxels), so gaining or losing one or two
  voxels is already a large percentage; a large lesion barely notices one
  voxel. Small lesions are therefore the hardest case — and where the losses
  help most (small bone lesions: 33 % → 10 %).
- **Location, without the losses:** IO increases lesion distortion, most for
  small bone lesions (median 20 % → 33 %). Bone lesions tend to grow (signed
  median +3 to +10 %), soft-tissue lesions to shrink (−9 to −13 %).
- **Rigidity acts only on bone lesions** (as its mechanism predicts): with IO
  it lowers the change of small / medium / large bone lesions (33 → 20 %,
  17 → 12 %, 9 → 5 %) but not of soft-tissue lesions (unchanged, 17–22 %).
- **PET acts in both locations:** e.g. medium soft-tissue lesions 18 → 10 %,
  small bone lesions 33 → 13 %. Exception: small soft-tissue lesions (n = 41,
  wide CIs), no clear effect.
- **Combined (PET + rigidity):** lowest change in bone at every size
  (10 / 9 / 4 %); the signed change of bone lesions drops to about 0, so the
  systematic growth disappears.
- **Network vs. network + IO:** same pattern. With the losses IO does not
  increase distortion; without them it does, most in bone.
- **Caveat (methods / caption):** medians of small lesions are quantised
  (e.g. exactly 10.0 %, 12.5 %, 20.0 %) because they are ratios of small voxel
  counts.

Source: `journal_results/paper/figure_lesions.py` → `lesions.pdf`,
`lesions_summary.txt`.

## [x] PET IO is limited by missed lesions (test, T1 + PET IO)
- On its own (predicted) masks PET IO removes almost all volume change (8.1 -> 1.0 %); on GT lesions only 10.6 -> 7.6 %. Higher PET weights (x0.1 -> x1) change nothing.
- GT lesions grouped by nnU-Net coverage: missed 214 (6 % of volume), partly < 50 % 155 (6 %), covered 1066 (88 %).
- PET IO lowers only covered lesions (7.8 -> 4.9 % MTV, 7.9 -> 5.1 % TLG); missed / partly unchanged (14.9 -> 15.7 %, 11.3 -> 11.5 %).
- Contribution to the evaluation metric after PET IO: missed 4.8 of 7.6 (MTV), 5.7 of 8.7 (TLG); covered 2.6 / 3.2.
- Why missed lesions weigh so much (I5, 26 patients, thirds by total lesion volume): low third has 45 % of its volume missed (mid 18 %, high 6 %) and gives 70 % of the missed contribution; Spearman(volume, missed contribution) = -0.32, p = 0.11 (descriptive).
- IO without PET losses worsens covered lesions (7.4 -> 7.8 % MTV, 6.9 -> 7.9 % TLG) and overall TLG (10.8 -> 11.9 %); overall MTV slightly down (10.6 -> 10.2 %).
- Open: covered lesions still contribute 2.6 of 7.6 points although IO sees them -> test with IO on GT lesion masks (G1, G2).
- Scripts: paper/figure_detection.py (in results), table_detection_groups.py / table_detection_contributions.py (tables, possibly dropped).

## [ ] PET IO weights: tiny weights suffice; with GT masks IO preserves almost fully (test, T1 + PET IO)

**Status:** confirmed — test set (28 patients, 38 pairs); W0–W1 and G1/G2 partly on the cluster (W0 matches I1: means equal to 0.1 points).

| PET scale | Dice GT | MTV pred → GT | TLG pred → GT |
|---|---|---|---|
| 0 | 79.2 | 8.6 → 10.2 | 7.5 → 11.8 |
| 0.0001 | 79.1 | 2.2 → 8.0 | 3.2 → 8.9 |
| 0.001 | 78.7 | 1.5 → 7.9 | 3.1 → 8.7 |
| 0.1 (tuned) | 77.5 | 1.0 → 7.6 | 3.3 → 8.7 |
| 1 | 77.5 | 1.0 → 7.5 | 3.3 → 8.6 |
| G1: GT lesion masks, x0.1 | 78.1 | → 2.1 | → 3.2 |
| G2: GT lesion masks, x1 | 77.7 | → 2.1 | → 3.2 |

- Already x0.0001 gives most of the preservation (GT MTV 10.2 -> 8.0) for 0.1 Dice points; the tuned x0.1 costs 1.7 Dice for 0.4 points more. The val tuning chose a weight that is too high (report honestly).
- With GT lesion masks (G1), GT MTV drops to 2.1 % (vs 7.6 %), TLG to 3.2 % (near the 2.9 % resampling floor), and Dice is higher (78.1 vs 77.5): the predicted masks, not the optimisation, limit PET IO. x1 (G2) gives the same: higher weights do not help even with correct masks.
- Coverage breakdown (contribution to MTV / TLG): I5 missed 4.8/5.7, partly 1.2/0.9, covered 2.6/3.2; G1 missed 1.5/2.2, partly 1.3/1.0, covered 1.2/1.8. Mask inaccuracy explains most of both the missed and the covered part, not all.

Source: `evaluation/test/W*.csv`, `G1.csv`, `G2.csv`; `journal_results/paper/figure_tradeoff.py`, `detection.py`.

## [ ] Rigidity weight sweep (val, T2 + PET IO)

**Status:** 15 runs done (`journal_results/rigidity_sweep/curve.csv`); chosen = largest weight within 1 Dice point of R1 (76.7 %).

- Staring: chosen x1 (0.659); costs bone alignment (bone Dice 77.3 -> 75.8; x10: 74.0), soft-tissue Dice unchanged; short-distance distortion 0.106 -> 0.073.
- Staring + erosion and Kabsch: within 1 Dice point up to x10 (grid edge) -> grid extended by x30, x100 (4 cluster runs).
- At x10: Staring + erosion distortion 0.062 / 0.036, Kabsch 0.063 / 0.028 (short / long), bone Dice 77.0 / 76.0.
