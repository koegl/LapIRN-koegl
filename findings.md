# Findings

Running log of results that matter for the paper. Each entry states its data
basis; **preliminary** = not yet confirmed on the full dataset.

## Paper story (order of results and discussion)

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
   Early stopping does not replace them (T1-tum, text only; entry below).
4. **Q2 — where should the losses act?** (`q2.pdf`): losses only in IO (I7)
   vs. in training + IO (I4): Dice +1.7 points (significant), preservation not
   significantly worse. If IO is affordable, IO-only placement is enough and
   more accurate; without IO (14× faster: ~11 s vs. ~157 s per pair), the
   training losses are the only way to get preservation.
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

## Staring stencils leak; erosion leaves most of the skeleton unconstrained

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

## The scorer's warp changes TLG even for the identity transform

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

## NiftyReg folds more on test than on val

**Status:** confirmed — val-chosen config be 0.0002, sx −9 (38 val / 38 test pairs).

- %NDV mean 0.016 (val) vs. **0.093** (test); the median is 0 on both.
- On test, three pairs fold strongly (0.8–1.3%; cases 0338, 0166, 0170).
- Cause: the ranking picks the least-regularised config on val (the
  accuracy scores separate configs; NDV only barely). Report as is.

Source: `evaluation/test/B3.csv`, `journal_results/baselines/niftyreg/eval/val/`.

## IO PET terms work on their mask; predicted lesion labels limit them

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

## Early stopping on tumour preservation does not replace the PET losses

**Status:** confirmed — test set (28 patients, 38 pairs); text only, no table.

- T1-tum (T1, checkpoint selected on tumour preservation instead of accuracy)
  vs. T1: CT Dice −3.0 points (significant), |ΔMTV| −0.4 and |ΔTLG| −0.9
  points (both not significant).
- For comparison, training with the PET losses (T2 vs. T1): Dice −1.9,
  |ΔMTV| −3.0 points (both significant).
- So stopping early costs more accuracy and buys almost no preservation.

Source: `evaluation/test/T1.csv`, `T1-tum.csv`, `T2.csv`;
`journal_results/paper/table_ablations.py` (paired Wilcoxon, Holm).
