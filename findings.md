# Findings

Running log of results that matter for the paper. Each entry states its data
basis; **preliminary** = not yet confirmed on the full dataset.

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
