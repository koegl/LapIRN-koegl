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

**Status:** preliminary — 1 validation pair (case 0071, 01→00), our copy of the
scorer's transformer (`evaluate.SpatialTransformer`, from voxelmorph). To check
against the official scorer code.

- The transformer normalises voxel positions with (n − 1) but samples with
  `align_corners=False`, so voxel i is read at i·n/(n − 1) − 0.5: a zero field
  shifts the image by up to half a voxel (−0.5 at one end, +0.5 at the other).
- With an all-zero field, the linearly resampled PET changes TLG by **2.5%**;
  MTV (nearest-neighbour mask) stays at 0%.
- Handling: the "initial" row (B1) is scored without any resampling (MTV / TLG
  change exactly 0); all other methods keep the scorer's warp. The floor itself
  is measured by B1-floor (identity through the scorer's warp) on the full test
  set and stated in the paper as a resampling effect common to all methods.

Source: `evaluate.py` runs B1, B1-floor.
