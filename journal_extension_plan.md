# Journal extension plan

**Deadline:** October 30, 2026.  
**Aim:** Establish how PET preservation and bone rigidity affect alignment, lesion quantification, and deformation quality.

## Evaluation protocol

- Split the labelled challenge training dataset into train/validation/test **by patient**, keeping all scans and pairs together.
- Train registration and nnU-Net on training patients only; tune weights, checkpoints, and IO stopping on validation patients.
- At test time, use TotalSegmentator and nnU-Net predictions for IO. Reserve provided labels for evaluation only; document their provenance.
- Retrain under the new split, disclose previous patient exposure, and preferentially reserve previously unused patients for testing.
- Report paired effects and patient-level uncertainty, accounting for multiple pairs/lesions per patient.

## Core experiments

### 1. PET and rigidity ablations

- **2×2:** neither / PET only / rigidity only / both; retain the same accuracy and general regularisation terms.
- Leave out each PET term individually to measure its contribution within the full objective.
- Report alignment, NDV, and aggregate plus per-lesion MTV/TLG changes (signed, absolute, and upper-tail errors).
- Stratify lesions by bone association and size. Preservation means before versus after warping the same scan, not equality across time points.
- Distinguish matched training ablations from IO-only ablations; prioritise matched training for central claims.

### 2. Rigidity comparison

**Methods:** no rigidity / Staring (all three components) / Staring + erosion / Kabsch.

This is an evaluation of existing approaches in low-resolution whole-body CT, not a claim of a new rigidity method.

| Question | Measurement |
|---|---|
| Bone alignment | Per-bone Dice and surface distance |
| Soft-tissue alignment | Organ Dice and surface distance; near-bone landmarks if available |
| Bone shape preservation | Within-bone point-pair distance distortion in physical coordinates; sample short and long distances |
| Deformation quality | NDV/folding inside bones, in an outside boundary band, and across the body |
| Boundary mechanism | Direct stencil leakage; fraction of bones/voxels retained after erosion |

- Tune/sweep each method's weight separately; compare alignment versus distance distortion, with folding alongside.
- Use bone rigidity error (BRE) only as a secondary endpoint because it closely matches the Kabsch objective. Distance distortion provides a complementary endpoint.
- Kabsch avoids direct stencil leakage, but can still influence surrounding tissue through the shared deformation and regularisation.
- Scope conclusions to the tested resolutions, implementations, and weight ranges.

### 3. Accuracy–preservation trade-off

- Run a small PET-weight sweep including zero and the current setting.
- Plot alignment versus lesion preservation and report deformation quality.
- Select settings on validation data; evaluate frozen settings on test data.

## Supporting / optional work

- **Gradient analysis:** one loss-pair cosine matrix, selected IO trajectories, and representative maps. Distinguish displacement-space maps from gradients in the optimised velocity coordinates; mask near-zero gradients and report weighted magnitudes. Cosines describe local interactions, not final trade-offs.
- **External cohort:** only if feasible without delaying submission. Combine automatic surrogate measurements with blinded, randomised clinician review and sparse landmarks if possible. Keep IO labels separate from independent evaluation; distinguish resampling effects from deformation effects.

## Schedule

| Dates | Target |
|---|---|
| Sep 28–Oct 2 | Freeze split, protocol, endpoints, and experiment list; decide external-cohort feasibility |
| Oct 3–15 | Complete core ablations, rigidity comparison, and weight sweep |
| Oct 16–23 | Final analysis, figures, and complete manuscript draft |
| Oct 24–30 | Coauthor review, revisions, and submission |

**If time is short:** protect independent evaluation and core experiments; reduce gradient analysis and omit the external cohort first.
