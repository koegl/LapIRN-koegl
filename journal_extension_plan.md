# Journal extension plan

**Deadline:** October 30, 2026.  
**Aim:** Establish how PET preservation and bone rigidity affect alignment, lesion quantification, and deformation quality.

**Main questions**

1. Do PET preservation and bone rigidity help the full pipeline (network + IO)?
2. Must these losses be in training, or is adding them in IO enough?

## Evaluation protocol

- **Split (frozen, `split_journal.json`):** patient-level 70/15/15 % of pairs, balanced for tumour burden and bone/soft-tissue lesion location; 92/26/28 patients, 178/38/38 pairs. Train uses all forward pairs, val/test only follow-up→baseline pairs. All patients were seen in the conference work; disclose this.
- Train registration and nnU-Net on training patients only; tune all weights, checkpoints, and IO settings on validation patients, then freeze them for test.
- At test time, use TotalSegmentator and nnU-Net predictions for IO. Reserve provided labels for evaluation only; document their provenance.
- Report paired effects and patient-level uncertainty, accounting for multiple pairs/lesions per patient.

## Training setup

- **Baseline sweep (no PET/rigidity losses):** architecture first, (`n_resblocks`, `resblock_expansion`) ∈ {(5,1), (5,2), (10,1), (10,2)}, lowering `lr_lvl3` only if lvl3 diverges; then sweep regularisation (`w_smooth`, `w_non_diff`) against `w_ct` and `w_dice_ct_*` for the largest stable model.
- All runs: same step budget, warmup + cosine LR decay (to 10 %), same seed.
- **Checkpoint selection (all runs):** lvl1/2 by validation CT Dice; lvl3 `best_accuracy_capped` = best accuracy (CT Dice + HD95) with validation %NDV ≤ 0.00014. Not selected on MTV/TLG, so ablations are not selected on the effect they measure. Report the final checkpoint in the supplement.
- One shared lvl2 checkpoint for all later runs; PET and rigidity terms enter at lvl3 and in IO only (state this in the paper).

## Core experiments

Evaluation modes: **network** = network output only; **IO** = network + IO. IO settings are chosen on validation and identical across conditions.

### 1. PET and rigidity ablations (trained; answers both main questions)

- **2×2 training:** neither / PET only / rigidity only / both; same accuracy and regularisation terms.
- **Network:** primary result for the training effect.
- **IO (7 runs):**
  - *Matched (4):* each network with its own losses in IO → Question 1.
  - *IO-only (+3):* the "neither" network with each other IO setting → Question 2, compared with the matched runs.
- **Leave-one-out (IO, 4 runs):** the "neither" network with full PET IO minus one PET term at a time; reference is the full-PET IO run above. Measures each term's contribution in IO, not in training (state this).
- Report alignment, NDV, and aggregate plus per-lesion MTV/TLG changes (signed, absolute, upper-tail). Stratify lesions by bone association and size. Preservation means before vs. after warping the same scan.
- Early-stopping reference: also evaluate the "neither" network's `best_tumour` checkpoint to show what accuracy early stopping gives up compared with the PET losses.

### 2. Rigidity comparison (IO only)

**Methods:** no rigidity / Staring (all three components) / Staring + erosion / Kabsch, applied in IO on the **PET-only network** with PET terms on in IO for all methods. An evaluation of existing approaches in low-resolution whole-body CT, not a new method. Optionally retrain with the winning method if time allows.

| Question | Measurement |
|---|---|
| Bone alignment | Per-bone Dice and surface distance |
| Soft-tissue alignment | Organ Dice and surface distance; near-bone landmarks if available |
| Bone shape preservation | Within-bone point-pair distance distortion (physical coordinates; short and long distances) |
| Deformation quality | NDV/folding inside bones, in an outside boundary band, and across the body |
| Boundary mechanism | Direct stencil leakage; fraction of bones/voxels retained after erosion |

- Tune each method's weight separately; compare alignment vs. distance distortion, with folding alongside.
- BRE only as a secondary endpoint (it closely matches the Kabsch objective).
- Scope conclusions to the tested resolutions, implementations, and weight ranges.

### 3. Accuracy–preservation trade-off (IO only)

- Sweep the IO PET weight (including zero and the current setting) on a fixed network; plot alignment vs. lesion preservation with deformation quality.
- Select on validation; evaluate frozen settings on test.

### 4. Baseline comparison (network and IO)

- Initial alignment, affine, NiftyReg, ConvexAdam, and the full method with and without IO on the test set.

## Supporting / optional work

- **Gradient analysis:** one loss-pair cosine matrix, selected IO trajectories, and representative maps. Cosines describe local interactions, not final trade-offs.
- **External cohort:** only if feasible without delaying submission (surrogate measurements plus blinded clinician review).

**If time is short:** protect independent evaluation and experiments 1 and 4; reduce gradient analysis and omit the external cohort first.
