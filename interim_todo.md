# Work while T1–T4 train (temporary)

One step at a time: implement → user checks → tick.

- [ ] 1. **Predicted labels (user):** TotalSegmentator + nnU-Net on val and test scans, used as IO inputs.
- [x] 2. **Evaluation pipeline skeleton:** one script driven by the run IDs in the plan (T*, I*, L*, R*, W*, B*); writes one results row per pair (and per lesion); runs on val with current checkpoints.
- [x] 3. **Existing metrics wired in:** CT Dice, HD95, NDV, aggregate MTV/TLG bias, evaluated against ground-truth labels.
- [x] 4. **Per-lesion MTV/TLG:** signed, absolute, upper-tail; stratified by lesion size and bone association.
- [ ] 5. **Rigidity metrics:** per-bone Dice/surface distance, within-bone distance distortion, NDV in bone / boundary band / body, BRE, stencil leakage, erosion retention.
- [ ] 6. **Staring + Staring-with-erosion in IO:** available as IO loss options next to Kabsch.
- [ ] 7. **IO tuning on val:** step size, stopping rule, loss weights (predicted labels as inputs); freeze them.
- [ ] 8. **Baselines B1–B4:** initial alignment, affine, NiftyReg, ConvexAdam (tune the last two on val), run on test.
- [ ] 9. **Statistics:** patient-level paired effects with uncertainty (e.g. patient-cluster bootstrap).
- [ ] 10. **Dry run:** full pipeline on val with current checkpoints, so final runs only swap in the finished checkpoints.
