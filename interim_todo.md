# Work while T1–T4 train (temporary)

One step at a time: implement → user checks → tick.

- [ ] **Predicted labels (user):** TotalSegmentator + nnU-Net on val and test scans, used as IO inputs.
- [ ] **Evaluation pipeline skeleton:** one script driven by the run IDs in the plan (T*, I*, L*, R*, W*, B*); writes one results row per pair (and per lesion); runs on val with current checkpoints.
- [ ] **Existing metrics wired in:** CT Dice, HD95, NDV, aggregate MTV/TLG bias, evaluated against ground-truth labels.
- [ ] **Per-lesion MTV/TLG:** signed, absolute, upper-tail; stratified by lesion size and bone association.
- [ ] **Rigidity metrics:** per-bone Dice/surface distance, within-bone distance distortion, NDV in bone / boundary band / body, BRE, stencil leakage, erosion retention.
- [ ] **Staring + Staring-with-erosion in IO:** available as IO loss options next to Kabsch.
- [ ] **IO tuning on val:** step size, stopping rule, loss weights (predicted labels as inputs); freeze them.
- [ ] **Baselines B1–B4:** initial alignment, affine, NiftyReg, ConvexAdam (tune the last two on val), run on test.
- [ ] **Statistics:** patient-level paired effects with uncertainty (e.g. patient-cluster bootstrap).
- [ ] **Dry run:** full pipeline on val with current checkpoints, so final runs only swap in the finished checkpoints.
