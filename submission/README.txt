PSMAReg (Learn2Reg 2026) -- test-phase submission
Image: psmareg_koegl   (docker load --input psmareg_koegl.tar.gz  ->  psmareg_koegl:latest)

ARGUMENT ORDER
  Five positional paths, in this order:

  docker run --rm --ipc=host --memory 60g --gpus "device=0" \
      --user $(id -u):$(id -g) --network=none \
      --mount type=bind,source=<test image dir>,target=/app/input,readonly \
      --mount type=bind,source=<predictions dir>,target=/app/output \
      psmareg_koegl \
          /app/input/PSMARegPSMA_XXXX_0000_00.nii.gz   fixed CT   (baseline)
          /app/input/PSMARegPSMA_XXXX_0001_00.nii.gz   fixed PET  (baseline)
          /app/input/PSMARegPSMA_XXXX_0000_01.nii.gz   moving CT  (follow-up 01)
          /app/input/PSMARegPSMA_XXXX_0001_01.nii.gz   moving PET (follow-up 01)
          /app/output/disp_XXXX_00_XXXX_01.nii.gz      output displacement field

  No other arguments are needed; all defaults are baked into the image.

REQUIREMENTS
  GPU:   1x CUDA GPU, TODO GB VRAM peak
  CPU:   TODO cores
  RAM:   TODO GB peak
  Time:  ~90 s/pair  ->  ~5 h for 200 pairs

  The runtime is set, not measured: the container holds an internal 90s per-pair wall-clock budget, and the instance optimisation stage takes as many steps as fit inside it, stopping early enough that the field is always written. A faster machine therefore spends the same ~90 s and takes more steps; it does not finish sooner.

DETERMINISM
  torch.manual_seed(0), np.random.seed(0), cudnn.deterministic = True,
  cudnn.benchmark = False. Network inference and instance optimisation are
  deterministic.

  Two documented sources of nondeterminism:
  - The ANTs affine pre-registration (mattes metric, 32 sampling points) uses ITK's own RNG, not seeded from Python.
    Small run-to-run differences in the affine stage propagate to the field at the sub-voxel level.
  - The step count of the instance optimisation follows the wall-clock budget described under Time, so it varies with machine speed and load.
