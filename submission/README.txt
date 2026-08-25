PSMAReg (Learn2Reg 2026) -- test-phase submission
Method: LapIRN (Laplacian pyramid, diffeomorphic) with ANTs affine prereg
Image:  psmareg_lapirn

ARGUMENT ORDER (five positional paths, as specified by the organizers)
  <image>  fixed_ct  fixed_pet  moving_ct  moving_pet  output_disp
  i.e.     ..._0000_00  ..._0001_00  ..._0000_01  ..._0001_01  <output path>

OUTPUT
  NIfTI, channel-first (3, 192, 192, 288), float32.
  Voxel displacements on the fixed-image grid, full input resolution.
  Warping the moving image by this field aligns it to the fixed image.

REQUIREMENTS
  GPU:   1x CUDA GPU, TODO GB VRAM peak (measured on an RTX A6000)
  CPU:   TODO cores (the ANTs affine stage is CPU-bound and single-threaded)
  RAM:   TODO GB
  Time:  TODO s/pair  ->  TODO min for 200 pairs (budget: 300 min)

  Run as provided in the instructions; no network access is required.

DETERMINISM
  torch.manual_seed(0), np.random.seed(0), cudnn.deterministic = True,
  cudnn.benchmark = False. The network inference is deterministic.
  The ANTs affine registration (mattes metric, 32 sampling points) uses ITK's
  own RNG and is not seeded from Python; small run-to-run differences in the
  affine stage are therefore possible, and propagate to the field at the
  sub-voxel level.

NOTES
  TODO
