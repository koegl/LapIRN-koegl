"""Tune NiftyReg (reg_aladin -> reg_f3d, GPU, NMI) on val, then register test.

    ../.venv/bin/python tune_niftyreg.py               # grid on val -> chosen.json
    ../.venv/bin/python tune_niftyreg.py --split test  # chosen config on test

Output: baselines/niftyreg/ (see baselines.py).
"""

import baselines

# bending energy x control-point spacing (negative = voxels); everything else
# as in baselines.NIFTYREG_DEFAULTS (the previous challenge setup: be 0.005,
# sx -5). LNCC is not available on the GPU, so the similarity stays NMI.
GRID = {
    f"be{be}_sx{-sx:g}": {"f3d_be": be, "f3d_sx": sx}
    for be in (0.001, 0.005, 0.02)
    for sx in (-3.0, -5.0)
}

if __name__ == "__main__":
    baselines.main("niftyreg", GRID)
