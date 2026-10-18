"""Tune ConvexAdam (ANTs affine -> MIND/SVF) on val, then register test.

    ../.venv/bin/python tune_convexadam.py               # grid on val -> chosen.json
    ../.venv/bin/python tune_convexadam.py --split test  # chosen config on test

Output: baselines/convexadam/ (see baselines.py).
"""

import baselines

# regularisation weight; everything else as in baselines.CONVEXADAM_DEFAULTS
# (the previous challenge setup, lambda 2)
GRID = {f"lambda{lam:g}": {"convex_lambda": lam} for lam in (0.1, 0.25, 0.5, 1.25, 2.0, 4.0)}

if __name__ == "__main__":
    baselines.main("convexadam", GRID)
