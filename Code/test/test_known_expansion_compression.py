"""Known expansion/compression test for MTV and TLG physical scaling.

Run directly:
    uv run python Code/test/test_known_expansion_compression.py

A synthetic spherical lesion has constant PET intensity. A known forward length
scale ``s`` should change both its physical volume and TLG by approximately
``s**3``; small differences are expected from discrete mask sampling. PET is
set to the same constant inside the transformed lesion so this geometric check
is not confounded by the interpolation baseline tested separately.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


SHAPE = (65, 67, 69)
SPACING_MM = np.array([2.0, 2.5, 3.0], dtype=np.float64)
RADIUS_MM = 22.0
PET_INTENSITY = 7.0
LENGTH_SCALES = (0.80, 1.25)


def _synthetic_lesion() -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Return constant-intensity PET and a spherical binary lesion mask."""
    centre = (np.asarray(SHAPE, dtype=np.float64) - 1.0) / 2.0
    axes = np.meshgrid(
        *[np.arange(size, dtype=np.float64) for size in SHAPE],
        indexing="ij",
    )
    distance_sq_mm = sum(
        ((axis - centre[i]) * SPACING_MM[i]) ** 2
        for i, axis in enumerate(axes)
    )
    mask_np = distance_sq_mm <= RADIUS_MM**2
    pet_np = PET_INTENSITY * mask_np.astype(np.float32)

    pet = torch.from_numpy(pet_np)[None, None]
    mask = torch.from_numpy(mask_np.astype(np.float32))[None, None]
    return pet, mask, centre


def _inverse_scale_grid(
    shape: tuple[int, int, int],
    centre: np.ndarray,
    forward_scale: float,
) -> torch.Tensor:
    """Sampling grid for the forward map ``x' = c + s (x - c)``.

    ``grid_sample`` is a pull operation, so each output point ``x'`` samples
    the source at the inverse location ``x = c + (x' - c) / s``.
    """
    depth, height, width = shape
    c = torch.as_tensor(centre, dtype=torch.float32)

    d_out = torch.arange(depth, dtype=torch.float32)
    h_out = torch.arange(height, dtype=torch.float32)
    w_out = torch.arange(width, dtype=torch.float32)
    d_src = c[0] + (d_out - c[0]) / forward_scale
    h_src = c[1] + (h_out - c[1]) / forward_scale
    w_src = c[2] + (w_out - c[2]) / forward_scale

    d_norm = 2.0 * (d_src + 0.5) / depth - 1.0
    h_norm = 2.0 * (h_src + 0.5) / height - 1.0
    w_norm = 2.0 * (w_src + 0.5) / width - 1.0
    dd, hh, ww = torch.meshgrid(d_norm, h_norm, w_norm, indexing="ij")
    return torch.stack([ww, hh, dd], dim=-1).unsqueeze(0)


def _warp(volume: torch.Tensor, grid: torch.Tensor, mode: str) -> torch.Tensor:
    return F.grid_sample(
        volume,
        grid,
        mode=mode,
        padding_mode="zeros",
        align_corners=False,
    )


def evaluate_known_scales() -> list[dict[str, float]]:
    pet, mask, centre = _synthetic_lesion()
    voxel_volume_ml = float(np.prod(SPACING_MM) / 1000.0)
    mtv_before_ml = float(mask.sum().item() * voxel_volume_ml)
    tlg_before = float((pet * mask).sum().item() * voxel_volume_ml)

    rows = []
    for scale in LENGTH_SCALES:
        grid = _inverse_scale_grid(SHAPE, centre, scale)
        mask_after = _warp(mask, grid, mode="nearest")
        # Keep uptake constant by definition: only lesion geometry changes.
        pet_after = PET_INTENSITY * mask_after

        mtv_after_ml = float(mask_after.sum().item() * voxel_volume_ml)
        tlg_after = float((pet_after * mask_after).sum().item() * voxel_volume_ml)
        mtv_ratio = mtv_after_ml / mtv_before_ml
        tlg_ratio = tlg_after / tlg_before
        expected_ratio = scale**3

        rows.append(
            {
                "length_scale": scale,
                "expected_ratio": expected_ratio,
                "mtv_before_ml": mtv_before_ml,
                "mtv_after_ml": mtv_after_ml,
                "mtv_ratio": mtv_ratio,
                "tlg_before": tlg_before,
                "tlg_after": tlg_after,
                "tlg_ratio": tlg_ratio,
                "mtv_relative_error": abs(mtv_ratio - expected_ratio)
                / expected_ratio,
                "tlg_relative_error": abs(tlg_ratio - expected_ratio)
                / expected_ratio,
            }
        )
    return rows


def _assert_known_scales(rows: list[dict[str, float]]) -> None:
    assert len(rows) == len(LENGTH_SCALES)
    for row in rows:
        scale = row["length_scale"]
        expected = scale**3
        assert np.isclose(row["expected_ratio"], expected)
        assert row["mtv_relative_error"] < 0.05
        assert row["tlg_relative_error"] < 0.05
        assert np.isclose(row["tlg_before"], PET_INTENSITY * row["mtv_before_ml"])
        assert np.isclose(row["tlg_after"], PET_INTENSITY * row["mtv_after_ml"])
        assert np.isclose(row["tlg_ratio"], row["mtv_ratio"])
        if scale < 1.0:
            assert row["mtv_ratio"] < 1.0
            assert row["tlg_ratio"] < 1.0
        else:
            assert row["mtv_ratio"] > 1.0
            assert row["tlg_ratio"] > 1.0


def test_known_expansion_and_compression() -> None:
    _assert_known_scales(evaluate_known_scales())


def main() -> None:
    rows = evaluate_known_scales()
    _assert_known_scales(rows)
    print("Known expansion/compression of a constant-intensity synthetic lesion")
    print("scale  expected_s^3  MTV_ratio  TLG_ratio  MTV_err_%  TLG_err_%")
    for row in rows:
        print(
            f"{row['length_scale']:>5.2f}"
            f"  {row['expected_ratio']:>12.6f}"
            f"  {row['mtv_ratio']:>9.6f}"
            f"  {row['tlg_ratio']:>9.6f}"
            f"  {100.0 * row['mtv_relative_error']:>9.4f}"
            f"  {100.0 * row['tlg_relative_error']:>9.4f}"
        )
    print("OK - MTV and TLG follow the expected cubic physical scaling")


if __name__ == "__main__":
    main()
