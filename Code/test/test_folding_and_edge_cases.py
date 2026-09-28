"""Controls for regional folding detection and lesion metric edge cases.

Run directly:
    uv run python Code/test/test_folding_and_edge_cases.py

or with pytest:
    pytest Code/test/test_folding_and_edge_cases.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jacobian  # noqa: E402
from lesion_preservation_metric import (  # noqa: E402
    evaluate_lesion_preservation,
    transformed_in_fov_fraction,
)


SHAPE = (25, 27, 29)
VOXEL_VOLUME_ML = 0.012
MIN_LESION_VOXELS = 8


def _deliberately_folded_field() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reflect the left part of the image along D and leave the right rigid."""
    d, h, w = SHAPE
    dd = torch.arange(d, dtype=torch.float64).view(d, 1, 1)
    flow = torch.zeros((1, d, h, w, 3), dtype=torch.float64)
    flow[0, :, :, : w // 2, 0] = -2.0 * (dd - (d - 1.0) / 2.0)

    folded_region = torch.zeros((1, 1, d, h, w), dtype=torch.float64)
    folded_region[:, :, 2:-2, 2:-2, 2 : w // 2 - 1] = 1.0
    unaffected_region = torch.zeros_like(folded_region)
    unaffected_region[:, :, 2:-2, 2:-2, w // 2 + 2 : -2] = 1.0
    return flow, folded_region, unaffected_region


def test_folding_is_detected_only_in_the_folded_region() -> None:
    flow, folded_region, unaffected_region = _deliberately_folded_field()
    folded_ndv = jacobian.percent_ndv(flow, mask=folded_region)
    unaffected_ndv = jacobian.percent_ndv(flow, mask=unaffected_region)
    determinant, _ = jacobian.jacobian_matrix(flow)

    folded_values = determinant[folded_region.bool()]
    unaffected_values = determinant[unaffected_region.bool()]
    assert folded_ndv > 0.0
    assert unaffected_ndv == 0.0
    assert torch.all(folded_values < 0.0)
    assert torch.allclose(unaffected_values, torch.ones_like(unaffected_values))


def _constant_pet() -> np.ndarray:
    return np.full(SHAPE, 5.0, dtype=np.float32)


def test_empty_source_mask_is_explicitly_excluded() -> None:
    empty = np.zeros(SHAPE, dtype=bool)
    result = evaluate_lesion_preservation(
        empty,
        empty,
        _constant_pet(),
        _constant_pet(),
        VOXEL_VOLUME_ML,
        min_source_voxels=MIN_LESION_VOXELS,
        in_fov_fraction=None,
    )

    assert result["included"] is False
    assert result["status"] == "excluded_empty_source"
    assert result["mtv_percent_error"] is None
    assert result["tlg_percent_error"] is None


def test_tiny_lesion_is_reported_but_excluded() -> None:
    tiny = np.zeros(SHAPE, dtype=bool)
    tiny[8, 9, 10:13] = True
    result = evaluate_lesion_preservation(
        tiny,
        tiny,
        _constant_pet(),
        _constant_pet(),
        VOXEL_VOLUME_ML,
        min_source_voxels=MIN_LESION_VOXELS,
    )

    assert result["included"] is False
    assert result["status"] == "excluded_tiny_source"
    assert result["source_voxels"] == 3
    assert result["warped_voxels"] == 3
    assert result["mtv_percent_error"] == 0.0
    assert result["tlg_percent_error"] == 0.0


def test_lesion_leaving_image_is_reported_and_excluded() -> None:
    source = np.zeros(SHAPE, dtype=bool)
    source[9:12, 10:13, 22:27] = True
    shift = np.array([0.0, 0.0, 5.0])
    in_fov = transformed_in_fov_fraction(
        source,
        lambda points: points + shift,
        SHAPE,
    )

    warped = np.zeros_like(source)
    warped[9:12, 10:13, 27:29] = True
    result = evaluate_lesion_preservation(
        source,
        warped,
        _constant_pet(),
        _constant_pet(),
        VOXEL_VOLUME_ML,
        min_source_voxels=MIN_LESION_VOXELS,
        in_fov_fraction=in_fov,
    )

    assert np.isclose(in_fov, 2.0 / 5.0)
    assert result["included"] is False
    assert result["status"] == "excluded_out_of_fov"
    assert result["warped_voxels"] < result["source_voxels"]
    assert np.isfinite(result["mtv_percent_error"])
    assert np.isfinite(result["tlg_percent_error"])
    assert result["mtv_percent_error"] > 0.0
    assert result["tlg_percent_error"] > 0.0


def main() -> None:
    test_folding_is_detected_only_in_the_folded_region()
    test_empty_source_mask_is_explicitly_excluded()
    test_tiny_lesion_is_reported_but_excluded()
    test_lesion_leaving_image_is_reported_and_excluded()
    print(
        "OK - folding is localized; empty, tiny, and out-of-FOV lesions have "
        "explicit finite reporting rules"
    )


if __name__ == "__main__":
    main()
