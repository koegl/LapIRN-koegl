"""Synthetic control for cancellation of opposing per-lesion MTV changes.

Run directly:
    uv run python Code/test/test_per_lesion_cancellation.py

Two lesions start with the same physical volume. One is enlarged by exactly the
number of voxels removed from the other. The aggregate challenge-style MTV
error therefore vanishes, while each lesion has a nonzero signed and absolute
error and the per-component training loss remains nonzero.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import utils  # noqa: E402


SHAPE = (64, 64, 64)
SPACING_MM = np.array([2.0, 2.0, 3.0], dtype=np.float64)
INITIAL_VOXELS_PER_LESION = 600
TRANSFERRED_VOXELS = 120
CENTRES = (
    np.array([18.0, 32.0, 32.0]),
    np.array([46.0, 32.0, 32.0]),
)


def _radial_mask(centre: np.ndarray, n_voxels: int) -> np.ndarray:
    """Select the ``n_voxels`` voxel centres nearest to ``centre`` in mm."""
    coords = np.indices(SHAPE, dtype=np.float64).reshape(3, -1).T
    distance_sq_mm = np.sum(((coords - centre) * SPACING_MM) ** 2, axis=1)
    nearest = np.argpartition(distance_sq_mm, n_voxels - 1)[:n_voxels]
    mask = np.zeros(np.prod(SHAPE), dtype=bool)
    mask[nearest] = True
    return mask.reshape(SHAPE)


def _make_lesions() -> tuple[np.ndarray, np.ndarray]:
    before = np.stack(
        [
            _radial_mask(CENTRES[0], INITIAL_VOXELS_PER_LESION),
            _radial_mask(CENTRES[1], INITIAL_VOXELS_PER_LESION),
        ]
    )
    after = np.stack(
        [
            _radial_mask(
                CENTRES[0], INITIAL_VOXELS_PER_LESION + TRANSFERRED_VOXELS
            ),
            _radial_mask(
                CENTRES[1], INITIAL_VOXELS_PER_LESION - TRANSFERRED_VOXELS
            ),
        ]
    )

    if np.any(before[0] & before[1]) or np.any(after[0] & after[1]):
        raise AssertionError("synthetic lesions overlap")
    if not np.all(before[0] <= after[0]):
        raise AssertionError("lesion 1 was not a pure expansion")
    if not np.all(after[1] <= before[1]):
        raise AssertionError("lesion 2 was not a pure compression")
    return before, after


def evaluate_per_lesion_cancellation() -> dict[str, object]:
    before_np, after_np = _make_lesions()
    voxel_volume_ml = float(np.prod(SPACING_MM) / 1000.0)
    before_volume_ml = before_np.sum(axis=(1, 2, 3)) * voxel_volume_ml
    after_volume_ml = after_np.sum(axis=(1, 2, 3)) * voxel_volume_ml

    signed_change_ml = after_volume_ml - before_volume_ml
    signed_error = signed_change_ml / before_volume_ml
    absolute_error = np.abs(signed_error)

    aggregate_before_ml = float(before_volume_ml.sum())
    aggregate_after_ml = float(after_volume_ml.sum())
    aggregate_signed_error = (
        aggregate_after_ml - aggregate_before_ml
    ) / aggregate_before_ml
    aggregate_absolute_error = abs(aggregate_signed_error)

    before_cc = torch.from_numpy(before_np.astype(np.float32))[None]
    after_cc = torch.from_numpy(after_np.astype(np.float32))[None]
    before_union = before_cc.sum(dim=1, keepdim=True)
    after_union = after_cc.sum(dim=1, keepdim=True)
    aggregate_loss = float(utils.mtv_bias_loss(after_union, before_union).item())
    per_component_loss = float(
        utils.mtv_bias_loss_per_component(after_cc, before_cc).item()
    )

    return {
        "voxel_volume_ml": voxel_volume_ml,
        "before_volume_ml": before_volume_ml,
        "after_volume_ml": after_volume_ml,
        "signed_change_ml": signed_change_ml,
        "signed_error": signed_error,
        "absolute_error": absolute_error,
        "aggregate_before_ml": aggregate_before_ml,
        "aggregate_after_ml": aggregate_after_ml,
        "aggregate_signed_error": aggregate_signed_error,
        "aggregate_absolute_error": aggregate_absolute_error,
        "mean_individual_absolute_error": float(absolute_error.mean()),
        "aggregate_loss": aggregate_loss,
        "per_component_loss": per_component_loss,
    }


def _assert_per_lesion_cancellation(result: dict[str, object]) -> None:
    signed_change_ml = np.asarray(result["signed_change_ml"])
    signed_error = np.asarray(result["signed_error"])
    absolute_error = np.asarray(result["absolute_error"])

    expected_change_ml = TRANSFERRED_VOXELS * float(result["voxel_volume_ml"])
    assert np.allclose(signed_change_ml, [expected_change_ml, -expected_change_ml])
    assert signed_error[0] > 0.0
    assert signed_error[1] < 0.0
    assert np.all(absolute_error > 0.0)
    assert np.allclose(absolute_error, [0.20, 0.20])

    assert abs(float(result["aggregate_signed_error"])) < 1e-12
    assert float(result["aggregate_absolute_error"]) < 1e-12
    assert float(result["aggregate_loss"]) < 1e-12
    assert np.isclose(float(result["mean_individual_absolute_error"]), 0.20)
    assert np.isclose(float(result["per_component_loss"]), 0.20**2, atol=1e-7)


def test_per_lesion_cancellation() -> None:
    _assert_per_lesion_cancellation(evaluate_per_lesion_cancellation())


def main() -> None:
    result = evaluate_per_lesion_cancellation()
    _assert_per_lesion_cancellation(result)
    before = np.asarray(result["before_volume_ml"])
    after = np.asarray(result["after_volume_ml"])
    signed = 100.0 * np.asarray(result["signed_error"])
    absolute = 100.0 * np.asarray(result["absolute_error"])

    print("Per-lesion cancellation with equal and opposite physical-volume changes")
    print("lesion  MTV_before_ml  MTV_after_ml  signed_error_%  absolute_error_%")
    for index in range(2):
        print(
            f"{index + 1:>6d}"
            f"  {before[index]:>13.6f}"
            f"  {after[index]:>12.6f}"
            f"  {signed[index]:>14.6f}"
            f"  {absolute[index]:>16.6f}"
        )
    print(f"aggregate signed error: {100.0 * result['aggregate_signed_error']:.6f}%")
    print(
        "aggregate absolute error: "
        f"{100.0 * result['aggregate_absolute_error']:.6f}%"
    )
    print(
        "mean individual absolute error: "
        f"{100.0 * result['mean_individual_absolute_error']:.6f}%"
    )
    print(f"per-component squared loss: {result['per_component_loss']:.6f}")
    print("OK - aggregate cancellation does not hide per-lesion errors")


if __name__ == "__main__":
    main()
