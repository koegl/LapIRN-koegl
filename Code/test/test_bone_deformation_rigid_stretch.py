"""Validate the bone-distance metric with rigid motion and physical stretch.

Run directly:
    python Code/test/test_bone_deformation_rigid_stretch.py

or with pytest:
    pytest Code/test/test_bone_deformation_rigid_stretch.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bone_deformation_metric import (  # noqa: E402
    PairSamplingConfig,
    measure_bone_deformation,
)


SPACING_MM = np.array([1.3, 2.1, 3.7], dtype=np.float64)
STRETCH = 1.18


def _synthetic_bones() -> np.ndarray:
    labels = np.zeros((28, 30, 32), dtype=np.int16)
    labels[3:11, 4:14, 5:17] = 1
    labels[15:26, 12:28, 10:29] = 2
    return labels


def _sampling() -> PairSamplingConfig:
    return PairSamplingConfig(
        pairs_per_bin=96,
        candidate_multiplier=20,
        seed=23,
    )


def _physical_rigid_transform(labels: np.ndarray):
    centre_mm = 0.5 * (np.asarray(labels.shape, dtype=np.float64) - 1.0) * SPACING_MM
    translation_mm = np.array([8.0, -5.5, 3.25], dtype=np.float64)

    angle_z = np.deg2rad(31.0)
    angle_x = np.deg2rad(-17.0)
    cz, sz = np.cos(angle_z), np.sin(angle_z)
    cx, sx = np.cos(angle_x), np.sin(angle_x)
    rotate_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    rotate_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    rotation = rotate_x @ rotate_z

    def rigid(points_voxel: np.ndarray) -> np.ndarray:
        points_mm = points_voxel * SPACING_MM
        transformed_mm = (points_mm - centre_mm) @ rotation.T
        transformed_mm += centre_mm + translation_mm
        return transformed_mm / SPACING_MM

    def rigid_then_stretch(points_voxel: np.ndarray) -> np.ndarray:
        rigid_mm = rigid(points_voxel) * SPACING_MM
        stretched_mm = centre_mm + STRETCH * (rigid_mm - centre_mm)
        return stretched_mm / SPACING_MM

    return rigid, rigid_then_stretch


def test_rigid_motion_and_stretch_use_physical_millimetres() -> None:
    labels = _synthetic_bones()
    rigid, rigid_then_stretch = _physical_rigid_transform(labels)

    rigid_result = measure_bone_deformation(
        labels,
        SPACING_MM,
        transform_points=rigid,
        patient_id="anisotropic-rigid",
        sampling=_sampling(),
    )
    stretch_result = measure_bone_deformation(
        labels,
        SPACING_MM,
        transform_points=rigid_then_stretch,
        patient_id="anisotropic-stretch",
        sampling=_sampling(),
    )

    assert set(rigid_result["patient"]["distance_bin"]) == {"short", "long"}
    assert set(stretch_result["patient"]["distance_bin"]) == {"short", "long"}

    pairs = rigid_result["pairs"]
    p0 = pairs[["p0_d", "p0_h", "p0_w"]].to_numpy()
    p1 = pairs[["p1_d", "p1_h", "p1_w"]].to_numpy()
    expected_mm = np.linalg.norm((p1 - p0) * SPACING_MM, axis=1)
    assert np.allclose(pairs["original_distance_mm"], expected_mm, atol=1e-12)

    assert rigid_result["pairs"]["abs_distance_change_mm"].max() < 1e-10
    assert rigid_result["patient"]["mean_abs_relative_change_pct"].max() < 1e-10

    expected_change_pct = 100.0 * (STRETCH - 1.0)
    assert np.allclose(
        stretch_result["pairs"]["relative_change_pct"],
        expected_change_pct,
        atol=1e-10,
    )
    assert np.allclose(
        stretch_result["pairs"]["abs_distance_change_mm"],
        (STRETCH - 1.0) * stretch_result["pairs"]["original_distance_mm"],
        atol=1e-10,
    )
    assert np.allclose(
        stretch_result["patient"]["mean_abs_relative_change_pct"],
        expected_change_pct,
        atol=1e-10,
    )


def main() -> None:
    test_rigid_motion_and_stretch_use_physical_millimetres()
    print(
        "OK - anisotropic-spacing rigid motion preserves bone distances and "
        f"physical stretch changes them by {100.0 * (STRETCH - 1.0):.1f}%"
    )


if __name__ == "__main__":
    main()
