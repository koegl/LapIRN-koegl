"""Tests for the sampled bone distance-preservation metric.

Run directly:
    python Code/test_bone_deformation_metric.py

or with pytest:
    pytest Code/test_bone_deformation_metric.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bone_deformation_metric import (  # noqa: E402
    PairSamplingConfig,
    measure_bone_deformation,
)


def _two_bone_labels() -> np.ndarray:
    labels = np.zeros((24, 26, 28), dtype=np.int16)
    labels[2:8, 3:10, 4:12] = 1
    labels[12:22, 10:24, 9:25] = 2
    return labels


def _sampling() -> PairSamplingConfig:
    return PairSamplingConfig(pairs_per_bin=64, candidate_multiplier=20, seed=11)


def test_rigid_transform_preserves_all_sampled_distances() -> None:
    labels = _two_bone_labels()
    spacing_mm = (1.0, 1.0, 1.0)
    centre = np.array([(s - 1.0) / 2.0 for s in labels.shape])
    angle = np.deg2rad(35.0)
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = np.array([3.2, -1.7, 0.6])

    def rigid(points: np.ndarray) -> np.ndarray:
        return (points - centre) @ rotation.T + centre + translation

    result = measure_bone_deformation(
        labels,
        spacing_mm,
        transform_points=rigid,
        patient_id="rigid",
        sampling=_sampling(),
    )

    assert set(result["patient"]["distance_bin"]) == {"short", "long"}
    assert result["pairs"]["abs_distance_change_mm"].max() < 1e-10
    assert result["bones"]["mean_abs_relative_change_pct"].max() < 1e-10
    assert result["patient"]["mean_abs_relative_change_pct"].max() < 1e-10


def test_known_uniform_stretch_reports_expected_relative_change() -> None:
    labels = _two_bone_labels()
    spacing_mm = (1.0, 1.0, 1.0)
    centre = np.array([(s - 1.0) / 2.0 for s in labels.shape])
    scale = 1.12

    def stretch(points: np.ndarray) -> np.ndarray:
        return centre + scale * (points - centre)

    result = measure_bone_deformation(
        labels,
        spacing_mm,
        transform_points=stretch,
        patient_id="stretch",
        sampling=_sampling(),
    )

    assert np.allclose(result["pairs"]["relative_change_pct"], 12.0, atol=1e-10)
    assert np.allclose(
        result["pairs"]["abs_distance_change_mm"],
        0.12 * result["pairs"]["original_distance_mm"],
        atol=1e-10,
    )
    assert np.allclose(
        result["patient"]["mean_abs_relative_change_pct"], 12.0, atol=1e-10
    )


def test_dense_displacement_translation_preserves_distances() -> None:
    labels = _two_bone_labels()
    displacement = np.zeros(labels.shape + (3,), dtype=np.float64)
    displacement[..., 0] = 2.5
    displacement[..., 1] = -4.0
    displacement[..., 2] = 1.25

    result = measure_bone_deformation(
        labels,
        spacing_mm=(1.0, 1.0, 1.0),
        displacement_voxel=displacement,
        patient_id="translation",
        sampling=_sampling(),
    )

    assert result["pairs"]["abs_distance_change_mm"].max() < 1e-10
    assert result["patient"]["mean_abs_relative_change_pct"].max() < 1e-10


def main() -> None:
    test_rigid_transform_preserves_all_sampled_distances()
    test_known_uniform_stretch_reports_expected_relative_change()
    test_dense_displacement_translation_preserves_distances()
    print("OK - bone deformation metric preserves rigid distances and detects stretch")


if __name__ == "__main__":
    main()
