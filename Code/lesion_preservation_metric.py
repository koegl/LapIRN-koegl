"""Evaluation helpers for lesion-level MTV/TLG preservation.

The training losses in ``utils`` use smoothed denominators so optimization can
continue on unusual batches. Evaluation needs explicit statuses instead: an
excluded lesion must not silently contribute a zero or an infinite value.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np


PointTransform = Callable[[np.ndarray], np.ndarray]


def transformed_in_fov_fraction(
    source_mask: np.ndarray,
    transform_points: PointTransform,
    output_shape: Iterable[int],
) -> float | None:
    """Return the fraction of source voxel centres remaining in the output FOV.

    ``None`` is returned for an empty source mask because no retention fraction
    can be defined. Points and dimensions use array order ``(D, H, W)``.
    """
    mask = np.asarray(source_mask, dtype=bool)
    shape = np.asarray(tuple(output_shape), dtype=np.int64)
    if mask.ndim != 3 or shape.shape != (3,) or np.any(shape <= 0):
        raise ValueError("expected a 3D mask and three positive output dimensions")

    points = np.argwhere(mask).astype(np.float64)
    if points.size == 0:
        return None
    transformed = np.asarray(transform_points(points), dtype=np.float64)
    if transformed.shape != points.shape or not np.all(np.isfinite(transformed)):
        raise ValueError("transform_points must return finite points with shape (N, 3)")

    inside = np.all((transformed >= 0.0) & (transformed <= shape - 1.0), axis=1)
    return float(inside.mean())


def evaluate_lesion_preservation(
    source_mask: np.ndarray,
    warped_mask: np.ndarray,
    source_pet: np.ndarray,
    warped_pet: np.ndarray,
    voxel_volume_ml: float,
    *,
    min_source_voxels: int = 8,
    in_fov_fraction: float | None = 1.0,
    fov_tolerance: float = 1e-12,
) -> dict[str, object]:
    """Compute lesion MTV/TLG errors and an explicit inclusion status.

    Rules, applied in order:

    * an empty source lesion is excluded and its errors are ``None``;
    * a source lesion smaller than ``min_source_voxels`` is reported but
      excluded from the primary summary;
    * a lesion with any source voxel centre outside the transformed field of
      view is reported but excluded from the primary summary;
    * otherwise the lesion is included.

    For nonempty lesions, observed errors follow the challenge definition and
    remain available even when the lesion is excluded. This keeps clipping and
    small-lesion behaviour visible without treating it as a valid primary
    measurement.
    """
    source = np.asarray(source_mask, dtype=bool)
    warped = np.asarray(warped_mask, dtype=bool)
    pet_before = np.asarray(source_pet, dtype=np.float64)
    pet_after = np.asarray(warped_pet, dtype=np.float64)
    if not (source.shape == warped.shape == pet_before.shape == pet_after.shape):
        raise ValueError("PET arrays and masks must have identical shapes")
    if source.ndim != 3:
        raise ValueError("PET arrays and masks must be 3D")
    if not np.all(np.isfinite(pet_before)) or not np.all(np.isfinite(pet_after)):
        raise ValueError("PET arrays must contain only finite values")
    if not np.isfinite(voxel_volume_ml) or voxel_volume_ml <= 0.0:
        raise ValueError("voxel_volume_ml must be positive and finite")
    if min_source_voxels < 1:
        raise ValueError("min_source_voxels must be at least one")
    if in_fov_fraction is not None and not 0.0 <= in_fov_fraction <= 1.0:
        raise ValueError("in_fov_fraction must lie in [0, 1] or be None")

    source_voxels = int(source.sum())
    warped_voxels = int(warped.sum())
    mtv_before_ml = source_voxels * float(voxel_volume_ml)
    mtv_after_ml = warped_voxels * float(voxel_volume_ml)
    tlg_before = float(pet_before[source].sum() * voxel_volume_ml)
    tlg_after = float(pet_after[warped].sum() * voxel_volume_ml)

    result: dict[str, object] = {
        "included": False,
        "status": "excluded_empty_source",
        "source_voxels": source_voxels,
        "warped_voxels": warped_voxels,
        "in_fov_fraction": in_fov_fraction,
        "mtv_before_ml": mtv_before_ml,
        "mtv_after_ml": mtv_after_ml,
        "tlg_before": tlg_before,
        "tlg_after": tlg_after,
        "mtv_percent_error": None,
        "tlg_percent_error": None,
    }
    if source_voxels == 0:
        return result

    result["mtv_percent_error"] = (
        100.0 * abs(mtv_after_ml - mtv_before_ml) / max(mtv_before_ml, 1e-6)
    )
    result["tlg_percent_error"] = (
        100.0 * abs(tlg_after - tlg_before) / max(tlg_before, 1e-6)
    )

    if source_voxels < min_source_voxels:
        result["status"] = "excluded_tiny_source"
    elif in_fov_fraction is None:
        result["status"] = "excluded_unknown_fov"
    elif in_fov_fraction < 1.0 - fov_tolerance:
        result["status"] = "excluded_out_of_fov"
    else:
        result["included"] = True
        result["status"] = "included"
    return result

