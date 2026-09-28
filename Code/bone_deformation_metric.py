"""Distance-preservation metric for bone deformation fields.

For each labelled bone, this module samples short and long point pairs inside
the bone mask, transforms both points, and measures how much their physical
separation changed. Patient-level summaries average over bones first so large
bones do not dominate the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import pandas as pd


PointTransform = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class PairSamplingConfig:
    """Controls point-pair sampling within each bone."""

    pairs_per_bin: int = 256
    candidate_multiplier: int = 30
    short_quantile: float = 0.33
    long_quantile: float = 0.67
    min_voxels: int = 2
    seed: int = 0


def _normalise_displacement(displacement: np.ndarray) -> np.ndarray:
    """Return a displacement array as ``(D, H, W, 3)`` in voxel units."""
    disp = np.asarray(displacement, dtype=np.float64)
    if disp.ndim == 5 and disp.shape[0] == 1 and disp.shape[-1] == 3:
        disp = disp[0]
    elif disp.ndim == 5 and disp.shape[0] == 1 and disp.shape[1] == 3:
        disp = np.moveaxis(disp[0], 0, -1)
    elif disp.ndim == 4 and disp.shape[0] == 3:
        disp = np.moveaxis(disp, 0, -1)

    if disp.ndim != 4 or disp.shape[-1] != 3:
        raise ValueError(
            "displacement must have shape (D,H,W,3), (1,D,H,W,3), "
            "(3,D,H,W), or (1,3,D,H,W)"
        )
    return disp


def _sample_displacement_trilinear(
    displacement: np.ndarray,
    points_voxel: np.ndarray,
) -> np.ndarray:
    """Trilinearly sample ``displacement`` at ``points_voxel``."""
    disp = _normalise_displacement(displacement)
    shape = np.asarray(disp.shape[:3], dtype=np.int64)
    pts = np.asarray(points_voxel, dtype=np.float64)
    clipped = np.clip(pts, 0.0, shape.astype(np.float64) - 1.0)

    lo = np.floor(clipped).astype(np.int64)
    hi = np.minimum(lo + 1, shape - 1)
    frac = clipped - lo

    out = np.zeros((pts.shape[0], 3), dtype=np.float64)
    for dz in (0, 1):
        wz = frac[:, 0] if dz else 1.0 - frac[:, 0]
        iz = hi[:, 0] if dz else lo[:, 0]
        for dy in (0, 1):
            wy = frac[:, 1] if dy else 1.0 - frac[:, 1]
            iy = hi[:, 1] if dy else lo[:, 1]
            for dx in (0, 1):
                wx = frac[:, 2] if dx else 1.0 - frac[:, 2]
                ix = hi[:, 2] if dx else lo[:, 2]
                out += (wz * wy * wx)[:, None] * disp[iz, iy, ix]
    return out


def displacement_to_point_transform(displacement: np.ndarray) -> PointTransform:
    """Build a point transform from a dense voxel-unit displacement field."""
    disp = _normalise_displacement(displacement)

    def transform(points_voxel: np.ndarray) -> np.ndarray:
        points = np.asarray(points_voxel, dtype=np.float64)
        return points + _sample_displacement_trilinear(disp, points)

    return transform


def sample_bone_point_pairs(
    mask: np.ndarray,
    spacing_mm: Iterable[float],
    config: PairSamplingConfig | None = None,
) -> pd.DataFrame:
    """Sample short and long voxel-centre point pairs from one binary bone mask.

    The short and long groups are selected from the lower and upper distance
    quantiles of randomly drawn candidate pairs. Distances are measured in mm.
    """
    cfg = config or PairSamplingConfig()
    coords = np.argwhere(np.asarray(mask).astype(bool)).astype(np.float64)
    if coords.shape[0] < cfg.min_voxels:
        return pd.DataFrame()

    rng = np.random.default_rng(cfg.seed)
    n_candidates = max(cfg.pairs_per_bin * cfg.candidate_multiplier, 128)
    first = rng.integers(0, coords.shape[0], size=n_candidates)
    second = rng.integers(0, coords.shape[0], size=n_candidates)
    keep = first != second
    first = first[keep]
    second = second[keep]
    if first.size == 0:
        return pd.DataFrame()

    p0 = coords[first]
    p1 = coords[second]
    spacing = np.asarray(tuple(spacing_mm), dtype=np.float64)
    if spacing.shape != (3,):
        raise ValueError("spacing_mm must contain three values in (D,H,W) order")

    original = np.linalg.norm((p1 - p0) * spacing, axis=1)
    nonzero = original > 0
    p0 = p0[nonzero]
    p1 = p1[nonzero]
    original = original[nonzero]
    if original.size == 0:
        return pd.DataFrame()

    short_cut = np.quantile(original, cfg.short_quantile)
    long_cut = np.quantile(original, cfg.long_quantile)
    groups = {
        "short": np.flatnonzero(original <= short_cut),
        "long": np.flatnonzero(original >= long_cut),
    }

    rows = []
    for distance_bin, idx in groups.items():
        if idx.size == 0:
            continue
        take = rng.choice(idx, size=min(cfg.pairs_per_bin, idx.size), replace=False)
        for i in take:
            rows.append(
                {
                    "distance_bin": distance_bin,
                    "p0_d": p0[i, 0],
                    "p0_h": p0[i, 1],
                    "p0_w": p0[i, 2],
                    "p1_d": p1[i, 0],
                    "p1_h": p1[i, 1],
                    "p1_w": p1[i, 2],
                    "original_distance_mm": original[i],
                }
            )
    return pd.DataFrame(rows)


def measure_pair_distance_changes(
    pairs: pd.DataFrame,
    transform_points: PointTransform,
    spacing_mm: Iterable[float],
) -> pd.DataFrame:
    """Measure distance changes for sampled pairs after applying a transform."""
    if pairs.empty:
        return pairs.copy()

    p0 = pairs[["p0_d", "p0_h", "p0_w"]].to_numpy(dtype=np.float64)
    p1 = pairs[["p1_d", "p1_h", "p1_w"]].to_numpy(dtype=np.float64)
    t0 = np.asarray(transform_points(p0), dtype=np.float64)
    t1 = np.asarray(transform_points(p1), dtype=np.float64)
    if t0.shape != p0.shape or t1.shape != p1.shape:
        raise ValueError("transform_points must return an array with shape (N, 3)")

    spacing = np.asarray(tuple(spacing_mm), dtype=np.float64)
    new_distance = np.linalg.norm((t1 - t0) * spacing, axis=1)
    result = pairs.copy()
    result["new_distance_mm"] = new_distance
    result["signed_distance_change_mm"] = (
        result["new_distance_mm"] - result["original_distance_mm"]
    )
    result["abs_distance_change_mm"] = result["signed_distance_change_mm"].abs()
    result["relative_change_pct"] = (
        100.0
        * result["signed_distance_change_mm"]
        / result["original_distance_mm"]
    )
    result["abs_relative_change_pct"] = result["relative_change_pct"].abs()
    return result


def summarize_bone_distance_changes(pair_results: pd.DataFrame) -> pd.DataFrame:
    """Summarise measured pair errors per bone and distance bin."""
    if pair_results.empty:
        return pd.DataFrame()

    group_cols = ["patient_id", "bone_label", "distance_bin"]
    return (
        pair_results.groupby(group_cols, dropna=False)
        .agg(
            n_pairs=("original_distance_mm", "size"),
            mean_original_distance_mm=("original_distance_mm", "mean"),
            mean_new_distance_mm=("new_distance_mm", "mean"),
            mean_signed_distance_change_mm=("signed_distance_change_mm", "mean"),
            mean_abs_distance_change_mm=("abs_distance_change_mm", "mean"),
            max_abs_distance_change_mm=("abs_distance_change_mm", "max"),
            mean_relative_change_pct=("relative_change_pct", "mean"),
            mean_abs_relative_change_pct=("abs_relative_change_pct", "mean"),
            max_abs_relative_change_pct=("abs_relative_change_pct", "max"),
        )
        .reset_index()
    )


def summarize_patient_distance_changes(bone_summary: pd.DataFrame) -> pd.DataFrame:
    """Average per-bone summaries per patient, keeping short and long separate."""
    if bone_summary.empty:
        return pd.DataFrame()

    group_cols = ["patient_id", "distance_bin"]
    return (
        bone_summary.groupby(group_cols, dropna=False)
        .agg(
            n_bones=("bone_label", "nunique"),
            total_pairs=("n_pairs", "sum"),
            mean_original_distance_mm=("mean_original_distance_mm", "mean"),
            mean_abs_distance_change_mm=("mean_abs_distance_change_mm", "mean"),
            max_abs_distance_change_mm=("max_abs_distance_change_mm", "max"),
            mean_relative_change_pct=("mean_relative_change_pct", "mean"),
            mean_abs_relative_change_pct=("mean_abs_relative_change_pct", "mean"),
            max_abs_relative_change_pct=("max_abs_relative_change_pct", "max"),
        )
        .reset_index()
    )


def measure_bone_deformation(
    labels: np.ndarray,
    spacing_mm: Iterable[float],
    transform_points: PointTransform | None = None,
    displacement_voxel: np.ndarray | None = None,
    bone_labels: Iterable[int] | None = None,
    patient_id: str = "patient",
    sampling: PairSamplingConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Measure bone distance changes for one patient.

    Exactly one of ``transform_points`` or ``displacement_voxel`` must be
    supplied. ``transform_points`` receives and returns ``(N, 3)`` points in
    voxel coordinates, ordered as ``(D, H, W)``.
    """
    if (transform_points is None) == (displacement_voxel is None):
        raise ValueError("provide exactly one of transform_points or displacement_voxel")
    if transform_points is None:
        transform_points = displacement_to_point_transform(displacement_voxel)

    labels_np = np.asarray(labels)
    if bone_labels is None:
        values = [int(v) for v in np.unique(labels_np) if int(v) != 0]
    else:
        values = [int(v) for v in bone_labels]

    cfg = sampling or PairSamplingConfig()
    all_results = []
    for offset, label_value in enumerate(values):
        mask = labels_np == label_value
        if int(mask.sum()) < cfg.min_voxels:
            continue
        pair_cfg = PairSamplingConfig(
            pairs_per_bin=cfg.pairs_per_bin,
            candidate_multiplier=cfg.candidate_multiplier,
            short_quantile=cfg.short_quantile,
            long_quantile=cfg.long_quantile,
            min_voxels=cfg.min_voxels,
            seed=cfg.seed + offset,
        )
        pairs = sample_bone_point_pairs(mask, spacing_mm, pair_cfg)
        if pairs.empty:
            continue
        measured = measure_pair_distance_changes(pairs, transform_points, spacing_mm)
        measured.insert(0, "bone_label", label_value)
        measured.insert(0, "patient_id", patient_id)
        all_results.append(measured)

    if all_results:
        pair_results = pd.concat(all_results, ignore_index=True)
    else:
        pair_results = pd.DataFrame()

    bone_summary = summarize_bone_distance_changes(pair_results)
    patient_summary = summarize_patient_distance_changes(bone_summary)
    return {
        "pairs": pair_results,
        "bones": bone_summary,
        "patient": patient_summary,
    }
