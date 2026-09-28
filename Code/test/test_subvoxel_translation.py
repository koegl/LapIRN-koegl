"""Subvoxel interpolation baselines on PSMAReg case 0006, timepoint 02.

Run directly:
    uv run python Code/test/test_subvoxel_translation.py

PET is warped linearly and the lesion mask with nearest-neighbour interpolation,
matching the evaluation path. The control linearly warps the already-masked PET
(``PET * mask``) once and sums it over the complete output image.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import utils  # noqa: E402


DEFAULT_DATASET_ROOT = Path(
    "/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset"
)
CASE_ID = "0006"
TIMEPOINT = "02"
SHIFT_AMOUNTS = (0.25, 0.49, 0.50, 0.51, 0.75)
# Case-specific interpolation baseline. The tolerance below allows small
# floating-point differences while still catching a changed resampling path.
EXPECTED_TLG_BIAS = {
    0.25: 0.01497690,
    0.49: 0.02935492,
    0.50: 0.02949403,
    0.51: 0.02755238,
    0.75: 0.01405737,
}
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_PET = OUTPUT_DIR / "PSMARegPSMA_0006_0001_02_subvoxel_pet.nii.gz"
OUTPUT_SEG = OUTPUT_DIR / "PSMARegPSMA_0006_0001_02_subvoxel_seg.nii.gz"


def _dataset_root() -> Path:
    return Path(os.environ.get("PSMAREG_DATASET_ROOT", DEFAULT_DATASET_ROOT))


def _case_path(root: Path, folder: str, modality: str) -> Path:
    return root / folder / f"PSMARegPSMA_{CASE_ID}_{modality}_{TIMEPOINT}.nii.gz"


def _translation_grid(
    shape: tuple[int, int, int],
    shift_voxel: np.ndarray,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build an align_corners=False sampling grid for a positive content shift."""
    if shift_voxel.shape != (3,):
        raise ValueError("shift_voxel must have three components in (D,H,W) order")
    depth, height, width = shape
    shift = torch.as_tensor(shift_voxel, dtype=dtype)

    # grid_sample maps output positions back to source positions. Sampling at
    # output - shift moves image content in the positive direction.
    z = 2.0 * (torch.arange(depth, dtype=dtype) + 0.5 - shift[0]) / depth - 1.0
    y = 2.0 * (torch.arange(height, dtype=dtype) + 0.5 - shift[1]) / height - 1.0
    x = 2.0 * (torch.arange(width, dtype=dtype) + 0.5 - shift[2]) / width - 1.0
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    return torch.stack([xx, yy, zz], dim=-1).unsqueeze(0)


def _resample(volume: torch.Tensor, grid: torch.Tensor, mode: str) -> torch.Tensor:
    return F.grid_sample(
        volume,
        grid,
        mode=mode,
        padding_mode="zeros",
        align_corners=False,
    )


def _weighted_centroid(mask: np.ndarray) -> np.ndarray:
    total = float(mask.sum(dtype=np.float64))
    if total <= 0.0:
        raise ValueError("cannot measure the centroid of an empty mask")
    return np.array(
        [
            np.sum(mask, axis=tuple(j for j in range(3) if j != axis), dtype=np.float64)
            @ np.arange(mask.shape[axis], dtype=np.float64)
            / total
            for axis in range(3)
        ]
    )


def evaluate_subvoxel_translations() -> list[dict[str, float]]:
    root = _dataset_root()
    pet_path = _case_path(root, "imagesTr", "0001")
    label_path = _case_path(root, "labelsTr", "0001")
    missing = [str(path) for path in (pet_path, label_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing subvoxel-test input(s): " + ", ".join(missing))

    pet_nii = nib.load(pet_path)
    label_nii = nib.load(label_path)
    pet = np.asarray(pet_nii.dataobj, dtype=np.float32)
    lesion_mask = np.asarray(label_nii.dataobj, dtype=np.int16) > 0
    if pet.shape != lesion_mask.shape:
        raise ValueError("PET and PET labels must use the same voxel grid")

    lesion_coords = np.argwhere(lesion_mask)
    max_shift = np.array([max(SHIFT_AMOUNTS), 0.0, 0.0])
    shifted_bounds = lesion_coords.astype(np.float64) + max_shift
    if np.any(shifted_bounds < 0.0) or np.any(
        shifted_bounds > np.asarray(pet.shape, dtype=np.float64) - 1.0
    ):
        raise ValueError("configured subvoxel translations would clip the lesion")

    original_pet = torch.from_numpy(pet)[None, None]
    original_mask = torch.from_numpy(lesion_mask.astype(np.float32))[None, None]
    original_product = original_pet * original_mask
    original_mtv = float(original_mask.sum().item())
    original_tlg = float(original_product.sum().item())
    original_centroid = _weighted_centroid(lesion_mask.astype(np.float32))
    spacing_d_mm = float(label_nii.header.get_zooms()[0])

    rows = []
    for amount in SHIFT_AMOUNTS:
        shift = np.array([amount, 0.0, 0.0], dtype=np.float64)
        grid = _translation_grid(pet.shape, shift, original_pet.dtype)

        # One linear resampling call gives the PET, soft mask used for movement
        # verification, and the pre-multiplied control image.
        linear_inputs = torch.cat([original_pet, original_mask, original_product], dim=1)
        linear_outputs = _resample(linear_inputs, grid, mode="bilinear")
        translated_pet = linear_outputs[:, 0:1]
        translated_soft_mask = linear_outputs[:, 1:2]
        translated_product_control = linear_outputs[:, 2:3]
        translated_nearest_mask = _resample(original_mask, grid, mode="nearest")

        mtv_bias = float(
            utils.mtv_bias_loss(translated_nearest_mask, original_mask).item()
        )
        tlg_bias = float(
            utils.tlg_bias_loss(
                translated_pet,
                translated_nearest_mask,
                original_pet,
                original_mask,
            ).item()
        )
        translated_tlg = float(
            (translated_pet * translated_nearest_mask).sum().item()
        )

        control_tlg = float(translated_product_control.sum().item())
        control_tlg_bias = abs(control_tlg - original_tlg) / original_tlg
        translated_mtv = float(translated_nearest_mask.sum().item())

        soft_mask_np = translated_soft_mask[0, 0].cpu().numpy()
        nearest_mask_np = translated_nearest_mask[0, 0].cpu().numpy()
        soft_shift_d = float(_weighted_centroid(soft_mask_np)[0] - original_centroid[0])
        nearest_shift_d = float(
            _weighted_centroid(nearest_mask_np)[0] - original_centroid[0]
        )

        if amount == 0.50:
            nib.save(
                nib.Nifti1Image(
                    translated_pet[0, 0].cpu().numpy(),
                    pet_nii.affine,
                    pet_nii.header.copy(),
                ),
                OUTPUT_PET,
            )
            nib.save(
                nib.Nifti1Image(
                    nearest_mask_np.astype(np.int16),
                    label_nii.affine,
                    label_nii.header.copy(),
                ),
                OUTPUT_SEG,
            )

        rows.append(
            {
                "requested_shift_voxel": amount,
                "soft_centroid_shift_voxel": soft_shift_d,
                "nearest_centroid_shift_voxel": nearest_shift_d,
                "movement_distance_mm": soft_shift_d * spacing_d_mm,
                "original_mtv_voxels": original_mtv,
                "translated_mtv_voxels": translated_mtv,
                "mtv_bias": mtv_bias,
                "original_tlg": original_tlg,
                "translated_tlg": translated_tlg,
                "tlg_bias": tlg_bias,
                "control_tlg": control_tlg,
                "control_tlg_bias": control_tlg_bias,
            }
        )
    return rows


def _assert_subvoxel_metrics(rows: list[dict[str, float]]) -> None:
    assert [row["requested_shift_voxel"] for row in rows] == list(SHIFT_AMOUNTS)
    for row in rows:
        for key in ("mtv_bias", "tlg_bias", "control_tlg_bias"):
            assert np.isfinite(row[key])
            assert row[key] >= 0.0
        assert np.isclose(
            row["soft_centroid_shift_voxel"],
            row["requested_shift_voxel"],
            atol=1e-5,
        )
        assert row["control_tlg_bias"] < 1e-6
        assert np.isclose(
            row["tlg_bias"],
            EXPECTED_TLG_BIAS[row["requested_shift_voxel"]],
            atol=5e-5,
        )

    below = next(row for row in rows if row["requested_shift_voxel"] == 0.49)
    above = next(row for row in rows if row["requested_shift_voxel"] == 0.51)
    assert np.isclose(below["nearest_centroid_shift_voxel"], 0.0, atol=1e-12)
    assert np.isclose(above["nearest_centroid_shift_voxel"], 1.0, atol=1e-12)
    assert below["mtv_bias"] < 1e-6
    assert above["mtv_bias"] < 1e-6
    assert OUTPUT_PET.is_file()
    assert OUTPUT_SEG.is_file()


def test_subvoxel_translation_real_case() -> None:
    _assert_subvoxel_metrics(evaluate_subvoxel_translations())


def main() -> None:
    rows = evaluate_subvoxel_translations()
    _assert_subvoxel_metrics(rows)
    print(f"Subvoxel translation: PSMARegPSMA_{CASE_ID}, timepoint {TIMEPOINT}")
    print("shift  nearest_shift  MTV_bias_%  TLG_bias_%  control_TLG_bias_%")
    for row in rows:
        print(
            f"{row['requested_shift_voxel']:>5.2f}"
            f"  {row['nearest_centroid_shift_voxel']:>13.6f}"
            f"  {100.0 * row['mtv_bias']:>10.6f}"
            f"  {100.0 * row['tlg_bias']:>10.6f}"
            f"  {100.0 * row['control_tlg_bias']:>18.9f}"
        )
    print(f"Saved 0.50-voxel PET: {OUTPUT_PET}")
    print(f"Saved 0.50-voxel segmentation: {OUTPUT_SEG}")
    print("OK - subvoxel interpolation sweep and pre-multiplied control measured")


if __name__ == "__main__":
    main()
