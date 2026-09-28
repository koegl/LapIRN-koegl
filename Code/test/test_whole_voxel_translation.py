"""Whole-voxel translation check on PSMAReg case 0006, timepoint 02.

Run directly:
    uv run python Code/test/test_whole_voxel_translation.py

The translated PET and lesion segmentation are saved beside this file for
visual inspection. The dataset root can be overridden with
``PSMAREG_DATASET_ROOT``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jacobian  # noqa: E402
import synthetic  # noqa: E402
import utils  # noqa: E402
from bone_deformation_metric import (  # noqa: E402
    PairSamplingConfig,
    measure_bone_deformation,
)


DEFAULT_DATASET_ROOT = Path(
    "/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset"
)
CASE_ID = "0006"
TIMEPOINT = "02"
TRANSLATION_VOXEL = np.array([5, -7, 4], dtype=np.int64)
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_PET = OUTPUT_DIR / "PSMARegPSMA_0006_0001_02_translated_pet.nii.gz"
OUTPUT_SEG = OUTPUT_DIR / "PSMARegPSMA_0006_0001_02_translated_seg.nii.gz"


def _dataset_root() -> Path:
    return Path(os.environ.get("PSMAREG_DATASET_ROOT", DEFAULT_DATASET_ROOT))


def _case_path(root: Path, folder: str, modality: str) -> Path:
    return root / folder / f"PSMARegPSMA_{CASE_ID}_{modality}_{TIMEPOINT}.nii.gz"


def _translate_whole_voxels(array: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """Translate an array with zero fill and no wraparound."""
    if array.ndim != 3 or shift.shape != (3,):
        raise ValueError("expected a 3D array and a three-component shift")

    source_slices = []
    target_slices = []
    for size, amount in zip(array.shape, shift):
        amount = int(amount)
        if abs(amount) >= size:
            raise ValueError("translation must be smaller than the image dimensions")
        if amount >= 0:
            source_slices.append(slice(0, size - amount))
            target_slices.append(slice(amount, size))
        else:
            source_slices.append(slice(-amount, size))
            target_slices.append(slice(0, size + amount))

    translated = np.zeros_like(array)
    translated[tuple(target_slices)] = array[tuple(source_slices)]
    return translated


def evaluate_whole_voxel_translation() -> dict[str, float | str]:
    root = _dataset_root()
    paths = {
        "pet": _case_path(root, "imagesTr", "0001"),
        "ct_labels": _case_path(root, "labelsTr", "0000"),
        "pet_labels": _case_path(root, "labelsTr", "0001"),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing translation-test input(s): " + ", ".join(missing))

    pet_nii = nib.load(paths["pet"])
    ct_label_nii = nib.load(paths["ct_labels"])
    pet_label_nii = nib.load(paths["pet_labels"])
    pet = np.asarray(pet_nii.dataobj, dtype=np.float32)
    ct_labels = np.asarray(ct_label_nii.dataobj, dtype=np.int16)
    pet_labels = np.asarray(pet_label_nii.dataobj, dtype=np.int16)
    if pet.shape != ct_labels.shape or pet.shape != pet_labels.shape:
        raise ValueError("PET, CT labels, and PET labels must use the same voxel grid")

    lesion_mask = pet_labels > 0
    lesion_coords = np.argwhere(lesion_mask)
    if lesion_coords.size == 0:
        raise ValueError("the selected PET segmentation contains no lesion")
    translated_coords = lesion_coords + TRANSLATION_VOXEL
    if np.any(translated_coords < 0) or np.any(
        translated_coords >= np.asarray(pet.shape)
    ):
        raise ValueError("configured translation would clip the lesion")

    translated_pet = _translate_whole_voxels(pet, TRANSLATION_VOXEL)
    translated_labels = _translate_whole_voxels(pet_labels, TRANSLATION_VOXEL)
    translated_mask = translated_labels > 0
    if int(translated_mask.sum()) != int(lesion_mask.sum()):
        raise AssertionError("translation clipped lesion voxels")

    nib.save(
        nib.Nifti1Image(translated_pet, pet_nii.affine, pet_nii.header.copy()),
        OUTPUT_PET,
    )
    nib.save(
        nib.Nifti1Image(translated_labels, pet_label_nii.affine, pet_label_nii.header.copy()),
        OUTPUT_SEG,
    )

    original_pet_t = torch.from_numpy(pet)[None, None]
    translated_pet_t = torch.from_numpy(translated_pet)[None, None]
    original_mask_t = torch.from_numpy(lesion_mask.astype(np.float32))[None, None]
    translated_mask_t = torch.from_numpy(translated_mask.astype(np.float32))[None, None]
    mtv_bias = float(utils.mtv_bias_loss(translated_mask_t, original_mask_t).item())
    tlg_bias = float(
        utils.tlg_bias_loss(
            translated_pet_t,
            translated_mask_t,
            original_pet_t,
            original_mask_t,
        ).item()
    )

    spacing_mm = np.asarray(ct_label_nii.header.get_zooms()[:3], dtype=np.float64)
    shift_float = TRANSLATION_VOXEL.astype(np.float64)
    bone_result = measure_bone_deformation(
        ct_labels,
        spacing_mm,
        transform_points=lambda points: points + shift_float,
        bone_labels=synthetic.BONE_LABEL_VALUES,
        patient_id=f"{CASE_ID}_{TIMEPOINT}",
        sampling=PairSamplingConfig(
            pairs_per_bin=64,
            candidate_multiplier=20,
            seed=6,
        ),
    )
    if bone_result["pairs"].empty:
        raise AssertionError("no bone point pairs were sampled")
    bone_max_abs_change_mm = float(
        bone_result["pairs"]["abs_distance_change_mm"].max()
    )

    displacement = torch.as_tensor(shift_float, dtype=torch.float32).view(1, 1, 1, 1, 3)
    displacement = displacement.expand(1, *ct_labels.shape, 3)
    folding_pct = jacobian.percent_ndv(displacement)

    original_centroid = lesion_coords.mean(axis=0)
    translated_centroid = np.argwhere(translated_mask).mean(axis=0)
    measured_shift_voxel = translated_centroid - original_centroid
    measured_distance_mm = float(np.linalg.norm(measured_shift_voxel * spacing_mm))
    expected_distance_mm = float(np.linalg.norm(shift_float * spacing_mm))

    return {
        "mtv_bias": mtv_bias,
        "tlg_bias": tlg_bias,
        "bone_max_abs_change_mm": bone_max_abs_change_mm,
        "folding_pct": folding_pct,
        "shift_d_voxel": float(measured_shift_voxel[0]),
        "shift_h_voxel": float(measured_shift_voxel[1]),
        "shift_w_voxel": float(measured_shift_voxel[2]),
        "movement_distance_mm": measured_distance_mm,
        "expected_distance_mm": expected_distance_mm,
        "output_pet": str(OUTPUT_PET),
        "output_seg": str(OUTPUT_SEG),
    }


def _assert_translation_metrics(metrics: dict[str, float | str]) -> None:
    assert float(metrics["mtv_bias"]) == 0.0
    assert float(metrics["tlg_bias"]) < 1e-6
    assert float(metrics["bone_max_abs_change_mm"]) < 1e-10
    assert float(metrics["folding_pct"]) == 0.0
    measured_shift = np.array(
        [
            metrics["shift_d_voxel"],
            metrics["shift_h_voxel"],
            metrics["shift_w_voxel"],
        ],
        dtype=np.float64,
    )
    assert np.allclose(measured_shift, TRANSLATION_VOXEL, atol=1e-12)
    assert np.isclose(
        float(metrics["movement_distance_mm"]),
        float(metrics["expected_distance_mm"]),
        atol=1e-12,
    )
    assert Path(str(metrics["output_pet"])).is_file()
    assert Path(str(metrics["output_seg"])).is_file()


def test_whole_voxel_translation_real_case() -> None:
    _assert_translation_metrics(evaluate_whole_voxel_translation())


def main() -> None:
    metrics = evaluate_whole_voxel_translation()
    _assert_translation_metrics(metrics)
    print(f"Whole-voxel translation: PSMARegPSMA_{CASE_ID}, timepoint {TIMEPOINT}")
    for name, value in metrics.items():
        print(f"  {name}: {value}")
    print("OK - translation preserves MTV/TLG and bone distances, with no folding")


if __name__ == "__main__":
    main()
