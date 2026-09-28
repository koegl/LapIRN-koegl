"""End-to-end identity-transform check on PSMAReg case 0006, timepoint 02.

Run directly:
    uv run python Code/test/test_identity_transform.py

The dataset location can be overridden with ``PSMAREG_DATASET_ROOT``.
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


def _dataset_root() -> Path:
    return Path(os.environ.get("PSMAREG_DATASET_ROOT", DEFAULT_DATASET_ROOT))


def _case_path(root: Path, folder: str, modality: str) -> Path:
    return root / folder / f"PSMARegPSMA_{CASE_ID}_{modality}_{TIMEPOINT}.nii.gz"


def _load_identity_case() -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, ...]]:
    root = _dataset_root()
    paths = {
        "pet": _case_path(root, "imagesTr", "0001"),
        "ct_labels": _case_path(root, "labelsTr", "0000"),
        "pet_labels": _case_path(root, "labelsTr", "0001"),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing identity-test input(s): " + ", ".join(missing))

    pet_nii = nib.load(paths["pet"])
    ct_label_nii = nib.load(paths["ct_labels"])
    pet_label_nii = nib.load(paths["pet_labels"])

    pet = np.asarray(pet_nii.dataobj, dtype=np.float32)
    ct_labels = np.asarray(ct_label_nii.dataobj, dtype=np.int16)
    pet_labels = np.asarray(pet_label_nii.dataobj, dtype=np.int16)
    if pet.shape != ct_labels.shape or pet.shape != pet_labels.shape:
        raise ValueError(
            "PET, CT labels, and PET labels must have the same voxel grid; got "
            f"{pet.shape}, {ct_labels.shape}, and {pet_labels.shape}"
        )

    spacing_mm = tuple(float(value) for value in ct_label_nii.header.get_zooms()[:3])
    return pet, ct_labels, pet_labels, spacing_mm


def evaluate_identity_transform_real_case() -> dict[str, float]:
    """Evaluate all identity-transform metrics on the configured real case."""
    pet, ct_labels, pet_labels, spacing_mm = _load_identity_case()

    pet_image = torch.from_numpy(pet)[None, None]
    lesion_mask = torch.from_numpy((pet_labels > 0).astype(np.float32))[None, None]
    ct_label_tensor = torch.from_numpy(ct_labels.astype(np.float32))[None, None]

    # Under identity, warped inputs are the original arrays themselves.
    mtv_bias = float(utils.mtv_bias_loss(lesion_mask, lesion_mask).item())
    tlg_bias = float(
        utils.tlg_bias_loss(pet_image, lesion_mask, pet_image, lesion_mask).item()
    )
    dice = float(1.0 - utils.soft_dice_loss_binary(lesion_mask, lesion_mask).item())
    hd95_mm = utils.hd95_ct_labels(
        ct_label_tensor,
        ct_label_tensor,
        ct_label_tensor,
        spacing_mm,
    )

    bone_result = measure_bone_deformation(
        ct_labels,
        spacing_mm,
        transform_points=lambda points: points.copy(),
        bone_labels=synthetic.BONE_LABEL_VALUES,
        patient_id=f"{CASE_ID}_{TIMEPOINT}",
        sampling=PairSamplingConfig(
            pairs_per_bin=64,
            candidate_multiplier=20,
            seed=6,
        ),
    )
    assert not bone_result["pairs"].empty, "no bone point pairs were sampled"
    bone_max_abs_change_mm = float(
        bone_result["pairs"]["abs_distance_change_mm"].max()
    )
    bone_max_abs_relative_change_pct = float(
        bone_result["pairs"]["abs_relative_change_pct"].max()
    )

    zero_displacement = torch.zeros(
        (1, *ct_labels.shape, 3),
        dtype=torch.float32,
    )
    folding_pct = jacobian.percent_ndv(zero_displacement)

    return {
        "mtv_bias": mtv_bias,
        "tlg_bias": tlg_bias,
        "dice": dice,
        "hd95_mm": hd95_mm,
        "bone_max_abs_change_mm": bone_max_abs_change_mm,
        "bone_max_abs_relative_change_pct": bone_max_abs_relative_change_pct,
        "folding_pct": folding_pct,
    }


def _assert_identity_metrics(metrics: dict[str, float]) -> None:
    assert metrics["mtv_bias"] == 0.0
    assert metrics["tlg_bias"] == 0.0
    assert metrics["dice"] == 1.0
    assert metrics["hd95_mm"] == 0.0
    assert metrics["bone_max_abs_change_mm"] == 0.0
    assert metrics["bone_max_abs_relative_change_pct"] == 0.0
    assert metrics["folding_pct"] == 0.0


def test_identity_transform_real_case() -> None:
    """Identity must preserve tumour, anatomy, and deformation metrics."""
    _assert_identity_metrics(evaluate_identity_transform_real_case())


def main() -> None:
    metrics = evaluate_identity_transform_real_case()
    _assert_identity_metrics(metrics)
    print(f"Identity transform: PSMARegPSMA_{CASE_ID}, timepoint {TIMEPOINT}")
    for name, value in metrics.items():
        print(f"  {name}: {value:.12g}")
    print("OK - identity preserves MTV/TLG, bone distances, Dice, and HD95; no folding")


if __name__ == "__main__":
    main()
