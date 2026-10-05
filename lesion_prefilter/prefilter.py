"""Quick pre-filter: how well does each lesion-mask arm reproduce the
ground-truth MTV and TLG, without running IO?

Arms
    threshold   PET SUV >= 3 minus dilated high-uptake organs (TotalSegmentator)

Per image the ground truth (labelsTr ``_0001_``) is compared with each arm:
MTV/TLG and their errors, missed GT volume, false-positive volume, missed
lesions, and how much GT volume lies inside each dilated organ (the cost of
subtracting that organ).

Run ``segment_organs.py`` first. Output: ``results.csv`` in this folder.
"""

import json
from pathlib import Path
from typing import Dict

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
from tqdm import tqdm

from segment_organs import DATA, HERE, image_path, label_path

SUV_THRESHOLD = 3.0
# TotalSegmentator label -> dilation in mm (compensates PET/CT misregistration)
ORGAN_DILATION_MM: Dict[int, float] = {
    1: 10,  # spleen
    2: 10,  # kidney_right
    3: 10,  # kidney_left
    4: 10,  # gallbladder
    5: 10,  # liver
    6: 10,  # stomach
    7: 10,  # pancreas
    18: 10,  # small_bowel
    19: 10,  # duodenum
    20: 10,  # colon
    21: 5,  # urinary_bladder: small, mainly filling changes during the scan
    64: 10,  # portal_vein_and_splenic_vein
}
# TotalSegmentator head_glands_cavities label -> dilation in mm
HEAD_DILATION_MM: Dict[int, float] = {
    1: 10,  # eye_left: proxy for the lacrimal gland (no own label)
    2: 10,  # eye_right
    7: 10,  # parotid_gland_left
    8: 10,  # parotid_gland_right
    9: 10,  # submandibular_gland_right
    10: 10,  # submandibular_gland_left
}


def dilate(mask: np.ndarray, mm: float, spacing: tuple) -> np.ndarray:
    if mm <= 0:
        return mask
    return ndimage.distance_transform_edt(~mask, sampling=spacing) <= mm


def organ_masks(
    labels: np.ndarray, dilation_mm: Dict[int, float], spacing: tuple, prefix: str
) -> Dict[str, np.ndarray]:
    return {
        f"{prefix}{label}": dilate(labels == label, mm, spacing)
        for label, mm in dilation_mm.items()
    }


def load_organs(case: str, tp: str, spacing: tuple) -> Dict[str, np.ndarray]:
    total = nib.load(HERE / "organs" / f"{case}_{tp}.nii.gz").dataobj
    head = nib.load(
        DATA / "labelsTr_head_glands_cavities" / f"PSMARegPSMA_{case}_0000_{tp}.nii.gz"
    ).dataobj
    return {
        **organ_masks(np.asarray(total), ORGAN_DILATION_MM, spacing, "total"),
        **organ_masks(np.asarray(head), HEAD_DILATION_MM, spacing, "head"),
    }


def threshold_arm(pet: np.ndarray, organs: Dict[str, np.ndarray]) -> np.ndarray:
    return (pet >= SUV_THRESHOLD) & ~np.any(list(organs.values()), axis=0)


def mtv_tlg(mask: np.ndarray, pet: np.ndarray, voxel_ml: float) -> tuple:
    return mask.sum() * voxel_ml, pet[mask].sum() * voxel_ml


def missed_lesions(gt: np.ndarray, pred: np.ndarray) -> tuple:
    components, n = ndimage.label(gt)
    hit = np.unique(components[pred & gt])
    return n, n - np.count_nonzero(hit)


def arm_metrics(arm, gt, pred, pet, voxel_ml) -> dict:
    gt_mtv, gt_tlg = mtv_tlg(gt, pet, voxel_ml)
    mtv, tlg = mtv_tlg(pred, pet, voxel_ml)
    n_lesions, n_missed = missed_lesions(gt, pred)
    return {
        "arm": arm,
        "gt_mtv_ml": gt_mtv,
        "gt_tlg": gt_tlg,
        "mtv_ml": mtv,
        "tlg": tlg,
        "mtv_err_ml": mtv - gt_mtv,
        "tlg_err": tlg - gt_tlg,
        "missed_gt_ml": (gt & ~pred).sum() * voxel_ml,
        "fp_ml": (pred & ~gt).sum() * voxel_ml,
        "n_lesions": n_lesions,
        "n_missed": n_missed,
    }


def save_mask(mask: np.ndarray, ref: nib.Nifti1Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), ref.affine), path)


def evaluate(name: str) -> list:
    case, tp = name.split("_")
    pet_img = nib.load(image_path(case, tp, "0001"))
    spacing = pet_img.header.get_zooms()[:3]
    voxel_ml = np.prod(spacing) / 1000
    pet = pet_img.get_fdata(dtype=np.float32)
    gt = np.asarray(nib.load(label_path(case, tp)).dataobj) > 0
    organs = load_organs(case, tp, spacing)

    gt_in_organ = {
        f"gt_in_{key}_ml": (gt & mask).sum() * voxel_ml
        for key, mask in organs.items()
    }
    arms = {"threshold": threshold_arm(pet, organs)}
    for arm, pred in arms.items():
        save_mask(pred, pet_img, HERE / "masks" / arm / f"{name}.nii.gz")
    return [
        {"image": name, **arm_metrics(arm, gt, pred, pet, voxel_ml), **gt_in_organ}
        for arm, pred in arms.items()
    ]


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(
        abs_mtv_err_ml=df.mtv_err_ml.abs(),
        abs_tlg_err=df.tlg_err.abs(),
    )
    cols = ["abs_mtv_err_ml", "abs_tlg_err", "missed_gt_ml", "fp_ml", "n_missed"]
    return df.groupby("arm")[cols].median()


def main() -> None:
    images = json.loads((HERE / "cases.json").read_text())
    rows = [row for name in tqdm(images, desc="evaluate") for row in evaluate(name)]
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "results.csv", index=False)
    print(summarise(df).to_string())


if __name__ == "__main__":
    main()
