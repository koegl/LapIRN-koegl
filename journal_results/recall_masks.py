"""Recall-biased PET lesion masks for IO (the approach of the challenge winner,
github.com/neel-dey/reps-for-reg, regrep/segmentation/masks.py):

    mask = (SUV > 3  AND NOT  halo)  OR  nnU-Net prediction

halo = the 16 TotalSegmentator organs with physiological PSMA uptake, dilated
by a box of half-width 10 mm. They use LION; we use our nnU-Net (better here).
Organs come from the predicted CT labels (labelsTr_pred), so no ground truth is
used. Written next to the other labels as labelsTr_pred_recall (PET channel
only, same file names); the overlap with the ground-truth lesions is printed
and saved to recall_masks/summary.csv.

    ../.venv/bin/python recall_masks.py
"""

import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))

import evaluate  # noqa: E402

SUV_THRESHOLD = 3.0
ORGAN_HALO_MM = 10.0
# TotalSegmentator "total" ids of spleen, kidney_right, kidney_left,
# gallbladder, liver, stomach, pancreas, esophagus, small_bowel, duodenum,
# colon, urinary_bladder, kidney_cyst_left, kidney_cyst_right, heart, brain
PHYSIOLOGICAL_IDS = (1, 2, 3, 4, 5, 6, 7, 15, 18, 19, 20, 21, 23, 24, 51, 90)

DATA_DIR = evaluate.DATA_PATH / "PSMAReg/PSMAReg_dataset"
OUT_DIR = Path(__file__).resolve().parent / "recall_masks"


def dilate_box(mask: np.ndarray, radii) -> np.ndarray:
    kernel = tuple(2 * r + 1 for r in radii)
    pooled = F.max_pool3d(
        torch.from_numpy(mask).float()[None, None], kernel, stride=1, padding=tuple(radii)
    )
    return pooled[0, 0].numpy() > 0.5


def overlap(pred: np.ndarray, gt: np.ndarray) -> dict:
    tp = float((pred & gt).sum())
    components, n = ndimage.label(gt, structure=np.ones((3, 3, 3), bool))
    detected = len(np.unique(components[pred & (components > 0)]))
    return {
        "recall": tp / max(gt.sum(), 1),
        "precision": tp / max(pred.sum(), 1),
        "dice": 2 * tp / max(pred.sum() + gt.sum(), 1),
        "lesions_detected": detected / n if n else np.nan,
    }


def main() -> None:
    evaluate.RECALL_LABEL_DIR.mkdir(exist_ok=True)
    rows = []
    for nnunet_path in sorted(evaluate.PRED_LABEL_DIR.glob("PSMARegPSMA_*_0001_*.nii.gz")):
        name = nnunet_path.name
        organ_path = evaluate.PRED_LABEL_DIR / name.replace("_0001_", "_0000_")
        pet = nib.load(str(DATA_DIR / "imagesTr" / name))
        radii = [max(1, round(ORGAN_HALO_MM / float(s))) for s in pet.header.get_zooms()[:3]]

        nnunet = np.asarray(nib.load(str(nnunet_path)).dataobj) > 0
        organs = np.asarray(nib.load(str(organ_path)).dataobj)
        halo = dilate_box(np.isin(organs, PHYSIOLOGICAL_IDS), radii)
        suv = np.asarray(pet.dataobj, dtype=np.float32)
        mask = ((suv > SUV_THRESHOLD) & ~halo) | nnunet

        ref = nib.load(str(nnunet_path))
        nib.save(
            nib.Nifti1Image(mask.astype(np.uint8), ref.affine, ref.header),
            str(evaluate.RECALL_LABEL_DIR / name),
        )

        gt = np.asarray(nib.load(str(evaluate.GT_LABEL_DIR / name)).dataobj) == 1
        row = {"scan": name}
        for source, pred in (("nnunet", nnunet), ("recall", mask)):
            row.update({f"{source}_{k}": v for k, v in overlap(pred, gt).items()})
        rows.append(row)
        print(
            f"{name}: nnunet recall {row['nnunet_recall']:.2f} dice {row['nnunet_dice']:.2f} | "
            f"recall-mask recall {row['recall_recall']:.2f} dice {row['recall_dice']:.2f}",
            flush=True,
        )

    OUT_DIR.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "summary.csv", index=False)
    print(df.drop(columns="scan").describe().round(3).to_string())


if __name__ == "__main__":
    main()
