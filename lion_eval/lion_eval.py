"""Evaluate LION (https://github.com/ENHANCE-PET/LION, PSMA model) on a few PSMAReg training cases.

  python lion_eval.py prepare  [--n 6]   # subject folders with PT_ symlinks to our PET images
  bash run_lion.sh                       # LION docker, writes lionz-*/segmentations/ into each subject folder
  python lion_eval.py evaluate           # Dice etc. vs. our lesion labels

  # all validation cases of an nnU-Net fold, into a separate work folder:
  python lion_eval.py prepare  --fold 2 --work /home/iml/fryderyk.koegl/data/PSMAReg/lion_eval_fold2_val
  bash run_lion.sh /home/iml/fryderyk.koegl/data/PSMAReg/lion_eval_fold2_val
  python lion_eval.py evaluate --work /home/iml/fryderyk.koegl/data/PSMAReg/lion_eval_fold2_val

Dice is computed like nnU-Net's validation: per case 2TP / (2TP + FP + FN), NaN if reference and prediction are
both empty, averaged over cases ignoring NaN.
"""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import SimpleITK as sitk

DATA = Path("/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset")
SPLIT = Path("/home/iml/fryderyk.koegl/code/LapIRN-koegl/split_journal.json")
WORK = Path("/home/iml/fryderyk.koegl/data/PSMAReg/lion_eval")
SPLITS_NNUNET = Path("/home/iml/fryderyk.koegl/data/PSMAReg/lesion_seg_nnunet_autopet3/nnUNet_preprocessed/"
                     "Dataset901_PSMAReg_lesions/splits_final.json")


def prepare(work, n, seed, fold):
    if fold is None:  # n random training-split patients, baseline session
        patients = sorted(json.loads(SPLIT.read_text())["train"])
        random.Random(seed).shuffle(patients)
        cases = [(pid, "00") for pid in patients[:n]]
    else:  # all validation cases of an nnU-Net fold
        cases = [tuple(c.split("_")[1:3]) for c in json.loads(SPLITS_NNUNET.read_text())[fold]["val"]]
    for pid, session in cases:
        subject = work / f"PSMARegPSMA_{pid}_{session}"
        subject.mkdir(parents=True, exist_ok=True)
        link = subject / f"PT_PSMARegPSMA_{pid}_{session}.nii.gz"
        if not link.is_symlink():
            link.symlink_to(DATA / "imagesTr" / f"PSMARegPSMA_{pid}_0001_{session}.nii.gz")
    print(f"{len(cases)} subject folders in {work}")


def metrics(pred, ref, voxel_ml):
    tp, fp, fn = int((pred & ref).sum()), int((pred & ~ref).sum()), int((~pred & ref).sum())
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
    return {"dice": round(dice, 4), "tp": tp, "fp": fp, "fn": fn, "ref_ml": round(float(ref.sum()) * voxel_ml, 1), "pred_ml": round(float(pred.sum()) * voxel_ml, 1),
            "tp_ml": round(tp * voxel_ml, 1), "fp_ml": round(fp * voxel_ml, 1), "fn_ml": round(fn * voxel_ml, 1)}


def evaluate(work):
    rows = []
    for subject in sorted(p for p in work.iterdir() if p.is_dir()):
        segs = sorted(subject.glob("lionz-*/segmentations/*tumor_seg.nii.gz"))
        if not segs:
            print(f"{subject.name}: no LION output")
            continue
        pid, session = subject.name.split("_")[1:3]
        ref_img = sitk.ReadImage(str(DATA / "labelsTr" / f"PSMARegPSMA_{pid}_0001_{session}.nii.gz"))
        # LION should return the input grid; resample nearest-neighbour onto the label grid just in case
        pred_img = sitk.Resample(sitk.ReadImage(str(segs[-1])), ref_img, sitk.Transform(), sitk.sitkNearestNeighbor, 0)
        ref, pred = sitk.GetArrayFromImage(ref_img) > 0, sitk.GetArrayFromImage(pred_img) > 0
        voxel_ml = float(np.prod(ref_img.GetSpacing())) / 1000
        rows.append({"case": subject.name, **metrics(pred, ref, voxel_ml)})
    if rows:
        dice = [r["dice"] for r in rows if not np.isnan(r["dice"])]  # NaN = no lesion in reference and prediction
        print(f"mean Dice over {len(dice)} cases with lesions or predictions ({len(rows)} evaluated): {np.mean(dice):.4f}")
        print("mean voxels per case: " + ", ".join(f"{k.upper()} {np.mean([r[k] for r in rows]):.1f}" for k in ("tp", "fp", "fn")))
        with open(work / "lion_metrics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["prepare", "evaluate"])
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fold", type=int, default=None, help="use all validation cases of this nnU-Net fold")
    p.add_argument("--work", type=Path, default=WORK)
    a = p.parse_args()
    prepare(a.work, a.n, a.seed, a.fold) if a.step == "prepare" else evaluate(a.work)
