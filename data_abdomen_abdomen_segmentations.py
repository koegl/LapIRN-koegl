"""Segment the *original* NLST CTs with TotalSegmentator, then resample the
label maps onto the preprocessed grid using the exact same transform that was
applied to the images in preprocess_nlst.py.

Running TotalSegmentator on the original resolution and resampling afterwards
gives noticeably better labels than segmenting the already-downsampled,
mostly-empty preprocessed volumes.
"""

import json
from pathlib import Path
from typing import Dict

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from data_abdomen_preprocess import compute_offset, resample_with_offset

mapping_path = Path("/home/iml/fryderyk.koegl/code/LapIRN-koegl/mapping_abdomen.json")
orig_dir = Path("/home/iml/fryderyk.koegl/data/AbdomenCTCT/imagesTr")
seg_orig_dir = Path("/home/iml/fryderyk.koegl/data/AbdomenCTCT/labelsTr")
out_dir = Path("/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/labelsTr_abdomen")

skip_existing = True


def load_mapping() -> Dict[str, str]:
    with open(mapping_path, "r") as f:
        mapping = json.load(f)
    return mapping


def main() -> None:
    seg_orig_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    mapping = load_mapping()

    # keep only original -> new direction (mapping stores both)
    cases = {
        orig: new
        for orig, new in mapping.items()
        if not orig.startswith("PSMARegPSMA_")
    }

    pbar = tqdm(sorted(cases.items()), desc="Abdomen segmentations", unit="case")
    for orig_name, new_name in pbar:
        pbar.set_postfix_str(new_name)

        out_path = out_dir / new_name
        if skip_existing and out_path.exists():
            tqdm.write(f"skip {new_name}")
            continue

        orig_path = orig_dir / orig_name
        seg_orig_path = seg_orig_dir / orig_name

        seg = sitk.ReadImage(seg_orig_path.as_posix())

        # remove labels 4,5,12,13
        seg_arr = sitk.GetArrayViewFromImage(seg)
        labels_to_remove = (4, 5, 12, 13)
        new_arr = np.where(np.isin(seg_arr, labels_to_remove), 0, seg_arr)
        seg_clean = sitk.GetImageFromArray(new_arr)
        seg_clean.CopyInformation(seg)  # preserve spacing/origin/direction
        seg = seg_clean

        # the offset is a pure function of the original CT, so
        # recomputing it here reproduc  es exactly what preprocess_nlst.py applied
        offset = compute_offset(sitk.ReadImage(orig_path.as_posix()))

        resampled = resample_with_offset(
            seg,
            offset,
            interpolator=sitk.sitkNearestNeighbor,
            default_value=0.0,
        )
        sitk.WriteImage(resampled, out_path.as_posix())
        x = 0


if __name__ == "__main__":
    main()
