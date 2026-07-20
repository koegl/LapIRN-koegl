"""Segment the *original* NLST CTs with TotalSegmentator, then resample the
label maps onto the preprocessed grid using the exact same transform that was
applied to the images in preprocess_nlst.py.

Running TotalSegmentator on the original resolution and resampling afterwards
gives noticeably better labels than segmenting the already-downsampled,
mostly-empty preprocessed volumes.
"""

import json
import subprocess
from pathlib import Path
from typing import Dict

import SimpleITK as sitk
from tqdm import tqdm

from preprocess_nlst import compute_offset, flip_ap, resample_with_offset

mapping_path = Path(
    "/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/mapping_nlst.json"
)
orig_dir = Path("/home/iml/fryderyk.koegl/data/NLST_l2r_2023/imagesTr")
# full-resolution segmentations of the originals (kept as a cache)
seg_orig_dir = Path("/home/iml/fryderyk.koegl/data/NLST_l2r_2023/labelsTr_totalseg")
out_dir = Path("/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/labelsTr_nlst")

fast = False
skip_existing = True


def segment(image: Path, output: Path, fast: bool = False) -> None:
    cmd = ["TotalSegmentator", "-i", str(image), "-o", str(output), "--ml"]
    if fast:
        cmd.append("-f")
    subprocess.run(cmd, check=True)


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

    pbar = tqdm(sorted(cases.items()), desc="NLST segmentations", unit="case")
    for orig_name, new_name in pbar:
        pbar.set_postfix_str(new_name)

        out_path = out_dir / new_name
        if skip_existing and out_path.exists():
            tqdm.write(f"skip {new_name}")
            continue

        orig_path = orig_dir / orig_name
        seg_orig_path = seg_orig_dir / orig_name

        if not seg_orig_path.exists():
            segment(orig_path, seg_orig_path, fast=fast)

        # match the AP flip that preprocess_nlst.py applies to the images: both
        # the label map and the CT used for the offset must be flipped the same
        # way, since the anchor depends on the (now flipped) foreground.
        seg = flip_ap(sitk.ReadImage(seg_orig_path.as_posix()))
        # the offset is a pure function of the (flipped) original CT, so
        # recomputing it here reproduces exactly what preprocess_nlst.py applied
        offset = compute_offset(flip_ap(sitk.ReadImage(orig_path.as_posix())))

        resampled = resample_with_offset(
            seg,
            offset,
            interpolator=sitk.sitkNearestNeighbor,
            default_value=0.0,
        )
        sitk.WriteImage(resampled, out_path.as_posix())
        tqdm.write(f"done {new_name}")


if __name__ == "__main__":
    main()
