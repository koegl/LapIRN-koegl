"""Segment the *original* NLST CTs with TotalSegmentator, then resample the
label maps onto the preprocessed grid using the exact same transform that was
applied to the images in preprocess_nlst.py.

Running TotalSegmentator on the original resolution and resampling afterwards
gives noticeably better labels than segmenting the already-downsampled,
mostly-empty preprocessed volumes.
"""

import json
from pathlib import Path
from typing import Dict, Literal

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

from data_klinikum_preprocess_images import (
    compute_offset,
    resample_labels_with_offset,
    resample_pet_label_with_offset,
)

# TotalSegmentator rib labels (rib_left_1..rib_right_12)
RIB_LABELS = list(range(92, 116))


def load_mapping(mapping_path) -> Dict[str, str]:
    with open(mapping_path, "r") as f:
        mapping = json.load(f)
    return mapping


def get_orig_path(
    seg_dir_orig: Path,
    name: str,
    output_type: Literal["CT", "SEG_TOTAL", "SEG_RIB", "SEG_PET"],
) -> Path:

    name_clean = name.replace(".nii.gz", "")
    # recursively find the segmentation file in the original segmentations directory - there could be two
    found = []
    for path in seg_dir_orig.rglob(f"{name_clean}*.nii.gz"):
        found.append(path)

    if len(found) == 0:
        raise FileNotFoundError(
            f"Could not find segmentation for {name_clean} in {seg_dir_orig}"
        )
    elif len(found) > 2:
        raise FileExistsError(
            f"Found multiple segmentations for {name_clean} in {seg_dir_orig}: {found}"
        )

    path_rib = (
        found[0]
        if "ribs" in found[0].as_posix()
        else found[1]
        if len(found) > 1
        else None
    )
    path_seg = (
        found[0]
        if "ribs" not in found[0].as_posix()
        else found[1]
        if len(found) > 1
        else None
    )
    path_pet = (
        found[0]
        if "pet_labels" in found[0].as_posix()
        else found[1]
        if len(found) > 1
        else None
    )

    if path_rib is None and path_seg is None and path_pet is None:
        raise FileNotFoundError(
            f"Could not find both rib and total segmentations for {name_clean} in {seg_dir_orig}: {found}"
        )

    if output_type == "CT":
        output_path = Path(
            str(path_seg).replace("segmentations_total_segmentator", "raw_data")
        )
    elif output_type == "SEG_TOTAL":
        output_path = path_seg
    elif output_type == "SEG_RIB":
        output_path = path_rib
    elif output_type == "SEG_PET":
        output_path = path_pet
    else:
        raise ValueError(
            f"output_type must be 'CT', 'SEG_TOTAL', or 'SEG_RIB', got {output_type}"
        )

    if not output_path.exists():
        raise FileNotFoundError(f"Expected file {output_path} does not exist.")

    return output_path


def join_ribs(
    seg_total: sitk.Image,
    seg_rib: sitk.Image,
    rib_label: int = RIB_LABELS[0],
    overwrite_other_labels: bool = True,
) -> sitk.Image:
    """Merge the TotalSegmentator ribs (92..115) with the rib model's mask.

    A voxel becomes a rib if *either* model calls it one; both are binary in the
    output, so all ribs collapse to a single label (`rib_label`, which stays
    inside the rib range the label groups in Code/config.py use). Where the rib
    mask lands on a non-rib TotalSegmentator structure, the rib wins unless
    `overwrite_other_labels` is cleared.
    """
    if seg_rib.GetSize() != seg_total.GetSize():
        seg_rib = sitk.Resample(
            seg_rib, seg_total, sitk.Transform(), sitk.sitkNearestNeighbor, 0
        )

    total_arr = sitk.GetArrayFromImage(seg_total)
    rib_arr = sitk.GetArrayViewFromImage(seg_rib)

    was_rib = (total_arr >= RIB_LABELS[0]) & (total_arr <= RIB_LABELS[-1])
    total_arr[was_rib] = 0

    is_rib = was_rib | (rib_arr > 0)
    if not overwrite_other_labels:
        is_rib = is_rib & (total_arr == 0)
    total_arr[is_rib] = rib_label

    out = sitk.GetImageFromArray(total_arr)
    out.CopyInformation(seg_total)
    return out


def main() -> None:

    mapping = load_mapping(
        Path("/home/iml/fryderyk.koegl/code/LapIRN-koegl/data_klinikum_mapping.json")
    )
    with open(Path("/home/iml/fryderyk.koegl/code/LapIRN-koegl/split.json"), "r") as f:
        split = json.load(f)
    dir_orig_seg_total = Path(
        "/home/iml/fryderyk.koegl/data/PET_CT_bone/segmentations_total_segmentator"
    )
    dir_orig_seg_rib = Path(
        "/home/iml/fryderyk.koegl/data/PET_CT_bone/segmentations_ribs"
    )
    dir_orig_seg_pet = Path("/home/iml/fryderyk.koegl/data/PET_CT_bone/pet_labels")
    out_dir = Path(
        "/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/labelsTr_klinikum"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = {
        orig: new
        for orig, new in mapping.items()
        if not orig.startswith("PSMARegPSMA_")
    }

    case_items = sorted(cases.items(), key=lambda x: x[1])  # sort by new name

    # print("warning, reducing to one case for debug only")
    # case_items = case_items[0:2]

    pbar = tqdm(case_items, desc="Klinikum segmentations", unit="case", ncols=150)

    # offsets stored by patient~session
    offsets: Dict[str, np.ndarray] = {}

    failed_cases = []

    for orig_name, new_name in pbar:
        try:
            orig_name = "~".join(orig_name.split("~")[0:2])

            out_path = out_dir / new_name
            if out_path.exists():
                tqdm.write(f"skip {new_name}")
                continue

            patient_id = orig_name.split("~")[0]
            session_id = orig_name.split("~")[1]

            new_patient_id = new_name.split("_")[1]
            if new_patient_id in split["test"]:
                continue

            pbar.set_postfix_str(new_name)

            if "_0000_" in new_name:
                path_orig_seg_total = get_orig_path(
                    dir_orig_seg_total, orig_name, "SEG_TOTAL"
                )
                path_orig_seg_rib = get_orig_path(
                    dir_orig_seg_rib, orig_name, "SEG_RIB"
                )
                seg_orig_total = sitk.ReadImage(path_orig_seg_total.as_posix())
                seg_orig_rib = sitk.ReadImage(path_orig_seg_rib.as_posix())
                seg = join_ribs(seg_orig_total, seg_orig_rib)
            elif "_0001_" in new_name:
                path_orig_seg_pet = get_orig_path(
                    dir_orig_seg_pet, orig_name, "SEG_PET"
                )
                seg = sitk.ReadImage(path_orig_seg_pet.as_posix())
            else:
                raise ValueError(f"Unexpected new_name format: {new_name}")

            path_orig_im = get_orig_path(dir_orig_seg_total, orig_name, "CT")
            # im_orig = sitk.ReadImage(path_orig_im.as_posix())

            if f"{patient_id}~{session_id}" in offsets:
                offset = offsets[f"{patient_id}~{session_id}"]
            else:
                offset = compute_offset(sitk.ReadImage(path_orig_im.as_posix()))
                offsets[f"{patient_id}~{session_id}"] = offset

            if "_0000_" in new_name:
                resampled_seg = resample_labels_with_offset(seg, offset)
            elif "_0001_" in new_name:
                resampled_seg = resample_pet_label_with_offset(seg, offset)

            sitk.WriteImage(resampled_seg, out_path.as_posix())

        except Exception as e:
            tqdm.write(f"Failed to process {new_name}: {e}")
            failed_cases.append(new_name)

    with open(
        Path(
            "/home/iml/fryderyk.koegl/code/LapIRN-koegl/data_klinikum_preprocess_segmentations_failed.txt"
        ),
        "w",
    ) as f:
        for case in failed_cases:
            f.write(f"{case}\n")


if __name__ == "__main__":
    main()
