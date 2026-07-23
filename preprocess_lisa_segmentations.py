"""Segment the *original* NLST CTs with TotalSegmentator, then resample the
label maps onto the preprocessed grid using the exact same transform that was
applied to the images in preprocess_nlst.py.

Running TotalSegmentator on the original resolution and resampling afterwards
gives noticeably better labels than segmenting the already-downsampled,
mostly-empty preprocessed volumes.
"""

from pathlib import Path

import numpy as np
import SimpleITK as sitk

from preprocess_lisa import compute_offset, resample_labels_with_offset


def main() -> None:

    out_dir = Path("/home/iml/fryderyk.koegl/code/LapIRN-koegl/lisa/preprocessed")

    paths_seg = [
        # Path(
        #     "/home/iml/fryderyk.koegl/code/LapIRN-koegl/lisa/sub-0L4e4-f1XHo_ses-20180604_sequ-6_acq-cor_ce-ContrastAgent_part-axial_ct_label.nii.gz"
        # ),
        Path(
            "/home/iml/fryderyk.koegl/code/LapIRN-koegl/lisa/sub-0L4e4-f1XHo_ses-20180919_sequ-6_acq-cor_ce-ContrastAgent_part-axial_ct_label.nii.gz"
        ),
    ]
    paths_origs = [
        # Path(
        #     "/home/iml/fryderyk.koegl/code/LapIRN-koegl/lisa/sub-0L4e4-f1XHo_ses-20180604_sequ-6_acq-cor_ce-ContrastAgent_part-axial_ct.nii.gz"
        # ),
        Path(
            "/home/iml/fryderyk.koegl/code/LapIRN-koegl/lisa/sub-0L4e4-f1XHo_ses-20180919_sequ-6_acq-cor_ce-ContrastAgent_part-axial_ct.nii.gz"
        ),
    ]

    for path_seg, path_orig in zip(paths_seg, paths_origs):
        name = path_seg.name
        out_path = out_dir / name

        seg = sitk.ReadImage(path_seg.as_posix())

        # remove labels 4,5,12,13
        seg_arr = sitk.GetArrayViewFromImage(seg)
        labels_to_remove = ()
        # labels_to_remove = (4, 5, 12, 13)
        new_arr = np.where(np.isin(seg_arr, labels_to_remove), 0, seg_arr)
        seg_clean = sitk.GetImageFromArray(new_arr)
        seg_clean.CopyInformation(seg)  # preserve spacing/origin/direction
        seg = seg_clean

        # the offset is a pure function of the original CT, so
        # recomputing it here reproduc  es exactly what preprocess_nlst.py applied
        offset = compute_offset(sitk.ReadImage(path_orig.as_posix()))

        resampled = resample_labels_with_offset(seg, offset)
        sitk.WriteImage(resampled, out_path.as_posix())


if __name__ == "__main__":
    main()
