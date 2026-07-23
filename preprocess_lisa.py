from pathlib import Path
from typing import Tuple

import numpy as np
import SimpleITK as sitk


def foreground_anchor(
    image: sitk.Image, threshold: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = sitk.GetArrayFromImage(image)  # (z, y, x)
    idx = np.array(np.nonzero(arr > threshold))
    lo_zyx = idx.min(axis=1).astype(float)
    hi_zyx = idx.max(axis=1).astype(float)
    mid_zyx = (lo_zyx + hi_zyx) / 2.0

    lo_xyz = lo_zyx[::-1]
    hi_xyz = hi_zyx[::-1]
    mid_xyz = mid_zyx[::-1]

    # center in x/y, top edge in z
    anchor_idx = np.array([mid_xyz[0], mid_xyz[1], hi_xyz[2]])
    anchor = np.array(
        image.TransformContinuousIndexToPhysicalPoint(anchor_idx.tolist())
    )
    return anchor, lo_xyz, hi_xyz


OUTPUT_SIZE: Tuple[int, int, int] = (192, 192, 288)
OUTPUT_SPACING: Tuple[float, float, float] = (2.7344, 2.7344, 3.27)
THRESHOLD: float = -500.0
TOP_MARGIN_VOX: int = 0


def make_reference(
    pixel_id: int,
    output_size: Tuple[int, int, int] = OUTPUT_SIZE,
    output_spacing: Tuple[float, float, float] = OUTPUT_SPACING,
) -> sitk.Image:
    ref = sitk.Image(list(output_size), pixel_id)
    ref.SetOrigin((0.0, 0.0, 0.0))
    ref.SetDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
    ref.SetSpacing(output_spacing)
    return ref


def compute_offset(
    image: sitk.Image,
    output_size: Tuple[int, int, int] = OUTPUT_SIZE,
    output_spacing: Tuple[float, float, float] = OUTPUT_SPACING,
    threshold: float = THRESHOLD,
    top_margin_vox: int = TOP_MARGIN_VOX,
) -> np.ndarray:
    """Translation (physical mm) mapping the reference grid onto `image`.
    Depends only on `image`, so it can be recomputed at any time."""
    ref = make_reference(image.GetPixelID(), output_size, output_spacing)

    n = np.array(output_size, dtype=float)
    target_idx = np.array(
        [(n[0] - 1.0) / 2.0, (n[1] - 1.0) / 2.0, n[2] - 1.0 - top_margin_vox]
    )
    target = np.array(ref.TransformContinuousIndexToPhysicalPoint(target_idx.tolist()))

    anchor, lo_xyz, hi_xyz = foreground_anchor(image, threshold)
    offset = anchor - target
    return offset


def resample_with_offset(
    image: sitk.Image,
    offset: np.ndarray,
    interpolator: int,
    default_value: float,
    output_size: Tuple[int, int, int] = OUTPUT_SIZE,
    output_spacing: Tuple[float, float, float] = OUTPUT_SPACING,
) -> sitk.Image:
    ref = make_reference(image.GetPixelID(), output_size, output_spacing)
    transform = sitk.TranslationTransform(3)
    transform.SetOffset(np.asarray(offset, dtype=float).tolist())
    resampled = sitk.Resample(
        image, ref, transform, interpolator, default_value, image.GetPixelID()
    )
    return resampled


def resample_labels_with_offset(
    seg: sitk.Image,
    offset: np.ndarray,
    output_size: Tuple[int, int, int] = OUTPUT_SIZE,
    output_spacing: Tuple[float, float, float] = OUTPUT_SPACING,
    sigma_scale: float = 0.5,
    foreground_threshold: float = 0.35,
) -> sitk.Image:
    """Anti-aliased label resampling.

    Nearest-neighbour point-samples the label map, which punches holes into
    structures thinner than the output voxel (ribs). Instead each label's
    indicator function is low-pass filtered, resampled linearly and the label
    with the highest response wins, so a rib covering a decent fraction of an
    output voxel still claims it.
    """
    in_spacing = np.asarray(seg.GetSpacing(), dtype=float)
    out_spacing = np.asarray(output_spacing, dtype=float)
    sigma = np.maximum(sigma_scale * (out_spacing - in_spacing), 1e-3)

    transform = sitk.TranslationTransform(3)
    transform.SetOffset(np.asarray(offset, dtype=float).tolist())
    ref = make_reference(sitk.sitkFloat32, output_size, output_spacing)

    seg_arr = sitk.GetArrayViewFromImage(seg)
    labels = [int(v) for v in np.unique(seg_arr) if v != 0]

    best_score = np.full(
        tuple(output_size)[::-1], foreground_threshold, dtype=np.float32
    )
    out_arr = np.zeros(tuple(output_size)[::-1], dtype=np.int16)

    for label in labels:
        mask = sitk.Cast(sitk.Equal(seg, label), sitk.sitkFloat32)
        mask = sitk.SmoothingRecursiveGaussian(mask, sigma.tolist(), False)
        resampled = sitk.Resample(
            mask, ref, transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32
        )
        score = sitk.GetArrayFromImage(resampled)
        wins = score > best_score
        best_score[wins] = score[wins]
        out_arr[wins] = label

    out = sitk.GetImageFromArray(out_arr)
    out.CopyInformation(make_reference(sitk.sitkInt16, output_size, output_spacing))
    return out


def preprocess_file(
    in_path: Path,
    out_path: Path,
    output_size: Tuple[int, int, int] = OUTPUT_SIZE,
    output_spacing: Tuple[float, float, float] = OUTPUT_SPACING,
    threshold: float = THRESHOLD,
    top_margin_vox: int = TOP_MARGIN_VOX,
) -> None:
    image = sitk.ReadImage(in_path)
    offset = compute_offset(
        image, output_size, output_spacing, threshold, top_margin_vox
    )
    resampled = resample_with_offset(
        image,
        offset,
        interpolator=sitk.sitkLinear,
        default_value=-1024.0,
        output_size=output_size,
        output_spacing=output_spacing,
    )
    sitk.WriteImage(resampled, out_path.as_posix())


def main() -> None:

    mapping = {}

    out_dir = Path("/home/iml/fryderyk.koegl/code/LapIRN-koegl/lisa/preprocessed")
    out_dir.mkdir(parents=True, exist_ok=True)

    mapping_path = Path(
        "/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/mapping_nlst.json"
    )
    if mapping_path.exists():
        mapping_path.unlink()

    path_fixed = Path(
        "/home/iml/fryderyk.koegl/code/LapIRN-koegl/lisa/sub-0L4e4-f1XHo_ses-20180604_sequ-6_acq-cor_ce-ContrastAgent_part-axial_ct.nii.gz"
    )
    path_moving = Path(
        "/home/iml/fryderyk.koegl/code/LapIRN-koegl/lisa/sub-0L4e4-f1XHo_ses-20180919_sequ-6_acq-cor_ce-ContrastAgent_part-axial_ct.nii.gz"
    )

    new_name_fix = path_fixed.name
    new_name_mov = path_moving.name

    preprocess_file(path_fixed, out_dir / new_name_fix)
    preprocess_file(path_moving, out_dir / new_name_mov)


if __name__ == "__main__":
    main()
