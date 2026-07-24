import json
from pathlib import Path
from typing import Tuple

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


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
TOP_MARGIN_VOX: int = 110


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

    # The body top edge (anchor) is placed at output z-index
    # `output_size[2] - 1 - top_margin_vox`. Everything above it is outside the
    # body and, after resampling, is a mix of source zero-padding (0) and the
    # resample default (-1024). Blank that margin to a uniform background so the
    # top strip is clean air instead of a 0/-1024 patchwork.
    arr = sitk.GetArrayFromImage(resampled)  # (z, y, x)
    top_start = output_size[2] - top_margin_vox
    arr[top_start:, :, :] = -1024.0
    cleaned = sitk.GetImageFromArray(arr)
    cleaned.CopyInformation(resampled)

    sitk.WriteImage(cleaned, out_path.as_posix())


def main() -> None:

    mapping = {}

    inp_dir = Path("/home/iml/fryderyk.koegl/data/AbdomenCTCT/imagesTr")
    out_dir = Path(
        "/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/imagesTr_abdomen"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    mapping_path = Path(
        "/home/iml/fryderyk.koegl/code/LapIRN-koegl/mapping_abdomen.json"
    )
    if mapping_path.exists():
        mapping_path.unlink()

    images = sorted(inp_dir.glob("*.nii.gz"))

    for idx, image in enumerate(tqdm(images)):
        new_name = f"PSMARegPSMA_3{idx:03d}_0000_{idx:02d}.nii.gz"
        new_path = out_dir / new_name

        if new_path.exists():
            tqdm.write(f"skip {new_path.name}")
            continue

        preprocess_file(image, new_path)

        mapping[image.name] = new_name
        mapping[new_name] = image.name
    # save the mapping to json with 4 idnentiaiotn
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=4)


if __name__ == "__main__":
    main()
