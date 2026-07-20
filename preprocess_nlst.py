import json
from pathlib import Path
from typing import Dict, List, Tuple

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


def flip_ap(image: sitk.Image) -> sitk.Image:
    """Flip the voxel data along the anterior-posterior (y) axis while keeping
    the affine (origin/spacing/direction) unchanged. NLST is stored with the
    spine in front; this moves it to the back to match PSMA. Because the affine
    is untouched, downstream geometry (anchor, offset, resample) is unaffected
    except for the intended repositioning of the anatomy."""
    arr = sitk.GetArrayFromImage(image)  # (z, y, x)
    flipped = np.ascontiguousarray(arr[:, ::-1, :])
    out = sitk.GetImageFromArray(flipped)
    out.CopyInformation(image)
    return out


OUTPUT_SIZE: Tuple[int, int, int] = (192, 192, 288)
OUTPUT_SPACING: Tuple[float, float, float] = (2.7344, 2.7344, 3.27)
THRESHOLD: float = -500.0
TOP_MARGIN_VOX: int = 50


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
    image = flip_ap(image)
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


def collect_pairs(input_dir: Path) -> Dict[str, Dict[str, List[Path]]]:
    files = sorted(input_dir.glob("*.nii.gz"))

    pairs: Dict[str, Dict[str, List[Path]]] = {}
    for f in files:
        if "_0000.nii.gz" in f.name:
            key = "fix"
        else:
            key = "mov"

        pid = f.name.split("_")[1]

        if pid not in pairs:
            pairs[pid] = {"fix": "", "mov": ""}
        pairs[pid][key] = f

    return pairs


def main() -> None:

    mapping = {}

    inp_dir = Path("/home/iml/fryderyk.koegl/data/NLST_l2r_2023/imagesTr")
    out_dir = Path(
        "/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/imagesTr_nlst"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    mapping_path = Path(
        "/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/mapping_nlst.json"
    )
    if mapping_path.exists():
        mapping_path.unlink()

    pairs = collect_pairs(inp_dir)

    for idx, (pid, d) in enumerate(tqdm(pairs.items())):
        # if pid not in "1d7f7abc18_BL_img_BL_img_00":
        #     continue
        path_fix = d["fix"]
        path_mov = d["mov"]

        new_name_fix = f"PSMARegPSMA_2{idx:03d}_0000_00.nii.gz"
        new_name_mov = f"PSMARegPSMA_2{idx:03d}_0000_01.nii.gz"

        preprocess_file(path_fix, out_dir / new_name_fix)
        preprocess_file(path_mov, out_dir / new_name_mov)

        mapping[path_fix.name] = new_name_fix
        mapping[new_name_fix] = path_fix.name
        mapping[path_mov.name] = new_name_mov
        mapping[new_name_mov] = path_mov.name

        # if idx == 1:
        #     break

    # save the mapping to json with 4 idnentiaiotn
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=4)


if __name__ == "__main__":
    main()
