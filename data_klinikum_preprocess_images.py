import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

"""
Step ANY: preprocess_klinikum.py (can be done at any time)

Step 1 (order doesn't matter)
    - segment_ribs.py
    - segment_total_segmentator.py
Step 2
    - process_klinikum_segmentations.py
        - remove ribs from total segmentator labels and replace with rib segmentations
"""

# Whole-body axial reconstructions, best first. Everything else in a session is
# a topogram, a thorax-only breath-hold, or a derived cor/sag MPR.
WHOLE_BODY_SERIES_BY_PRIORITY = (
    "TK Diagn EFoV",
    "TK Nativ EFoV",
    "CT ax 3mm SAF",
)

# need to exclude patients with PETs that we couldnt convert to SUV
EXLUCDE_PET = [
    "8M-rhLIeYlw",
    "T-2E5gxfsmI",
    "GDYu5GavJ8c",
    "2VYgP7OOA-w",
    "uWVuIYc-6gw",
    "TNhz3uRIjqo",
    "kA2OcxmxWDc",
    "BilJ1J-PJCw",
    "BilJ1J-PJCw",
    "5kcfs3M-ZEc",
    "5kcfs3M-ZEc",
    "jbHwWjKIrss",
    "DuEANzDZh6I",
    "D5nsU3QtnJM",
    "eaYmLmKQdZk",
    "2DX3v-69KUM",
    "rJZ-Ou2gwrs",
    "wzY5r43UD-I",
    "wzY5r43UD-I",
    "Q37sPdVoiWQ",
    "XHHoQGaCKCA",
    "XJgaQCRJmDE",
    "y06xSlyG2us",
    "y06xSlyG2us",
    "Da0fgYJxw8A",
    "F-eW7FMy79M",
    "vJYsjU0-u-w",
    "Ja9vNzAx-U ",
    "YxCW0a1l4Mk",
]


def select_ct(ct_dir: Path) -> Path:
    """Return the whole-body axial CT of a session's ``ct`` folder."""
    all_sidecars = sorted(ct_dir.glob("*_ct.json"))
    volumetric_sidecars = [p for p in all_sidecars if "part-localizer" not in p.name]

    candidates_by_series: Dict[str, List[Tuple[Path, float]]] = {}
    for sidecar in volumetric_sidecars:
        metadata = json.loads(sidecar.read_text())

        series_description = (metadata.get("SeriesDescription") or "").strip()
        if series_description not in WHOLE_BODY_SERIES_BY_PRIORITY:
            continue

        grid = metadata.get("grid") or {}
        shape, spacing = grid.get("shape"), grid.get("spacing")
        if not shape or not spacing:
            continue

        z_extent_mm = shape[2] * spacing[2]
        candidates_by_series.setdefault(series_description, []).append(
            (sidecar, z_extent_mm)
        )

    for series_description in WHOLE_BODY_SERIES_BY_PRIORITY:
        candidates = candidates_by_series.get(series_description)
        if not candidates:
            continue

        best_sidecar, _ = max(candidates, key=lambda candidate: candidate[1])
        selected_volume = best_sidecar.with_name(
            best_sidecar.name.removesuffix(".json") + ".nii.gz"
        )
        if not selected_volume.exists():
            raise FileNotFoundError(f"Missing volume for sidecar {best_sidecar}")
        return selected_volume

    raise FileNotFoundError(f"No whole-body axial CT found in {ct_dir}")


def automatically_find_pairs(data_dir: Path) -> List[Dict[str, Path]]:

    files = []

    for patient in sorted(data_dir.iterdir()):
        if not patient.is_dir():
            continue

        sessions = [s for s in patient.iterdir() if s.is_dir()]

        if len(sessions) != 2:
            continue

        sessions.sort(key=lambda s: s.name)

        session_fixed = sessions[0]
        session_moving = sessions[1]

        path_fixed = list((session_fixed / "ct").iterdir())[0]
        path_moving = list((session_moving / "ct").iterdir())[0]

        pair = {"fixed": path_fixed, "moving": path_moving}
        files.append(pair)

    return files


def find_corresponding_suv(ct_path: Path) -> Path:

    session = ct_path.parent.parent.name
    patient = ct_path.parent.parent.parent.name
    main_dir = ct_path.parent.parent.parent.parent

    suv_dir = main_dir.parent / "suv" / patient / session / "pet"

    suv_files = list(suv_dir.glob("*.nii.gz"))

    if len(suv_files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one SUV file in {suv_dir}, found {len(suv_files)}"
        )

    return suv_files[0]


def find_files_from_manual_selection(data_dir: Path) -> list[Path]:
    files = []
    for pid, sessions in SELECTED_FILES.items():
        for ses, filename in sessions.items():
            path = data_dir / pid / ses / "ct" / filename
            if not path.exists():
                raise FileNotFoundError(f"File {path} does not exist")
            files.append(path)
    return files


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


def resample_pet_label_with_offset(
    seg: sitk.Image,
    offset: np.ndarray,
    output_size: Tuple[int, int, int] = OUTPUT_SIZE,
    output_spacing: Tuple[float, float, float] = OUTPUT_SPACING,
    sigma_scale: float = 0.5,
    foreground_threshold: float = 0.45,
) -> sitk.Image:
    """Anti-aliased single-label resampling for PET tumour masks."""
    in_spacing = np.asarray(seg.GetSpacing(), dtype=float)
    out_spacing = np.asarray(output_spacing, dtype=float)
    sigma = np.maximum(sigma_scale * (out_spacing - in_spacing), 1e-3)

    transform = sitk.TranslationTransform(3)
    transform.SetOffset(np.asarray(offset, dtype=float).tolist())
    ref = make_reference(sitk.sitkFloat32, output_size, output_spacing)

    mask = sitk.Cast(sitk.NotEqual(seg, 0), sitk.sitkFloat32)
    mask = sitk.SmoothingRecursiveGaussian(mask, sigma.tolist(), False)
    resampled = sitk.Resample(
        mask, ref, transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32
    )
    score = sitk.GetArrayFromImage(resampled)
    out_arr = (score >= foreground_threshold).astype(np.int16)

    out = sitk.GetImageFromArray(out_arr)
    out.CopyInformation(make_reference(sitk.sitkInt16, output_size, output_spacing))
    return out


def preprocess_ct(
    in_path: Path,
    out_path: Path,
    output_size: Tuple[int, int, int] = OUTPUT_SIZE,
    output_spacing: Tuple[float, float, float] = OUTPUT_SPACING,
    threshold: float = THRESHOLD,
    top_margin_vox: int = TOP_MARGIN_VOX,
) -> np.ndarray:
    image = sitk.ReadImage(in_path)
    offset = compute_offset(
        image, output_size, output_spacing, threshold, top_margin_vox
    )

    if out_path.exists():
        return offset

    resampled = resample_with_offset(
        image,
        offset,
        interpolator=sitk.sitkLinear,
        default_value=-1024.0,
        output_size=output_size,
        output_spacing=output_spacing,
    )
    sitk.WriteImage(resampled, out_path.as_posix())

    return offset


def preprocess_suv(
    in_path: Path,
    out_path: Path,
    offset: np.ndarray,
    output_size: Tuple[int, int, int] = OUTPUT_SIZE,
    output_spacing: Tuple[float, float, float] = OUTPUT_SPACING,
) -> None:

    if out_path.exists():
        return

    image = sitk.ReadImage(in_path)

    resampled = resample_with_offset(
        image,
        offset,
        interpolator=sitk.sitkLinear,
        default_value=0.0,
        output_size=output_size,
        output_spacing=output_spacing,
    )
    sitk.WriteImage(resampled, out_path.as_posix())


def main() -> None:

    mapping = {}

    in_dir = Path("/home/iml/fryderyk.koegl/data/PET_CT_bone/raw_data")

    out_dir = Path(
        "/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/imagesTr_klinikum"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    mapping_path = Path(
        "/home/iml/fryderyk.koegl/code/LapIRN-koegl/data_klinikum_mapping.json"
    )
    if mapping_path.exists():
        try:
            with open(mapping_path, "r") as f:
                mapping = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            mapping = {}

    path_pairs = automatically_find_pairs(in_dir)

    pbar = tqdm(path_pairs, ncols=170, desc="Preprocessing CT/SUV pairs")
    for idx, pair in enumerate(pbar):
        path_fixed = pair["fixed"]

        patient_id = path_fixed.parent.parent.parent.name

        if patient_id in EXLUCDE_PET:
            tqdm.write(f"Skipping patient {patient_id} due to missing SUV")
            continue

        try:
            path_fixed_suv = find_corresponding_suv(path_fixed)
            path_moving = pair["moving"]
            path_moving_suv = find_corresponding_suv(path_moving)

            pbar.set_postfix_str(
                f"{path_fixed.name.replace('.nii.gz', '')} / {path_moving.name.replace('.nii.gz', '')}"
            )

            new_name_fix = f"PSMARegPSMA_4{idx:03d}_0000_00.nii.gz"
            new_name_fix_suv = f"PSMARegPSMA_4{idx:03d}_0001_00.nii.gz"
            new_name_mov = f"PSMARegPSMA_4{idx:03d}_0000_01.nii.gz"
            new_name_mov_suv = f"PSMARegPSMA_4{idx:03d}_0001_01.nii.gz"

            offset_fixed = preprocess_ct(path_fixed, out_dir / new_name_fix)
            offset_moving = preprocess_ct(path_moving, out_dir / new_name_mov)

            preprocess_suv(path_fixed_suv, out_dir / new_name_fix_suv, offset_fixed)
            preprocess_suv(path_moving_suv, out_dir / new_name_mov_suv, offset_moving)

            mapping[path_fixed.name] = new_name_fix
            mapping[new_name_fix] = path_fixed.name
            mapping[path_fixed_suv.name] = new_name_fix_suv
            mapping[new_name_fix_suv] = path_fixed_suv.name
            mapping[path_moving.name] = new_name_mov
            mapping[new_name_mov] = path_moving.name
            mapping[path_moving_suv.name] = new_name_mov_suv
            mapping[new_name_mov_suv] = path_moving_suv.name
        except Exception as e:
            tqdm.write(f"Error processing pair {path_fixed} and {path_moving}: {e}")
            continue
        x = 0
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=4)

    x = 0


if __name__ == "__main__":
    main()
