import json
from pathlib import Path
from typing import Dict, Hashable, List, Mapping, Optional, Tuple

import config
import monai.data as monai_data
import nibabel as nib
import numpy as np
import sdt
import synthetic
import torch
from monai.transforms import (
    Compose,
    ConcatItemsd,
    MapTransform,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Transform,
)
from scipy import ndimage
from torch.utils import data as torch_data


class ComputeSDTChannelsd(MapTransform):
    """Compute grouped SDT channels from a label map and store under out_key.

    Runs once inside the cached load transform. Output is (n_groups, H, W, D).
    """

    def __init__(
        self,
        label_key: str,
        out_key: str,
        label_groups: List[List[int]],
        clip_vox: float,
    ) -> None:
        super().__init__(keys=[label_key])
        self.label_key = label_key
        self.out_key = out_key
        self.label_groups = label_groups
        self.clip_vox = clip_vox

    def __call__(
        self, data: Mapping[Hashable, torch.Tensor]
    ) -> Mapping[Hashable, torch.Tensor]:
        d = dict(data)
        seg = d[self.label_key].squeeze(0).cpu().numpy().astype(np.int64)
        channels = []
        for group in self.label_groups:
            mask = np.isin(seg, group)
            signed_dt = sdt.compute_clipped_sdt(mask, self.clip_vox)
            channels.append(signed_dt)
        stacked = np.stack(channels, axis=0)
        d[self.out_key] = torch.from_numpy(stacked).float()
        return d


def save_volume(
    volume: torch.Tensor,
    out_dir: Path,
    epoch,
    reference_path: Optional[Path] = None,
    name: Optional[str] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    if reference_path is not None:
        fixed_nib = nib.load(reference_path.as_posix())
        affine = fixed_nib.affine
    else:
        affine = np.eye(4)

    if name is None:
        name = "temp"

    nib.save(
        nib.Nifti1Image(volume.detach().squeeze().cpu().numpy(), affine),
        str(out_dir / f"{name}_{epoch:05d}.nii.gz"),
    )


def save_initial(
    model: torch.nn.Module,
    X: torch.Tensor,
    X_prereg: torch.Tensor,
    Y_4x: torch.Tensor,
    F_X_Y: torch.Tensor,
    cfg: config.TrainingConfig,
    epoch: int,
    run_name: str,
    batch: Dict[str, List[str]],
    level: int,
) -> None:
    zero_disp = torch.zeros_like(F_X_Y)
    x_ref = model.transform(X, zero_disp.permute(0, 2, 3, 4, 1), model.grid_1)
    x_prereg = model.transform(X_prereg, zero_disp.permute(0, 2, 3, 4, 1), model.grid_1)
    save_volume(
        volume=x_ref[:, 0:1, ...],
        out_dir=cfg.save_dir / "initial",
        epoch=epoch,
        name=f"x_ref_ct_lvl{level}_{run_name}_{batch['case_id'][0]}",
    )
    save_volume(
        volume=x_prereg[:, 0:1, ...],
        out_dir=cfg.save_dir / "initial",
        epoch=epoch,
        name=f"x_prereg_ct_lvl{level}_{run_name}_{batch['case_id'][0]}",
    )
    save_volume(
        volume=Y_4x[:, 0:1, ...],
        out_dir=cfg.save_dir / "initial",
        epoch=epoch,
        name=f"y_ref_ct_lvl{level}_{run_name}_{batch['case_id'][0]}",
    )


def norm_ct(vol: np.ndarray) -> np.ndarray:
    """Min-max normalize a CT volume to [0, 1]."""
    return (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)


def norm_pet(vol: np.ndarray, suv_max: float = 20.0) -> np.ndarray:
    """Clip and scale a PET SUV volume to [0, 1]."""
    vol = np.clip(vol, 0.0, suv_max)
    return vol / suv_max


def list_case_timepoints(data_dir: Path) -> Dict[str, List[str]]:
    """Map each case id to its sorted list of available timepoint suffixes.

    Discovery is based on the CT channel ("0000") files only, since CT and
    PET are assumed to share the same set of timepoints per patient.

    Args:
        data_dir: Dataset root containing imagesTr and labelsTr.

    Returns:
        Dict mapping case id (e.g. "0006") to a sorted list of timepoint
        strings (e.g. ["00", "01", "02"]).
    """
    image_dir = data_dir / "imagesTr"
    timepoints: Dict[str, List[str]] = {}
    for path in image_dir.glob("PSMARegPSMA_*_0000_*.nii.gz"):
        case_id, _, timepoint = path.name.removesuffix(".nii.gz").split("_")[1:4]
        timepoints.setdefault(case_id, []).append(timepoint)
    return {case_id: sorted(tps) for case_id, tps in sorted(timepoints.items())}


def build_registration_pairs(
    case_timepoints: Dict[str, List[str]],
    case_ids: Optional[List[str]] = None,
    min_timepoints: int = 2,
    baseline_tp: str = "00",
    include_intermediate_pairs: bool = False,
) -> List[Tuple[str, str, str]]:
    selected_ids = sorted(case_ids) if case_ids is not None else sorted(case_timepoints)
    pairs: List[Tuple[str, str, str]] = []
    for case_id in selected_ids:
        tps = case_timepoints[case_id]
        if len(tps) < min_timepoints or baseline_tp not in tps:
            continue

        if include_intermediate_pairs:
            # tps is sorted, so j > i means tps[j] is temporally later.
            # moving = later scan, fixed = earlier scan.
            for i in range(len(tps)):
                for j in range(i + 1, len(tps)):
                    moving_tp = tps[j]
                    fixed_tp = tps[i]
                    pairs.append((case_id, moving_tp, fixed_tp))
        else:
            for tp in tps:
                if tp == baseline_tp:
                    continue
                pairs.append((case_id, tp, baseline_tp))

    return pairs


def remove_tubingen(
    split_train: List[str], split_valid: List[str]
) -> Tuple[List[str], List[str]]:
    split_train_filtered = [cid for cid in split_train if not cid.startswith("1")]
    split_valid_filtered = [cid for cid in split_valid if not cid.startswith("1")]

    return split_train_filtered, split_valid_filtered


def remove_nlst(
    split_train: List[str], split_valid: List[str]
) -> Tuple[List[str], List[str]]:
    split_train_filtered = [cid for cid in split_train if not cid.startswith("2")]
    split_valid_filtered = [cid for cid in split_valid if not cid.startswith("2")]

    return split_train_filtered, split_valid_filtered


def remove_abdomen(
    split_train: List[str], split_valid: List[str]
) -> Tuple[List[str], List[str]]:
    split_train_filtered = [cid for cid in split_train if not cid.startswith("3")]
    split_valid_filtered = [cid for cid in split_valid if not cid.startswith("3")]

    return split_train_filtered, split_valid_filtered


def remove_non_tubingen(
    split_train: List[str], split_valid: List[str]
) -> Tuple[List[str], List[str]]:

    split_train_filtered = [cid for cid in split_train if cid.startswith("1")]
    split_valid_filtered = [cid for cid in split_valid if cid.startswith("1")]

    return split_train_filtered, split_valid_filtered


def remove_non_nlst(
    split_train: List[str], split_valid: List[str]
) -> Tuple[List[str], List[str]]:

    split_train_filtered = [cid for cid in split_train if cid.startswith("2")]
    split_valid_filtered = [cid for cid in split_valid if cid.startswith("2")]

    return split_train_filtered, split_valid_filtered


def remove_non_abdomen(
    split_train: List[str], split_valid: List[str]
) -> Tuple[List[str], List[str]]:

    split_train_filtered = [cid for cid in split_train if cid.startswith("3")]
    split_valid_filtered = [cid for cid in split_valid if cid.startswith("3")]

    return split_train_filtered, split_valid_filtered


def get_train_val_split(
    data_dir: Path,
    split_path: Path,
    val_fraction: float = 0.2,
    seed: int = 0,
    min_timepoints: int = 2,
    tubingen: bool = False,
    nlst: bool = False,
    abdomen: bool = False,
) -> Tuple[List[str], List[str]]:
    """Get or create a patient-level train/val split."""
    if split_path.exists():
        with open(split_path, "r") as f:
            split = json.load(f)

        train, val = split["train"], split["val"]

        if tubingen:
            train, val = remove_non_tubingen(train, val)
        else:
            train, val = remove_tubingen(train, val)

        if nlst:
            train, val = remove_non_nlst(train, val)
        else:
            train, val = remove_nlst(train, val)

        if abdomen:
            train, val = remove_non_abdomen(train, val)
        else:
            train, val = remove_abdomen(train, val)

        return train, val

    case_timepoints = list_case_timepoints(data_dir)
    eligible_ids = sorted(
        case_id
        for case_id, tps in case_timepoints.items()
        if len(tps) >= min_timepoints
    )

    rng = np.random.RandomState(seed)
    shuffled = eligible_ids.copy()
    rng.shuffle(shuffled)
    n_val = int(round(len(shuffled) * val_fraction))
    val_ids = sorted(shuffled[:n_val])
    train_ids = sorted(shuffled[n_val:])

    split_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_path, "w") as f:
        if tubingen:
            train_ids, val_ids = remove_non_tubingen(train_ids, val_ids)
        else:
            train_ids, val_ids = remove_tubingen(train_ids, val_ids)

        json.dump({"train": train_ids, "val": val_ids}, f, indent=2)

    return train_ids, val_ids


def list_single_session_sources(
    data_dir: Path,
    exclude_case_ids: Optional[List[str]] = None,
) -> List[Tuple[str, str]]:
    """(case_id, tp) for every case with < 2 timepoints (the unpaired singletons)."""
    case_timepoints = list_case_timepoints(data_dir)
    exclude = set(exclude_case_ids or [])
    sources: List[Tuple[str, str]] = []
    for case_id, tps in case_timepoints.items():
        if case_id in exclude:
            continue
        if len(tps) < 2:
            for tp in tps:
                sources.append((case_id, tp))
    return sources


def load_single_session(data_dir: Path, case_id: str, tp: str) -> dict:
    """Load + body-mask + normalize one session. Keys mirror the fixed side."""
    image_dir = data_dir / "imagesTr"
    label_dir = data_dir / "labelsTr"

    def load_vol(path: Path) -> np.ndarray:
        return nib.load(path).get_fdata().astype(np.float32)

    def load_lbl(path: Path) -> np.ndarray:
        return nib.load(path).get_fdata().astype(np.uint8)

    def to_t(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(arr).unsqueeze(0)

    ct_raw = load_vol(image_dir / f"PSMARegPSMA_{case_id}_0000_{tp}.nii.gz")
    mask = get_body_mask(ct_raw)
    ct_raw = apply_body_mask(ct_raw, mask, fill_value=float(np.percentile(ct_raw, 0.5)))

    pet_raw = load_vol(image_dir / f"PSMARegPSMA_{case_id}_0001_{tp}.nii.gz")
    pet_raw = apply_body_mask(pet_raw, mask, fill_value=0.0)

    body_map = synthetic.build_body_index_map(
        to_t(load_lbl(label_dir / f"PSMARegPSMA_{case_id}_0000_{tp}.nii.gz")),
        synthetic.BONE_LABEL_VALUES,
        min_voxels=200,
    )
    out = {
        "ct": to_t(norm_ct(ct_raw)).float(),
        "pet": to_t(norm_pet(pet_raw)).float(),
        "label_ct": to_t(
            load_lbl(label_dir / f"PSMARegPSMA_{case_id}_0000_{tp}.nii.gz")
        ),
        "label_pet": to_t(
            load_lbl(label_dir / f"PSMARegPSMA_{case_id}_0001_{tp}.nii.gz")
        ),
        "body_map": body_map,
        "body_mask": to_t(mask.astype(np.float32)).float(),
    }
    return out


def component_extent(coords):
    return coords.max(axis=0) - coords.min(axis=0) + 1


def slice_border_hits(component):
    return int(
        component[0, :].sum()
        + component[-1, :].sum()
        + component[:, 0].sum()
        + component[:, -1].sum()
    )


def select_body_components(mask2d, center_mask, prev_support):
    labels, num_labels = ndimage.label(mask2d)
    if num_labels == 0:
        return np.zeros_like(mask2d, dtype=bool)
    selected = np.zeros_like(mask2d, dtype=bool)
    support = (
        None
        if prev_support is None
        else ndimage.binary_dilation(
            prev_support, structure=np.ones((9, 9), dtype=bool)
        )
    )
    fallback_component = np.zeros_like(mask2d, dtype=bool)
    fallback_score = -np.inf
    for label_idx in range(1, num_labels + 1):
        component = labels == label_idx
        coords = np.argwhere(component)
        area = int(coords.shape[0])
        if area < 64:
            continue
        extent = component_extent(coords)
        if int(extent.min()) < 6:
            continue
        center_hits = int(np.logical_and(component, center_mask).sum())
        border_hits = slice_border_hits(component)
        overlap_hits = (
            0 if support is None else int(np.logical_and(component, support).sum())
        )
        score = float(
            area
            + 4 * center_hits
            + 8 * extent.min()
            + 6 * overlap_hits
            - 3 * border_hits
        )
        if support is not None and overlap_hits > 0:
            selected |= component
            continue
        if center_hits > 0 and border_hits < int(0.35 * max(area, 1)):
            selected |= component
            continue
        if score > fallback_score:
            fallback_score = score
            fallback_component = component
    if selected.sum() == 0:
        selected = fallback_component
    return selected


def get_largest_cc(segmentation):
    labels, num_labels = ndimage.label(segmentation)
    if num_labels == 0:
        return np.zeros_like(segmentation, dtype=bool)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    return labels == int(np.argmax(counts))


def slice_center_mask(shape):
    center_mask = np.zeros(shape, dtype=bool)
    x0 = int(round(shape[0] * 0.2))
    x1 = int(round(shape[0] * 0.8))
    y0 = int(round(shape[1] * 0.2))
    y1 = int(round(shape[1] * 0.8))
    center_mask[x0:x1, y0:y1] = True
    return center_mask


def get_body_mask(ct_hu: np.ndarray) -> np.ndarray:
    """Compute a body mask from a raw HU CT volume.

    Identifies the patient body by thresholding at -700 HU, removing small
    components, and filling holes. The scanner bed and air outside the body
    are excluded.

    Args:
        ct_hu: Raw CT volume in HU values, shape (H, W, D).

    Returns:
        Boolean mask of shape (H, W, D), True inside the body.
    """
    body_candidate = ct_hu >= -700
    body_candidate = ndimage.binary_opening(
        body_candidate, structure=np.ones((3, 3, 3), dtype=bool)
    )
    tracked_mask = np.zeros_like(body_candidate, dtype=bool)
    center_mask = slice_center_mask(body_candidate.shape[:2])
    mid_slice = body_candidate.shape[2] // 2

    prev_support = None
    for z_idx in range(mid_slice, -1, -1):
        current = select_body_components(
            body_candidate[:, :, z_idx], center_mask, prev_support
        )
        tracked_mask[:, :, z_idx] = current
        if current.any():
            prev_support = current

    prev_support = None
    for z_idx in range(mid_slice + 1, body_candidate.shape[2]):
        current = select_body_components(
            body_candidate[:, :, z_idx], center_mask, prev_support
        )
        tracked_mask[:, :, z_idx] = current
        if current.any():
            prev_support = current

    if tracked_mask.sum() == 0:
        tracked_mask = get_largest_cc(body_candidate)
    else:
        tracked_mask = ndimage.binary_fill_holes(tracked_mask)
        tracked_mask = get_largest_cc(tracked_mask)

    tracked_mask = ndimage.binary_closing(
        tracked_mask, structure=np.ones((5, 5, 3), dtype=bool)
    )
    tracked_mask = ndimage.binary_fill_holes(tracked_mask)
    tracked_mask = ndimage.binary_dilation(
        tracked_mask, structure=np.ones((3, 3, 3), dtype=bool)
    )

    return tracked_mask.astype(bool)


def apply_body_mask(vol: np.ndarray, mask: np.ndarray, fill_value: float) -> np.ndarray:
    """Zero out voxels outside the body mask.

    Args:
        vol: Volume array of shape (H, W, D).
        mask: Boolean body mask of shape (H, W, D), True inside body.
        fill_value: Value to assign to voxels outside the mask.

    Returns:
        Masked volume of same shape as vol.
    """
    out = vol.copy()
    out[~mask] = fill_value
    return out


class LoadPairToDict(Transform):
    """Load a registration pair into a dict of per-channel tensors.

    Returns a dict with keys x_ct, x_pet, y_ct, y_pet, x_label_ct,
    x_label_pet, y_label_ct, y_label_pet, each of shape (1, H, W, D).
    Keeping CT and PET as separate keys allows MONAI intensity transforms
    to be applied independently per modality.
    """

    def __init__(
        self, data_dir: Path, ignore_pet: bool = False, abdomen: bool = False
    ) -> None:
        self.data_dir = data_dir
        self.ignore_pet = ignore_pet
        self.abdomen = abdomen

    def __call__(self, data: dict) -> dict:

        if self.abdomen:
            return load_abdomen_pair_to_dict(
                self.data_dir,
                data["case_id"],
                data["tp_x"],
                data["tp_y"],
            )
        else:
            return load_pair_to_dict(
                self.data_dir,
                data["case_id"],
                data["tp_x"],
                data["tp_y"],
                ignore_pet=self.ignore_pet,
            )


def load_pair_to_dict(
    data_dir: Path, case_id: str, tp_x: str, tp_y: str, ignore_pet: bool = False
) -> dict:
    """Load and normalize one longitudinal CT/PET pair into a dict of tensors.

    Args:
        data_dir: Dataset root containing imagesTr and labelsTr.
        case_id: Case identifier, e.g. "0006".
        tp_x: Timepoint suffix of the moving scan, e.g. "00".
        tp_y: Timepoint suffix of the fixed scan, e.g. "01".

    Returns:
        Dict with keys x_ct, x_pet, y_ct, y_pet (float32, shape 1,H,W,D)
        and x_label_ct, x_label_pet, y_label_ct, y_label_pet (uint8, shape 1,H,W,D).
    """
    image_dir = data_dir / "imagesTr"
    label_dir = data_dir / "labelsTr"

    def load_vol(path: Path) -> np.ndarray:
        return nib.load(path).get_fdata().astype(np.float32)

    def load_lbl(path: Path) -> np.ndarray:
        return nib.load(path).get_fdata().astype(np.uint8)

    def t(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(arr).unsqueeze(0)

    # Load raw CT in HU (before normalization) to compute body masks
    x_ct_raw = load_vol(image_dir / f"PSMARegPSMA_{case_id}_0000_{tp_x}.nii.gz")
    y_ct_raw = load_vol(image_dir / f"PSMARegPSMA_{case_id}_0000_{tp_y}.nii.gz")

    x_mask = get_body_mask(x_ct_raw)
    y_mask = get_body_mask(y_ct_raw)

    # Apply mask before normalization:
    #   CT: fill outside with 0.5th percentile of raw HU (original remove_bed behaviour)
    #   PET: fill outside with 0.0 (SUV=0 is correct background)
    x_ct_raw = apply_body_mask(
        x_ct_raw, x_mask, fill_value=float(np.percentile(x_ct_raw, 0.5))
    )
    y_ct_raw = apply_body_mask(
        y_ct_raw, y_mask, fill_value=float(np.percentile(y_ct_raw, 0.5))
    )

    if ignore_pet:
        return {
            "x_ct": t(norm_ct(x_ct_raw)).float(),
            "y_ct": t(norm_ct(y_ct_raw)).float(),
            "x_label_ct": t(
                load_lbl(label_dir / f"PSMARegPSMA_{case_id}_0000_{tp_x}.nii.gz")
            ),
            "y_label_ct": t(
                load_lbl(label_dir / f"PSMARegPSMA_{case_id}_0000_{tp_y}.nii.gz")
            ),
            "y_body_mask": t(y_mask.astype(np.float32)).float(),
        }

    else:
        x_pet_raw = load_vol(image_dir / f"PSMARegPSMA_{case_id}_0001_{tp_x}.nii.gz")
        y_pet_raw = load_vol(image_dir / f"PSMARegPSMA_{case_id}_0001_{tp_y}.nii.gz")

        x_pet_raw = apply_body_mask(x_pet_raw, x_mask, fill_value=0.0)
        y_pet_raw = apply_body_mask(y_pet_raw, y_mask, fill_value=0.0)

        x_label_pet = load_lbl(label_dir / f"PSMARegPSMA_{case_id}_0001_{tp_x}.nii.gz")

        y_label_pet = load_lbl(label_dir / f"PSMARegPSMA_{case_id}_0001_{tp_y}.nii.gz")

        return {
            "x_ct": t(norm_ct(x_ct_raw)).float(),
            "x_pet": t(norm_pet(x_pet_raw)).float(),
            "y_ct": t(norm_ct(y_ct_raw)).float(),
            "y_pet": t(norm_pet(y_pet_raw)).float(),
            "x_label_ct": t(
                load_lbl(label_dir / f"PSMARegPSMA_{case_id}_0000_{tp_x}.nii.gz")
            ),
            "x_label_pet": t(x_label_pet),
            "y_label_ct": t(
                load_lbl(label_dir / f"PSMARegPSMA_{case_id}_0000_{tp_y}.nii.gz")
            ),
            "y_label_pet": t(y_label_pet),
            "y_body_mask": t(y_mask.astype(np.float32)).float(),
        }


def load_abdomen_pair_to_dict(
    data_dir: Path, case_id: str, tp_x: str, tp_y: str
) -> dict:

    case_id_1, case_id_2 = case_id.split("_")

    image_dir = data_dir / "imagesTr"
    label_dir = data_dir / "labelsTr"

    def load_vol(path: Path) -> np.ndarray:
        return nib.load(path).get_fdata().astype(np.float32)

    def load_lbl(path: Path) -> np.ndarray:
        return nib.load(path).get_fdata().astype(np.uint8)

    def t(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(arr).unsqueeze(0)

    # Load raw CT in HU (before normalization) to compute body masks
    x_ct_raw = load_vol(image_dir / f"PSMARegPSMA_{case_id_2}_0000_{tp_y}.nii.gz")
    y_ct_raw = load_vol(image_dir / f"PSMARegPSMA_{case_id_1}_0000_{tp_x}.nii.gz")

    x_mask = get_body_mask(x_ct_raw)
    y_mask = get_body_mask(y_ct_raw)

    # Apply mask before normalization:
    #   CT: fill outside with 0.5th percentile of raw HU (original remove_bed behaviour)
    #   PET: fill outside with 0.0 (SUV=0 is correct background)
    x_ct_raw = apply_body_mask(
        x_ct_raw, x_mask, fill_value=float(np.percentile(x_ct_raw, 0.5))
    )
    y_ct_raw = apply_body_mask(
        y_ct_raw, y_mask, fill_value=float(np.percentile(y_ct_raw, 0.5))
    )

    return {
        "x_ct": t(norm_ct(x_ct_raw)).float(),
        "y_ct": t(norm_ct(y_ct_raw)).float(),
        "x_label_ct": t(
            load_lbl(label_dir / f"PSMARegPSMA_{case_id_2}_0000_{tp_y}.nii.gz")
        ),
        "y_label_ct": t(
            load_lbl(label_dir / f"PSMARegPSMA_{case_id_1}_0000_{tp_x}.nii.gz")
        ),
        "y_body_mask": t(y_mask.astype(np.float32)).float(),
    }


def build_intensity_transform(
    ct_shift_range: Tuple[float, float],
    ct_scale_range: Tuple[float, float],
    pet_scale_range: Tuple[float, float],
    use_ct_intensity: bool,
    use_pet_intensity: bool,
) -> Compose:
    """Build MONAI intensity-only transform pipeline.

    Spatial augmentation (flip, z-crop) is handled manually in __getitem__
    so augmentation parameters can be tracked and applied to the DVF.
    This pipeline handles only intensity augmentation.

    Args:
        ct_shift_range: (min, max) additive CT shift in normalized [0,1] space.
        ct_scale_range: (min, max) CT multiplicative scale.
        pet_scale_range: (min, max) PET multiplicative scale.
        use_ct_intensity: Whether to apply CT intensity augmentation.
        use_pet_intensity: Whether to apply PET intensity augmentation.

    Returns:
        MONAI Compose transform pipeline.
    """
    # RandScaleIntensityd: output = input * (1 + factor)
    # so scale in [a, b] requires factors in [a-1, b-1]
    ct_scale_factors = (ct_scale_range[0] - 1.0, ct_scale_range[1] - 1.0)
    pet_scale_factors = (pet_scale_range[0] - 1.0, pet_scale_range[1] - 1.0)

    transforms = []

    if use_ct_intensity:
        transforms.append(
            RandShiftIntensityd(keys=["x_ct", "y_ct"], offsets=ct_shift_range, prob=1.0)
        )
        transforms.append(
            RandScaleIntensityd(
                keys=["x_ct", "y_ct"], factors=ct_scale_factors, prob=1.0
            )
        )

    if use_pet_intensity:
        transforms.append(
            RandScaleIntensityd(
                keys=["x_pet", "y_pet"], factors=pet_scale_factors, prob=1.0
            )
        )

    # Always stack channels at the end
    transforms.append(ConcatItemsd(keys=["x_ct", "x_pet"], name="x", dim=0))
    transforms.append(ConcatItemsd(keys=["y_ct", "y_pet"], name="y", dim=0))

    return Compose(transforms)


def build_val_transform() -> Compose:
    """Build the validation transform (stack only, no augmentation)."""
    return Compose(
        [
            ConcatItemsd(keys=["x_ct", "x_pet"], name="x", dim=0),
            ConcatItemsd(keys=["y_ct", "y_pet"], name="y", dim=0),
        ]
    )


def apply_flip(data: dict, keys: List[str]) -> dict:
    """Apply left-right flip (spatial axis 0 = H dimension) to specified keys.

    Args:
        data: Dict of tensors.
        keys: Keys to flip.

    Returns:
        Dict with flipped tensors.
    """
    for key in keys:
        data[key] = torch.flip(data[key], dims=[1])
    return data


def apply_z_crop(
    data: dict,
    keys: List[str],
    crop_head: int,
    crop_feet: int,
) -> dict:
    """Crop z-slices from head/feet and pad back to original size with zeros.

    Args:
        data: Dict of tensors, each of shape (C, H, W, D).
        keys: Keys to apply the crop to.
        crop_head: Slices to remove from the superior (head) end.
        crop_feet: Slices to remove from the inferior (feet) end.

    Returns:
        Dict with cropped-and-padded tensors.
    """
    if crop_head == 0 and crop_feet == 0:
        return data

    for key in keys:
        t = data[key]
        d = t.shape[-1]
        z_end = d - crop_feet if crop_feet > 0 else d
        cropped = t[..., crop_head:z_end]
        pad_head = torch.zeros(*t.shape[:-1], crop_head, dtype=t.dtype)
        pad_feet = torch.zeros(*t.shape[:-1], crop_feet, dtype=t.dtype)
        data[key] = torch.cat([pad_head, cropped, pad_feet], dim=-1)

    return data


class LoadSingleToDict(Transform):
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir

    def __call__(self, data: dict) -> dict:
        return load_single_session(self.data_dir, data["case_id"], data["tp"])


class SyntheticSourceDataset(torch_data.Dataset):
    """Single-session sources for on-the-fly synthetic pair generation.

    Each item carries the source as the FIXED image (y + y_label_*). The
    training loop generates the moving image (x) and gt DVF on GPU. x and
    x_label_* are placeholders (copies of the source) so collate keys match
    the real dataset; the loop overwrites them.
    """

    def __init__(
        self,
        cfg: config.TrainingConfig,
        source_ids: List[str],
        use_cache: bool = False,
        cache_rate: float = 1.0,
        num_workers: int = 4,
        repeat: int = 1,
        augment: bool = False,
    ) -> None:
        self.data_dir = cfg.data_dir
        self.cfg = cfg
        self.augment = augment

        all_sources = list_single_session_sources(cfg.data_dir)
        wanted = set(source_ids)
        sources = [(c, tp) for (c, tp) in all_sources if c in wanted]
        self.sources = sources * repeat

        if self.cfg.overfit:
            print("warning temp reduce size")
            self.sources = [self.sources[0]]

        data_dicts = [{"case_id": c, "tp": tp} for c, tp in self.sources]
        load_transform = Compose([LoadSingleToDict(self.data_dir)])

        if use_cache:
            print("Loading synthetic")
            self.dataset = monai_data.CacheDataset(
                data=data_dicts,
                transform=load_transform,
                cache_rate=cache_rate,
                num_workers=num_workers,
            )
        else:
            self.dataset = monai_data.Dataset(data=data_dicts, transform=load_transform)

    def __len__(self) -> int:
        return len(self.sources)

    def __getitem__(self, index: int) -> dict:
        # shallow-copy so augmentation key-reassignments do not mutate the
        # cached sample in place (CacheDataset returns a shared reference)
        s = dict(self.dataset[index])

        spatial_keys = ["ct", "pet", "label_ct", "label_pet", "body_map", "body_mask"]

        flipped = False
        crop_head = 0
        crop_feet = 0

        if self.augment:
            if self.cfg.aug_use_flip and np.random.random() < self.cfg.aug_flip_prob:
                s = apply_flip(s, spatial_keys)
                flipped = True

            if self.cfg.aug_use_z_crop:
                crop_head = int(np.random.randint(0, self.cfg.aug_max_crop_z_head + 1))
                crop_feet = int(np.random.randint(0, self.cfg.aug_max_crop_z_feet + 1))
                s = apply_z_crop(s, spatial_keys, crop_head, crop_feet)

        y = torch.cat([s["ct"], s["pet"]], dim=0)
        case_id, tp = self.sources[index]
        item = {
            "x": y.clone(),
            "y": y,
            "x_label_ct": s["label_ct"].clone(),
            "x_label_pet": s["label_pet"].clone(),
            "y_label_ct": s["label_ct"],
            "y_label_pet": s["label_pet"],
            "aug_flipped": flipped,
            "aug_crop_head": crop_head,
            "aug_crop_feet": crop_feet,
            "aug_crop_head_moving": 0,
            "aug_crop_feet_moving": 0,
            "aug_crop_head_fixed": 0,
            "aug_crop_feet_fixed": 0,
            "is_synthetic": True,
            "is_tubingen": False,
            "is_nlst": False,
            "is_abdomen": False,
            "case_id": case_id,
            "tp_x": tp,
            "tp_y": tp,
            "body_map": s["body_map"],
            "y_body_mask": s["body_mask"],
        }
        return item


class PSMARegDataset(torch_data.Dataset):
    """Dataset of consecutive longitudinal CT/PET registration pairs.

    Each item is a dict with keys:
        x, y               — (2, H, W, D) float32 (CT+PET stacked)
        x_label_ct         — (1, H, W, D)
        x_label_pet        — (1, H, W, D)
        y_label_ct         — (1, H, W, D)
        y_label_pet        — (1, H, W, D)
        aug_flipped        — bool, whether left-right flip was applied
        aug_crop_head      — int, slices removed from superior end
        aug_crop_feet      — int, slices removed from inferior end

    The aug_* keys are always present. For validation (augment=False) they
    are always False/0 so the training loop can use them unconditionally.

    Args:
        data_dir: Dataset root containing imagesTr and labelsTr.
        case_ids: Optional subset of case ids. If None, all eligible cases used.
        overfit: If set, restricts to a single patient's pair.
        use_cache: Cache all pairs in memory after first load.
        cache_rate: Fraction of dataset to cache.
        num_workers: Worker processes for cache building.
        augment: Apply random augmentation (True for train, False for val).
        aug_use_flip: Enable left-right flip augmentation.
        aug_flip_prob: Probability of applying the flip.
        aug_use_ct_intensity: Enable CT intensity shift+scale.
        aug_use_pet_intensity: Enable PET intensity scale.
        aug_use_z_crop: Enable z-axis FOV crop augmentation.
        aug_ct_shift_range: (min, max) additive CT shift in [0,1] space.
        aug_ct_scale_range: (min, max) CT multiplicative scale.
        aug_pet_scale_range: (min, max) PET multiplicative scale.
        aug_max_crop_z_head: Max z-slices removed from superior end.
        aug_max_crop_z_feet: Max z-slices removed from inferior end.
    """

    def __init__(
        self,
        cfg: config.TrainingConfig,
        case_ids: Optional[List[str]] = None,
        overfit: Optional[str] = None,
        use_cache: bool = False,
        cache_rate: float = 1.0,
        num_workers: int = 4,
        augment: bool = False,
        include_intermediate_pairs: bool = False,
    ) -> None:
        self.data_dir = cfg.data_dir

        self.cfg = cfg
        self.augment = augment

        case_timepoints = list_case_timepoints(cfg.data_dir)

        if overfit is not None:
            if overfit not in case_timepoints or len(case_timepoints[overfit]) < 2:
                raise ValueError(
                    f"Patient {overfit!r} must have at least two timepoints."
                )
            pairs = build_registration_pairs(
                case_timepoints,
                case_ids=[overfit],
                include_intermediate_pairs=include_intermediate_pairs,
            )
        else:
            pairs = build_registration_pairs(
                case_timepoints,
                case_ids=case_ids,
                include_intermediate_pairs=include_intermediate_pairs,
            )

        self.pairs = pairs

        if self.cfg.overfit:
            print("warning temp reduce size")
            # self.pairs = [self.pairs[0], self.pairs[2]]
            self.pairs = [self.pairs[1]]

        data_dicts = [
            {"case_id": case_id, "tp_x": tp_x, "tp_y": tp_y}
            for case_id, tp_x, tp_y in self.pairs
        ]

        if self.cfg.use_labels_directly:
            load_transform = Compose(
                [
                    LoadPairToDict(self.data_dir),
                    ComputeSDTChannelsd(
                        "x_label_ct", "x_sdt", cfg.label_groups, cfg.sdt_clip_vox
                    ),
                    ComputeSDTChannelsd(
                        "y_label_ct", "y_sdt", cfg.label_groups, cfg.sdt_clip_vox
                    ),
                ]
            )
        else:
            load_transform = Compose([LoadPairToDict(self.data_dir)])

        if use_cache:
            print("Loading PSMAReg")
            self.dataset = monai_data.CacheDataset(
                data=data_dicts,
                transform=load_transform,
                cache_rate=cache_rate,
                num_workers=num_workers,
            )
        else:
            self.dataset = monai_data.Dataset(data=data_dicts, transform=load_transform)

        # Intensity transform (MONAI) — spatial augmentation is done manually below
        if augment:
            self.intensity_transform = build_intensity_transform(
                ct_shift_range=cfg.aug_ct_shift_range,
                ct_scale_range=cfg.aug_ct_scale_range,
                pet_scale_range=cfg.aug_pet_scale_range,
                use_ct_intensity=cfg.aug_use_ct_intensity,
                use_pet_intensity=cfg.aug_use_pet_intensity,
            )
        else:
            self.intensity_transform = build_val_transform()

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict:
        # shallow-copy so augmentation key-reassignments do not mutate the
        # cached sample in place (CacheDataset returns a shared reference)
        data = dict(self.dataset[index])

        all_spatial_keys = [
            "x_ct",
            "x_pet",
            "y_ct",
            "y_pet",
            "x_label_ct",
            "x_label_pet",
            "y_label_ct",
            "y_label_pet",
            "y_body_mask",
        ]

        moving_keys = ["x_ct", "x_pet", "x_label_ct", "x_label_pet"]
        fixed_keys = ["y_ct", "y_pet", "y_label_ct", "y_label_pet", "y_body_mask"]

        if self.cfg.use_labels_directly:
            all_spatial_keys += ["x_sdt", "y_sdt"]

            moving_keys += ["x_sdt"]
            fixed_keys += ["y_sdt"]

        # Augmentation parameters — always initialised so training loop can
        # use them unconditionally regardless of whether augment=True/False
        flipped = False
        crop_head = 0
        crop_feet = 0
        crop_head_moving = 0
        crop_feet_moving = 0
        crop_head_fixed = 0
        crop_feet_fixed = 0

        if self.augment:
            # --- Left-right flip --------------------------------------------
            if self.cfg.aug_use_flip and np.random.random() < self.cfg.aug_flip_prob:
                data = apply_flip(data, all_spatial_keys)
                flipped = True

            # --- Z-axis FOV crop --------------------------------------------
            if self.cfg.aug_use_z_crop:
                crop_head = int(np.random.randint(0, self.cfg.aug_max_crop_z_head + 1))
                crop_feet = int(np.random.randint(0, self.cfg.aug_max_crop_z_feet + 1))
                data = apply_z_crop(data, all_spatial_keys, crop_head, crop_feet)

                # --- Asymmetric Z-axis FOV crop (independent fixed/moving) ----
            if self.cfg.aug_use_z_crop_asym:
                crop_head_moving = int(
                    np.random.randint(0, self.cfg.aug_max_crop_z_head_asym + 1)
                )
                crop_feet_moving = int(
                    np.random.randint(0, self.cfg.aug_max_crop_z_feet_asym + 1)
                )
                crop_head_fixed = int(
                    np.random.randint(0, self.cfg.aug_max_crop_z_head_asym + 1)
                )
                crop_feet_fixed = int(
                    np.random.randint(0, self.cfg.aug_max_crop_z_feet_asym + 1)
                )

                data = apply_z_crop(
                    data, moving_keys, crop_head_moving, crop_feet_moving
                )
                data = apply_z_crop(data, fixed_keys, crop_head_fixed, crop_feet_fixed)

        # Intensity augmentation (MONAI) + channel stacking
        data = self.intensity_transform(data)

        # --- SDT label channels (global switch) ----------------------------
        if self.cfg.use_labels_directly:
            """
            save_volume(
                data["y_sdt"][0],
                out_dir=Path("/home/iml/fryderyk.koegl/code/LapIRN-koegl/temp_output"),
                epoch=0,
                name="fixed_lbl_0",
            )
            save_volume(
                data["y_sdt"][1],
                out_dir=Path("/home/iml/fryderyk.koegl/code/LapIRN-koegl/temp_output"),
                epoch=0,
                name="fixed_lbl_1",
            )
            save_volume(
                data["y_sdt"][2],
                out_dir=Path("/home/iml/fryderyk.koegl/code/LapIRN-koegl/temp_output"),
                epoch=0,
                name="fixed_lbl_2",
            )
            save_volume(
                data["y_sdt"][3],
                out_dir=Path("/home/iml/fryderyk.koegl/code/LapIRN-koegl/temp_output"),
                epoch=0,
                name="fixed_lbl_3",
            )
            save_volume(
                data["y"][0],
                out_dir=Path("/home/iml/fryderyk.koegl/code/LapIRN-koegl/temp_output"),
                epoch=0,
                name="fixed",
            )
            """
            data["x"] = torch.cat([data["x"], data["x_sdt"]], dim=0)
            data["y"] = torch.cat([data["y"], data["y_sdt"]], dim=0)
            x = 0
        # Attach augmentation parameters for DVF consistency in training loop
        data["aug_flipped"] = flipped

        data["aug_crop_head"] = crop_head
        data["aug_crop_feet"] = crop_feet

        data["aug_crop_head_moving"] = crop_head_moving
        data["aug_crop_feet_moving"] = crop_feet_moving
        data["aug_crop_head_fixed"] = crop_head_fixed
        data["aug_crop_feet_fixed"] = crop_feet_fixed

        case_id, tp_x, tp_y = self.pairs[index]
        data["case_id"] = case_id
        data["tp_x"] = tp_x
        data["tp_y"] = tp_y

        data["is_synthetic"] = False
        data["is_tubingen"] = False
        data["is_nlst"] = False
        data["is_abdomen"] = False

        return data


class TubingenDataset(torch_data.Dataset):
    def __init__(
        self,
        cfg: config.TrainingConfig,
        case_ids: Optional[List[str]] = None,
        overfit: Optional[str] = None,
        use_cache: bool = False,
        cache_rate: float = 1.0,
        num_workers: int = 4,
        augment: bool = False,
    ) -> None:
        self.data_dir = cfg.data_dir

        self.cfg = cfg
        self.augment = augment

        case_timepoints = list_case_timepoints(cfg.data_dir)

        if overfit is not None:
            if overfit not in case_timepoints or len(case_timepoints[overfit]) < 2:
                raise ValueError(
                    f"Patient {overfit!r} must have at least two timepoints."
                )
            pairs = build_registration_pairs(
                case_timepoints,
                case_ids=[overfit],
            )
        else:
            pairs = build_registration_pairs(
                case_timepoints,
                case_ids=case_ids,
            )

        self.pairs = pairs

        if self.cfg.overfit:
            i = 0
            print("warning temp reduce size")
            # self.pairs = [self.pairs[0], self.pairs[2]]
            self.pairs = [self.pairs[i]]

        data_dicts = [
            {"case_id": case_id, "tp_x": tp_x, "tp_y": tp_y}
            for case_id, tp_x, tp_y in self.pairs
        ]

        if self.cfg.use_labels_directly:
            load_transform = Compose(
                [
                    LoadPairToDict(self.data_dir, ignore_pet=True),
                    ComputeSDTChannelsd(
                        "x_label_ct", "x_sdt", cfg.label_groups, cfg.sdt_clip_vox
                    ),
                    ComputeSDTChannelsd(
                        "y_label_ct", "y_sdt", cfg.label_groups, cfg.sdt_clip_vox
                    ),
                ]
            )
        else:
            load_transform = Compose([LoadPairToDict(self.data_dir, ignore_pet=True)])

        if use_cache:
            print("Loading Tubingen")
            self.dataset = monai_data.CacheDataset(
                data=data_dicts,
                transform=load_transform,
                cache_rate=cache_rate,
                num_workers=num_workers,
            )
        else:
            self.dataset = monai_data.Dataset(data=data_dicts, transform=load_transform)

        # Intensity transform (MONAI) — spatial augmentation is done manually below
        if augment:
            self.intensity_transform = build_intensity_transform(
                ct_shift_range=cfg.aug_ct_shift_range,
                ct_scale_range=cfg.aug_ct_scale_range,
                pet_scale_range=cfg.aug_pet_scale_range,
                use_ct_intensity=cfg.aug_use_ct_intensity,
                use_pet_intensity=cfg.aug_use_pet_intensity,
            )
        else:
            self.intensity_transform = build_val_transform()

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict:
        # shallow-copy so augmentation key-reassignments do not mutate the
        # cached sample in place (CacheDataset returns a shared reference)
        data = dict(self.dataset[index])

        data["x_pet"] = torch.zeros_like(data["x_ct"])
        data["y_pet"] = torch.zeros_like(data["y_ct"])
        data["x_label_pet"] = torch.zeros_like(data["x_label_ct"])
        data["y_label_pet"] = torch.zeros_like(data["y_label_ct"])

        all_spatial_keys = [
            "x_ct",
            "x_pet",
            "y_ct",
            "y_pet",
            "x_label_ct",
            "x_label_pet",
            "y_label_ct",
            "y_label_pet",
            "y_body_mask",
        ]

        moving_keys = ["x_ct", "x_pet", "x_label_ct", "x_label_pet"]
        fixed_keys = ["y_ct", "y_pet", "y_label_ct", "y_label_pet", "y_body_mask"]

        if self.cfg.use_labels_directly:
            all_spatial_keys += ["x_sdt", "y_sdt"]

            moving_keys += ["x_sdt"]
            fixed_keys += ["y_sdt"]

        # Augmentation parameters — always initialised so training loop can
        # use them unconditionally regardless of whether augment=True/False
        flipped = False
        crop_head = 0
        crop_feet = 0
        crop_head_moving = 0
        crop_feet_moving = 0
        crop_head_fixed = 0
        crop_feet_fixed = 0

        if self.augment:
            # --- Left-right flip --------------------------------------------
            if self.cfg.aug_use_flip and np.random.random() < self.cfg.aug_flip_prob:
                data = apply_flip(data, all_spatial_keys)
                flipped = True

        # Intensity augmentation (MONAI) + channel stacking
        data = self.intensity_transform(data)

        # --- SDT label channels (global switch) ----------------------------
        if self.cfg.use_labels_directly:
            data["x"] = torch.cat([data["x"], data["x_sdt"]], dim=0)
            data["y"] = torch.cat([data["y"], data["y_sdt"]], dim=0)
            x = 0
        # Attach augmentation parameters for DVF consistency in training loop
        data["aug_flipped"] = flipped

        data["aug_crop_head"] = crop_head
        data["aug_crop_feet"] = crop_feet

        data["aug_crop_head_moving"] = crop_head_moving
        data["aug_crop_feet_moving"] = crop_feet_moving
        data["aug_crop_head_fixed"] = crop_head_fixed
        data["aug_crop_feet_fixed"] = crop_feet_fixed

        case_id, tp_x, tp_y = self.pairs[index]
        data["case_id"] = case_id
        data["tp_x"] = tp_x
        data["tp_y"] = tp_y

        data["is_synthetic"] = False
        data["is_tubingen"] = True
        data["is_nlst"] = False
        data["is_abdomen"] = False

        return data


class NLSTDataset(torch_data.Dataset):
    def __init__(
        self,
        cfg: config.TrainingConfig,
        case_ids: Optional[List[str]] = None,
        overfit: Optional[str] = None,
        use_cache: bool = False,
        cache_rate: float = 1.0,
        num_workers: int = 4,
        augment: bool = False,
    ) -> None:
        self.data_dir = cfg.data_dir

        self.cfg = cfg
        self.augment = augment

        case_timepoints = list_case_timepoints(cfg.data_dir)

        if overfit is not None:
            if overfit not in case_timepoints or len(case_timepoints[overfit]) < 2:
                raise ValueError(
                    f"Patient {overfit!r} must have at least two timepoints."
                )
            pairs = build_registration_pairs(
                case_timepoints,
                case_ids=[overfit],
            )
        else:
            pairs = build_registration_pairs(
                case_timepoints,
                case_ids=case_ids,
            )

        self.pairs = pairs

        if self.cfg.overfit:
            i = 0
            print("warning temp reduce size")
            # self.pairs = [self.pairs[0], self.pairs[2]]
            self.pairs = [self.pairs[i]]

        data_dicts = [
            {"case_id": case_id, "tp_x": tp_x, "tp_y": tp_y}
            for case_id, tp_x, tp_y in self.pairs
        ]

        if self.cfg.use_labels_directly:
            load_transform = Compose(
                [
                    LoadPairToDict(self.data_dir, ignore_pet=True),
                    ComputeSDTChannelsd(
                        "x_label_ct", "x_sdt", cfg.label_groups, cfg.sdt_clip_vox
                    ),
                    ComputeSDTChannelsd(
                        "y_label_ct", "y_sdt", cfg.label_groups, cfg.sdt_clip_vox
                    ),
                ]
            )
        else:
            load_transform = Compose([LoadPairToDict(self.data_dir, ignore_pet=True)])

        if use_cache:
            print("Loading NLST")
            self.dataset = monai_data.CacheDataset(
                data=data_dicts,
                transform=load_transform,
                cache_rate=cache_rate,
                num_workers=num_workers,
            )
        else:
            self.dataset = monai_data.Dataset(data=data_dicts, transform=load_transform)

        # Intensity transform (MONAI) — spatial augmentation is done manually below
        if augment:
            self.intensity_transform = build_intensity_transform(
                ct_shift_range=cfg.aug_ct_shift_range,
                ct_scale_range=cfg.aug_ct_scale_range,
                pet_scale_range=cfg.aug_pet_scale_range,
                use_ct_intensity=cfg.aug_use_ct_intensity,
                use_pet_intensity=cfg.aug_use_pet_intensity,
            )
        else:
            self.intensity_transform = build_val_transform()

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict:
        # shallow-copy so augmentation key-reassignments do not mutate the
        # cached sample in place (CacheDataset returns a shared reference)
        data = dict(self.dataset[index])

        data["x_pet"] = torch.zeros_like(data["x_ct"])
        data["y_pet"] = torch.zeros_like(data["y_ct"])
        data["x_label_pet"] = torch.zeros_like(data["x_label_ct"])
        data["y_label_pet"] = torch.zeros_like(data["y_label_ct"])

        all_spatial_keys = [
            "x_ct",
            "x_pet",
            "y_ct",
            "y_pet",
            "x_label_ct",
            "x_label_pet",
            "y_label_ct",
            "y_label_pet",
            "y_body_mask",
        ]

        moving_keys = ["x_ct", "x_pet", "x_label_ct", "x_label_pet"]
        fixed_keys = ["y_ct", "y_pet", "y_label_ct", "y_label_pet", "y_body_mask"]

        if self.cfg.use_labels_directly:
            all_spatial_keys += ["x_sdt", "y_sdt"]

            moving_keys += ["x_sdt"]
            fixed_keys += ["y_sdt"]

        # Augmentation parameters — always initialised so training loop can
        # use them unconditionally regardless of whether augment=True/False
        flipped = False
        crop_head = 0
        crop_feet = 0
        crop_head_moving = 0
        crop_feet_moving = 0
        crop_head_fixed = 0
        crop_feet_fixed = 0

        if self.augment:
            # --- Left-right flip --------------------------------------------
            if self.cfg.aug_use_flip and np.random.random() < self.cfg.aug_flip_prob:
                data = apply_flip(data, all_spatial_keys)
                flipped = True

        # Intensity augmentation (MONAI) + channel stacking
        data = self.intensity_transform(data)

        # --- SDT label channels (global switch) ----------------------------
        if self.cfg.use_labels_directly:
            data["x"] = torch.cat([data["x"], data["x_sdt"]], dim=0)
            data["y"] = torch.cat([data["y"], data["y_sdt"]], dim=0)
            x = 0
        # Attach augmentation parameters for DVF consistency in training loop
        data["aug_flipped"] = flipped

        data["aug_crop_head"] = crop_head
        data["aug_crop_feet"] = crop_feet

        data["aug_crop_head_moving"] = crop_head_moving
        data["aug_crop_feet_moving"] = crop_feet_moving
        data["aug_crop_head_fixed"] = crop_head_fixed
        data["aug_crop_feet_fixed"] = crop_feet_fixed

        case_id, tp_x, tp_y = self.pairs[index]
        data["case_id"] = case_id
        data["tp_x"] = tp_x
        data["tp_y"] = tp_y

        data["is_synthetic"] = False
        data["is_tubingen"] = False
        data["is_nlst"] = True
        data["is_abdomen"] = False

        return data


class AbdomenDataset(torch_data.Dataset):
    def __init__(
        self,
        cfg: config.TrainingConfig,
        case_ids: Optional[List[str]] = None,
        overfit: Optional[str] = None,
        use_cache: bool = False,
        cache_rate: float = 1.0,
        num_workers: int = 4,
        augment: bool = False,
    ) -> None:
        self.data_dir = cfg.data_dir
        self.cfg = cfg
        self.augment = augment
        self.case_ids = case_ids

        self.pairs = self._build_registration_pairs()

        if self.cfg.overfit:
            i = 0
            print("warning temp reduce size")
            # self.pairs = [self.pairs[0], self.pairs[2]]
            self.pairs = [self.pairs[i]]

        data_dicts = [
            {"case_id": case_id, "tp_x": tp_x, "tp_y": tp_y}
            for case_id, tp_x, tp_y in self.pairs
        ]

        if self.cfg.use_labels_directly:
            load_transform = Compose(
                [
                    LoadPairToDict(self.data_dir, abdomen=True),
                    ComputeSDTChannelsd(
                        "x_label_ct", "x_sdt", cfg.label_groups, cfg.sdt_clip_vox
                    ),
                    ComputeSDTChannelsd(
                        "y_label_ct", "y_sdt", cfg.label_groups, cfg.sdt_clip_vox
                    ),
                ]
            )
        else:
            load_transform = Compose([LoadPairToDict(self.data_dir, abdomen=True)])

        if use_cache:
            print("Loading Abdomen")
            self.dataset = monai_data.CacheDataset(
                data=data_dicts,
                transform=load_transform,
                cache_rate=cache_rate,
                num_workers=num_workers,
            )
        else:
            self.dataset = monai_data.Dataset(data=data_dicts, transform=load_transform)

        # Intensity transform (MONAI) — spatial augmentation is done manually below
        if augment:
            self.intensity_transform = build_intensity_transform(
                ct_shift_range=cfg.aug_ct_shift_range,
                ct_scale_range=cfg.aug_ct_scale_range,
                pet_scale_range=cfg.aug_pet_scale_range,
                use_ct_intensity=cfg.aug_use_ct_intensity,
                use_pet_intensity=cfg.aug_use_pet_intensity,
            )
        else:
            self.intensity_transform = build_val_transform()

    def _build_registration_pairs(self, seed: int = 42) -> List[Tuple[str, str]]:
        n = len(self.case_ids)
        base_pairs = [(self.case_ids[i], self.case_ids[(i + 1) % n]) for i in range(n)]

        total = self.cfg.use_abdomen * n
        n_extra = total - len(base_pairs)

        used = set(base_pairs)
        all_valid = [
            (a, b)
            for a in self.case_ids
            for b in self.case_ids
            if a != b and (a, b) not in used
        ]

        generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(all_valid), generator=generator)
        extra_pairs = [all_valid[i] for i in perm[:n_extra].tolist()]

        case_pairs = base_pairs + extra_pairs

        pairs = []
        for pair in case_pairs:
            case_id = f"{pair[0]}_{pair[1]}"
            tp_x = f"{int(pair[0][1:]):02d}"
            tp_y = f"{int(pair[1][1:]):02d}"

            pairs.append((case_id, tp_x, tp_y))

        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict:
        # shallow-copy so augmentation key-reassignments do not mutate the
        # cached sample in place (CacheDataset returns a shared reference)
        data = dict(self.dataset[index])

        data["x_pet"] = torch.zeros_like(data["x_ct"])
        data["y_pet"] = torch.zeros_like(data["y_ct"])
        data["x_label_pet"] = torch.zeros_like(data["x_label_ct"])
        data["y_label_pet"] = torch.zeros_like(data["y_label_ct"])

        all_spatial_keys = [
            "x_ct",
            "x_pet",
            "y_ct",
            "y_pet",
            "x_label_ct",
            "x_label_pet",
            "y_label_ct",
            "y_label_pet",
            "y_body_mask",
        ]

        moving_keys = ["x_ct", "x_pet", "x_label_ct", "x_label_pet"]
        fixed_keys = ["y_ct", "y_pet", "y_label_ct", "y_label_pet", "y_body_mask"]

        if self.cfg.use_labels_directly:
            all_spatial_keys += ["x_sdt", "y_sdt"]

            moving_keys += ["x_sdt"]
            fixed_keys += ["y_sdt"]

        # Augmentation parameters — always initialised so training loop can
        # use them unconditionally regardless of whether augment=True/False
        flipped = False
        crop_head = 0
        crop_feet = 0
        crop_head_moving = 0
        crop_feet_moving = 0
        crop_head_fixed = 0
        crop_feet_fixed = 0

        if self.augment:
            # --- Left-right flip --------------------------------------------
            if self.cfg.aug_use_flip and np.random.random() < self.cfg.aug_flip_prob:
                data = apply_flip(data, all_spatial_keys)
                flipped = True

        # Intensity augmentation (MONAI) + channel stacking
        data = self.intensity_transform(data)

        # --- SDT label channels (global switch) ----------------------------
        if self.cfg.use_labels_directly:
            data["x"] = torch.cat([data["x"], data["x_sdt"]], dim=0)
            data["y"] = torch.cat([data["y"], data["y_sdt"]], dim=0)
            x = 0
        # Attach augmentation parameters for DVF consistency in training loop
        data["aug_flipped"] = flipped

        data["aug_crop_head"] = crop_head
        data["aug_crop_feet"] = crop_feet

        data["aug_crop_head_moving"] = crop_head_moving
        data["aug_crop_feet_moving"] = crop_feet_moving
        data["aug_crop_head_fixed"] = crop_head_fixed
        data["aug_crop_feet_fixed"] = crop_feet_fixed

        case_id, tp_x, tp_y = self.pairs[index]
        data["case_id"] = case_id
        data["tp_x"] = tp_x
        data["tp_y"] = tp_y

        data["is_synthetic"] = False
        data["is_tubingen"] = False
        data["is_nlst"] = False
        data["is_abdomen"] = True

        return data
