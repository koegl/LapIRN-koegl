import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import config
import monai.data as monai_data
import nibabel as nib
import numpy as np
import torch
from monai.transforms import (
    Compose,
    ConcatItemsd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Transform,
)
from scipy import ndimage
from torch.utils import data as torch_data


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
) -> List[Tuple[str, str, str]]:
    """Build consecutive (case_id, tp_x, tp_y) registration pairs."""
    selected_ids = sorted(case_ids) if case_ids is not None else sorted(case_timepoints)
    pairs: List[Tuple[str, str, str]] = []
    for case_id in selected_ids:
        tps = case_timepoints[case_id]
        if len(tps) < min_timepoints:
            continue
        for tp_x, tp_y in zip(tps[:-1], tps[1:]):
            pairs.append((case_id, tp_x, tp_y))
    return pairs


def get_train_val_split(
    data_dir: Path,
    split_path: Path,
    val_fraction: float = 0.2,
    seed: int = 0,
    min_timepoints: int = 2,
) -> Tuple[List[str], List[str]]:
    """Get or create a patient-level train/val split."""
    if split_path.exists():
        with open(split_path, "r") as f:
            split = json.load(f)
        return split["train"], split["val"]

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
        json.dump({"train": train_ids, "val": val_ids}, f, indent=2)

    return train_ids, val_ids


class LoadPairToDict(Transform):
    """Load a registration pair into a dict of per-channel tensors.

    Returns a dict with keys x_ct, x_pet, y_ct, y_pet, x_label_ct,
    x_label_pet, y_label_ct, y_label_pet, each of shape (1, H, W, D).
    Keeping CT and PET as separate keys allows MONAI intensity transforms
    to be applied independently per modality.
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir

    def __call__(self, data: dict) -> dict:
        return load_pair_to_dict(
            self.data_dir, data["case_id"], data["tp_x"], data["tp_y"]
        )


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


def load_pair_to_dict(data_dir: Path, case_id: str, tp_x: str, tp_y: str) -> dict:
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

    x_pet_raw = load_vol(image_dir / f"PSMARegPSMA_{case_id}_0001_{tp_x}.nii.gz")
    y_pet_raw = load_vol(image_dir / f"PSMARegPSMA_{case_id}_0001_{tp_y}.nii.gz")

    x_pet_raw = apply_body_mask(x_pet_raw, x_mask, fill_value=0.0)
    y_pet_raw = apply_body_mask(y_pet_raw, y_mask, fill_value=0.0)

    return {
        "x_ct": t(norm_ct(x_ct_raw)).float(),
        "x_pet": t(norm_pet(x_pet_raw)).float(),
        "y_ct": t(norm_ct(y_ct_raw)).float(),
        "y_pet": t(norm_pet(y_pet_raw)).float(),
        "x_label_ct": t(
            load_lbl(label_dir / f"PSMARegPSMA_{case_id}_0000_{tp_x}.nii.gz")
        ),
        "x_label_pet": t(
            load_lbl(label_dir / f"PSMARegPSMA_{case_id}_0001_{tp_x}.nii.gz")
        ),
        "y_label_ct": t(
            load_lbl(label_dir / f"PSMARegPSMA_{case_id}_0000_{tp_y}.nii.gz")
        ),
        "y_label_pet": t(
            load_lbl(label_dir / f"PSMARegPSMA_{case_id}_0001_{tp_y}.nii.gz")
        ),
    }


class ZAxisFOVCropd(Transform):
    """Randomly remove z-slices from head/feet ends and pad back to original size.

    Simulates FOV variation between longitudinal scans. Crop amounts are
    sampled once per call and applied identically to all specified keys.
    Removed slices are replaced with zeros (background).

    Args:
        keys: Dict keys to apply the transform to.
        max_crop_head: Maximum slices to remove from the superior (head) end.
        max_crop_feet: Maximum slices to remove from the inferior (feet) end.
    """

    def __init__(
        self, keys: List[str], max_crop_head: int = 40, max_crop_feet: int = 40
    ) -> None:
        self.keys = keys
        self.max_crop_head = max_crop_head
        self.max_crop_feet = max_crop_feet

    def __call__(self, data: dict) -> dict:
        crop_head = int(np.random.randint(0, self.max_crop_head + 1))
        crop_feet = int(np.random.randint(0, self.max_crop_feet + 1))

        if crop_head == 0 and crop_feet == 0:
            return data

        d = data[self.keys[0]].shape[-1]
        z_end = d - crop_feet if crop_feet > 0 else d

        for key in self.keys:
            t = data[key]
            cropped = t[..., crop_head:z_end]
            pad_head = torch.zeros(*t.shape[:-1], crop_head, dtype=t.dtype)
            pad_feet = torch.zeros(*t.shape[:-1], crop_feet, dtype=t.dtype)
            data[key] = torch.cat([pad_head, cropped, pad_feet], dim=-1)

        return data


def build_augmentation_transform(
    flip_prob: float,
    ct_shift_range: Tuple[float, float],
    ct_scale_range: Tuple[float, float],
    pet_scale_range: Tuple[float, float],
    max_crop_z_head: int,
    max_crop_z_feet: int,
    use_flip: bool = True,
    use_ct_intensity: bool = True,
    use_pet_intensity: bool = True,
    use_z_crop: bool = True,
) -> Compose:
    """Build the MONAI augmentation pipeline for training.

    Spatial transforms (flip, z-crop) are applied to all keys consistently.
    Intensity transforms are applied per modality:
      - CT (x_ct, y_ct): independent shift + scale for x and y
      - PET (x_pet, y_pet): independent scale only (no shift — SUV=0 must stay 0)

    ConcatItemsd at the end stacks x_ct+x_pet -> x and y_ct+y_pet -> y,
    giving back (2, H, W, D) tensors as expected by the model.

    Args:
        flip_prob: Probability of left-right flip.
        ct_shift_range: (min, max) additive CT shift in normalized [0,1] space.
        ct_scale_range: (min, max) CT multiplicative scale.
        pet_scale_range: (min, max) PET multiplicative scale.
        max_crop_z_head: Max z-slices to remove from superior end.
        max_crop_z_feet: Max z-slices to remove from inferior end.

    Returns:
        MONAI Compose transform pipeline.
    """
    all_spatial_keys = [
        "x_ct",
        "x_pet",
        "y_ct",
        "y_pet",
        "x_label_ct",
        "x_label_pet",
        "y_label_ct",
        "y_label_pet",
    ]

    # RandScaleIntensityd applies output = input * (1 + factor)
    # so to get scale in [a, b] we need factors in [a-1, b-1]
    ct_scale_factors = (ct_scale_range[0] - 1.0, ct_scale_range[1] - 1.0)
    pet_scale_factors = (pet_scale_range[0] - 1.0, pet_scale_range[1] - 1.0)

    transforms = []

    if use_flip:
        transforms.append(
            RandFlipd(keys=all_spatial_keys, prob=flip_prob, spatial_axis=0)
        )

    if use_z_crop:
        transforms.append(
            ZAxisFOVCropd(
                keys=all_spatial_keys,
                max_crop_head=max_crop_z_head,
                max_crop_feet=max_crop_z_feet,
            )
        )

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

    # Always stack at the end
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


class PSMARegDataset(torch_data.Dataset):
    """Dataset of consecutive longitudinal CT/PET registration pairs.

    Each item is a dict with keys:
        x, y               — (2, H, W, D) float32 (CT+PET stacked)
        x_label_ct         — (1, H, W, D)
        x_label_pet        — (1, H, W, D)
        y_label_ct         — (1, H, W, D)
        y_label_pet        — (1, H, W, D)

    Args:
        data_dir: Dataset root containing imagesTr and labelsTr.
        case_ids: Optional subset of case ids. If None, all eligible cases are used.
        overfit: If set, restricts to a single patient's pair for pipeline debugging.
        use_cache: Cache all pairs in memory after first load.
        cache_rate: Fraction of dataset to cache.
        num_workers: Worker processes for cache building.
        augment: Apply random augmentation (True for train, False for val).
        aug_flip_prob: Probability of left-right flip.
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
        cache_rate: float = 1.0,
        num_workers: int = 4,
        augment: bool = False,
    ) -> None:
        self.data_dir = cfg.data_dir

        case_timepoints = list_case_timepoints(cfg.data_dir)

        if overfit is not None:
            if overfit not in case_timepoints or len(case_timepoints[overfit]) < 2:
                raise ValueError(
                    f"Patient {overfit!r} must have at least two timepoints."
                )
            pairs = build_registration_pairs(case_timepoints, case_ids=[overfit])
        else:
            pairs = build_registration_pairs(case_timepoints, case_ids=case_ids)

        self.pairs = pairs

        data_dicts = [
            {"case_id": case_id, "tp_x": tp_x, "tp_y": tp_y}
            for case_id, tp_x, tp_y in pairs
        ]

        load_transform = Compose([LoadPairToDict(self.data_dir)])

        if cfg.use_cache_train:
            self.dataset = monai_data.CacheDataset(
                data=data_dicts,
                transform=load_transform,
                cache_rate=cache_rate,
                num_workers=num_workers,
            )
        else:
            self.dataset = monai_data.Dataset(data=data_dicts, transform=load_transform)

        self.post_transform = (
            build_augmentation_transform(
                flip_prob=cfg.aug_flip_prob,
                ct_shift_range=cfg.aug_ct_shift_range,
                ct_scale_range=cfg.aug_ct_scale_range,
                pet_scale_range=cfg.aug_pet_scale_range,
                max_crop_z_head=cfg.aug_max_crop_z_head,
                max_crop_z_feet=cfg.aug_max_crop_z_feet,
                use_flip=cfg.aug_use_flip,
                use_ct_intensity=cfg.aug_use_ct_intensity,
                use_pet_intensity=cfg.aug_use_pet_intensity,
                use_z_crop=cfg.aug_use_z_crop,
            )
            if augment
            else build_val_transform()
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict:
        return self.post_transform(self.dataset[index])
