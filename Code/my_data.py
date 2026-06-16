import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import monai.data as monai_data
import nibabel as nib
import numpy as np
import torch
from monai.transforms import Compose, Transform
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
    """Build consecutive (case_id, tp_x, tp_y) registration pairs.

    Patients with fewer than min_timepoints timepoints are skipped. A
    patient with timepoints (a, b, c) contributes two pairs: (a, b) and
    (b, c).

    Args:
        case_timepoints: Mapping of case id to sorted timepoint list, as
            returned by list_case_timepoints.
        case_ids: Optional subset of case ids to restrict pair generation
            to. If None, all case ids in case_timepoints are considered.
        min_timepoints: Minimum number of timepoints required for a patient
            to contribute any pairs.

    Returns:
        List of (case_id, tp_x, tp_y) tuples, tp_x being the earlier and
        tp_y the later timepoint of each consecutive pair.
    """
    selected_ids = sorted(case_ids) if case_ids is not None else sorted(case_timepoints)
    pairs: List[Tuple[str, str, str]] = []
    for case_id in selected_ids:
        timepoints = case_timepoints[case_id]
        if len(timepoints) < min_timepoints:
            continue
        for tp_x, tp_y in zip(timepoints[:-1], timepoints[1:]):
            pairs.append((case_id, tp_x, tp_y))
    return pairs


def get_train_val_split(
    data_dir: Path,
    split_path: Path,
    val_fraction: float = 0.2,
    seed: int = 0,
    min_timepoints: int = 2,
) -> Tuple[List[str], List[str]]:
    """Get or create a patient-level train/val split.

    Splitting is done per patient, not per registration pair, so that all
    pairs derived from the same patient stay on the same side of the
    split. Only patients with at least min_timepoints timepoints are
    eligible. If split_path exists, the stored split is loaded as-is and
    the other arguments are ignored.

    Args:
        data_dir: Dataset root containing imagesTr and labelsTr.
        split_path: Path to a JSON file storing {"train": [...], "val": [...]}
            case ids. Created if it does not exist.
        val_fraction: Fraction of eligible patients assigned to validation
            when creating a new split.
        seed: Random seed used when creating a new split.
        min_timepoints: Minimum number of timepoints required for a patient
            to be eligible for the split.

    Returns:
        Tuple of (train_case_ids, val_case_ids).
    """
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


class LoadPair(Transform):
    """Wrap load_pair as a MONAI Transform so CacheDataset can cache its output."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir

    def __call__(
        self, data: dict
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        return load_pair(self.data_dir, data["case_id"], data["tp_x"], data["tp_y"])


def load_pair(
    data_dir: Path,
    case_id: str,
    tp_x: str,
    tp_y: str,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Load and normalize one longitudinal CT/PET registration pair.

    Args:
        data_dir: Dataset root containing imagesTr and labelsTr.
        case_id: Case identifier, e.g. "0006".
        tp_x: Timepoint suffix of the earlier scan, e.g. "00".
        tp_y: Timepoint suffix of the later scan, e.g. "01".

    Returns:
        Tuple of (x, y, x_label_ct, x_label_pet, y_label_ct, y_label_pet).
        x/y have shape (2, H, W, D), labels have shape (1, H, W, D).
    """
    image_dir = data_dir / "imagesTr"
    label_dir = data_dir / "labelsTr"

    x_vol_ct = (
        nib.load(image_dir / f"PSMARegPSMA_{case_id}_0000_{tp_x}.nii.gz")
        .get_fdata()
        .astype(np.float32)
    )
    x_vol_pet = (
        nib.load(image_dir / f"PSMARegPSMA_{case_id}_0001_{tp_x}.nii.gz")
        .get_fdata()
        .astype(np.float32)
    )
    y_vol_ct = (
        nib.load(image_dir / f"PSMARegPSMA_{case_id}_0000_{tp_y}.nii.gz")
        .get_fdata()
        .astype(np.float32)
    )
    y_vol_pet = (
        nib.load(image_dir / f"PSMARegPSMA_{case_id}_0001_{tp_y}.nii.gz")
        .get_fdata()
        .astype(np.float32)
    )

    x_label_ct = (
        nib.load(label_dir / f"PSMARegPSMA_{case_id}_0000_{tp_x}.nii.gz")
        .get_fdata()
        .astype(np.int64)
    )
    x_label_pet = (
        nib.load(label_dir / f"PSMARegPSMA_{case_id}_0001_{tp_x}.nii.gz")
        .get_fdata()
        .astype(np.int64)
    )
    y_label_ct = (
        nib.load(label_dir / f"PSMARegPSMA_{case_id}_0000_{tp_y}.nii.gz")
        .get_fdata()
        .astype(np.int64)
    )
    y_label_pet = (
        nib.load(label_dir / f"PSMARegPSMA_{case_id}_0001_{tp_y}.nii.gz")
        .get_fdata()
        .astype(np.int64)
    )

    x_vol_ct = norm_ct(x_vol_ct)
    x_vol_pet = norm_pet(x_vol_pet)
    y_vol_ct = norm_ct(y_vol_ct)
    y_vol_pet = norm_pet(y_vol_pet)

    x = torch.from_numpy(np.stack([x_vol_ct, x_vol_pet], axis=0))
    y = torch.from_numpy(np.stack([y_vol_ct, y_vol_pet], axis=0))

    x_label_ct_t = torch.from_numpy(x_label_ct).unsqueeze(0)
    x_label_pet_t = torch.from_numpy(x_label_pet).unsqueeze(0)
    y_label_ct_t = torch.from_numpy(y_label_ct).unsqueeze(0)
    y_label_pet_t = torch.from_numpy(y_label_pet).unsqueeze(0)

    return x, y, x_label_ct_t, x_label_pet_t, y_label_ct_t, y_label_pet_t


class PSMARegDataset(torch_data.Dataset):
    """Dataset of consecutive longitudinal CT/PET registration pairs.

    Each item is one (x, y) pair derived from two consecutive timepoints of
    a single patient. A patient with timepoints (a, b, c) contributes two
    pairs: (a, b) and (b, c). Patients with fewer than two timepoints are
    excluded automatically.

    Args:
        data_dir: Dataset root containing imagesTr and labelsTr.
        case_ids: Optional subset of case ids to restrict the dataset to,
            e.g. the train or val ids from get_train_val_split. If None,
            all eligible case ids found in data_dir are used.
        overfit: If set, holds the case id of a single patient assumed to
            have exactly two timepoints. The dataset is then restricted to
            that patient's single pair only, ignoring case_ids, useful for
            overfitting the training pipeline on one example.
        use_cache: If True, wrap loading in a MONAI CacheDataset so every
            pair is loaded and normalized once and kept in memory.
        cache_rate: Fraction of the dataset to cache when use_cache is True.
        num_workers: Number of worker processes used to build the cache.
    """

    def __init__(
        self,
        data_dir: Path,
        case_ids: Optional[List[str]] = None,
        overfit: Optional[str] = None,
        use_cache: bool = False,
        cache_rate: float = 1.0,
        num_workers: int = 4,
    ) -> None:
        self.data_dir = data_dir

        case_timepoints = list_case_timepoints(data_dir)

        if overfit is not None:
            if overfit not in case_timepoints or len(case_timepoints[overfit]) != 2:
                raise ValueError(
                    f"Patient {overfit!r} must have exactly two timepoints for overfit mode."
                )
            pairs = build_registration_pairs(case_timepoints, case_ids=[overfit])
        else:
            pairs = build_registration_pairs(case_timepoints, case_ids=case_ids)

        self.pairs = pairs
        self.use_cache = use_cache

        data_dicts = [
            {"case_id": case_id, "tp_x": tp_x, "tp_y": tp_y}
            for case_id, tp_x, tp_y in self.pairs
        ]

        load_transform = Compose([LoadPair(self.data_dir)])

        if self.use_cache:
            self.dataset = monai_data.CacheDataset(
                data=data_dicts,
                transform=load_transform,
                cache_rate=cache_rate,
                num_workers=num_workers,
            )
        else:
            self.dataset = monai_data.Dataset(data=data_dicts, transform=load_transform)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(
        self, index: int
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        return self.dataset[index]
