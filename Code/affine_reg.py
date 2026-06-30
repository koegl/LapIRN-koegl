"""
Affine preregistration cache for PSMAReg LapIRN training.

Computes ANTs affine DVFs for each (case_id, tp_x, tp_y) pair and caches
them to disk. On subsequent epochs the cached DVF is loaded instead of
recomputing. Augmentation-aware helpers flip and crop the DVF consistently
with the image augmentation applied in the dataloader.

DVF convention: (H, W, D, 3), float32, voxel displacements at full resolution.
"""

from pathlib import Path
from typing import Tuple

import ants
import my_data
import nibabel as nib
import numpy as np
import torch
from config import TrainingConfig
from scipy.ndimage import zoom

CACHE_DIR = Path("/home/iml/fryderyk.koegl/code/LapIRN-koegl/affine_cache")
DOWNSAMPLE_FACTOR = 2
ANTS_TRANSFORM = "Affine"
AFF_METRIC = "mattes"
AFF_SAMPLING = 32


# ---------------------------------------------------------------------------
# Helpers expected to be available from the existing codebase
# ---------------------------------------------------------------------------
# These are imported from wherever they are defined in your project:
#   make_lowres_ants_image(ct_np, fullres_spacing, downsample_factor, ct_window)
#   preprocess_ct(ct_np, args) — or equivalent HU windowing
#   ants_affine_to_fullres_voxel_disp(affine_transform, fixed_ants, fullres_spacing)
# They are called below but not re-implemented here.
# ---------------------------------------------------------------------------
def preprocess_ct(volume):
    volume = np.asarray(volume, dtype=np.float32)
    return volume


def ants_physical_delta_to_fullres_voxel_disp(disp, direction, fullres_spacing):
    spacing = np.asarray(fullres_spacing, dtype=np.float32)
    flat = disp.reshape(-1, 3)
    index_delta = flat.dot(direction) / spacing
    return index_delta.reshape(disp.shape).astype(np.float32)


def ants_affine_to_fullres_voxel_disp(transform, reference_image, fullres_spacing):
    params = np.asarray(transform.parameters, dtype=np.float32)
    fixed_params = np.asarray(transform.fixed_parameters, dtype=np.float32)
    if params.size != 12 or fixed_params.size < 3:
        raise ValueError(
            "Expected 3D affine transform with 12 parameters and center fixed parameters."
        )
    matrix = params[:9].reshape(3, 3)
    translation = params[9:12]
    center = fixed_params[:3]
    shape = tuple(int(v) for v in reference_image.shape)
    spacing_lr = np.asarray(reference_image.spacing, dtype=np.float32)
    direction = np.asarray(reference_image.direction, dtype=np.float32)
    origin = np.asarray(reference_image.origin, dtype=np.float32)
    grids = np.meshgrid(
        np.arange(shape[0], dtype=np.float32),
        np.arange(shape[1], dtype=np.float32),
        np.arange(shape[2], dtype=np.float32),
        indexing="ij",
    )
    index = np.stack(grids, axis=-1).reshape(-1, 3)
    physical = origin + (index * spacing_lr).dot(direction.T)
    moved = (physical - center).dot(matrix.T) + center + translation
    disp = (moved - physical).reshape(shape + (3,)).astype(np.float32)
    return ants_physical_delta_to_fullres_voxel_disp(disp, direction, fullres_spacing)


def ct_window_normalize(volume, window):
    lo, hi = [float(value) for value in window]
    if hi <= lo:
        raise ValueError("Invalid CT window: {}".format(window))
    volume = np.asarray(volume, dtype=np.float32)
    out = (np.clip(volume, lo, hi) - lo) / (hi - lo)
    out[~np.isfinite(volume)] = 0.0
    return out.astype(np.float32)


def downsample_volume(volume, factor, order=1):
    if factor == 1:
        return np.asarray(volume, dtype=np.float32)
    scale = tuple(1.0 / factor for _ in range(3))
    return zoom(volume, zoom=scale, order=order).astype(np.float32)


def make_lowres_ants_image(volume, spacing, factor, ct_window):
    lowres = downsample_volume(
        ct_window_normalize(volume, ct_window), factor=factor, order=1
    )
    image = ants.from_numpy(lowres)
    image.set_spacing(tuple(float(s) * factor for s in spacing))
    image.set_origin((0.0, 0.0, 0.0))
    image.set_direction(np.diag([-1.0, -1.0, 1.0]))
    return image


def _cache_path(case_id: str, tp_x: str, tp_y: str) -> Path:
    """Return the .npy cache path for a given pair.

    Args:
        case_id: Case identifier, e.g. "0006".
        tp_x: Moving timepoint suffix, e.g. "00".
        tp_y: Fixed timepoint suffix, e.g. "01".

    Returns:
        Absolute path to the cached DVF file.
    """
    return CACHE_DIR / f"affine_{case_id}_{tp_x}_{tp_y}.npy"


def compute_affine_dvf(
    fixed_ct_path: Path,
    moving_ct_path: Path,
    make_lowres_ants_image_fn,
    preprocess_ct_fn,
    ants_affine_to_fullres_voxel_disp_fn,
    ct_window: Tuple[float, float] = (-1000.0, 1000.0),
) -> np.ndarray:

    fixed_nii = nib.load(str(fixed_ct_path))
    moving_nii = nib.load(str(moving_ct_path))

    fixed_np = fixed_nii.get_fdata(dtype=np.float32)
    moving_np = moving_nii.get_fdata(dtype=np.float32)
    fullres_spacing = tuple(float(s) for s in fixed_nii.header.get_zooms()[:3])

    # Apply bed removal before passing to ANTs
    fixed_mask = my_data.get_body_mask(fixed_np)
    moving_mask = my_data.get_body_mask(moving_np)
    fixed_np = my_data.apply_body_mask(
        fixed_np, fixed_mask, float(np.percentile(fixed_np, 0.5))
    )
    moving_np = my_data.apply_body_mask(
        moving_np, moving_mask, float(np.percentile(moving_np, 0.5))
    )

    fixed_ants = make_lowres_ants_image_fn(
        preprocess_ct_fn(fixed_np),
        fullres_spacing,
        DOWNSAMPLE_FACTOR,
        ct_window,
    )
    fixed_ants_fullres = make_lowres_ants_image_fn(
        preprocess_ct_fn(fixed_np),
        fullres_spacing,
        1,  # downsample factor 1 = full resolution
        ct_window,
    )
    moving_ants = make_lowres_ants_image_fn(
        preprocess_ct_fn(moving_np),
        fullres_spacing,
        DOWNSAMPLE_FACTOR,
        ct_window,
    )

    tx = ants.registration(
        fixed=fixed_ants,
        moving=moving_ants,
        type_of_transform=ANTS_TRANSFORM,
        aff_metric=AFF_METRIC,
        aff_sampling=AFF_SAMPLING,
        verbose=False,
    )

    # debug_dir = Path("/home/iml/fryderyk.koegl/code/LapIRN-koegl/temppp")
    # debug_dir.mkdir(parents=True, exist_ok=True)
    # ants.image_write(
    #     tx["warpedmovout"],
    #     str(debug_dir / "ants_warped.nii.gz"),
    # )
    # ants.image_write(
    #     fixed_ants,
    #     str(debug_dir / "ants_fixed.nii.gz"),
    # )

    affine_transform = ants.read_transform(tx["fwdtransforms"][0])
    dvf = ants_affine_to_fullres_voxel_disp_fn(
        affine_transform,
        fixed_ants_fullres,
        fullres_spacing=fullres_spacing,
    )

    return dvf.astype(np.float32)


def get_affine_dvf(
    case_id: str,
    tp_x: str,
    tp_y: str,
    fixed_ct_path: Path,
    moving_ct_path: Path,
    make_lowres_ants_image_fn,
    preprocess_ct_fn,
    ants_affine_to_fullres_voxel_disp_fn,
    ct_window: Tuple[float, float] = (-1000.0, 1000.0),
) -> np.ndarray:
    """Load cached affine DVF or compute and cache it if not yet available.

    Args:
        case_id: Case identifier, e.g. "0006".
        tp_x: Moving timepoint suffix, e.g. "00".
        tp_y: Fixed timepoint suffix, e.g. "01".
        fixed_ct_path: Path to the fixed CT NIfTI file.
        moving_ct_path: Path to the moving CT NIfTI file.
        make_lowres_ants_image_fn: Passed to compute_affine_dvf.
        preprocess_ct_fn: Passed to compute_affine_dvf.
        ants_affine_to_fullres_voxel_disp_fn: Passed to compute_affine_dvf.
        ct_window: HU window for preprocessing.

    Returns:
        DVF of shape (H, W, D, 3), float32, voxel displacements.
    """
    cache_path = _cache_path(case_id, tp_x, tp_y)

    if cache_path.exists():
        return np.load(str(cache_path))

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dvf = compute_affine_dvf(
        fixed_ct_path=fixed_ct_path,
        moving_ct_path=moving_ct_path,
        make_lowres_ants_image_fn=make_lowres_ants_image_fn,
        preprocess_ct_fn=preprocess_ct_fn,
        ants_affine_to_fullres_voxel_disp_fn=ants_affine_to_fullres_voxel_disp_fn,
        ct_window=ct_window,
    )
    np.save(str(cache_path), dvf)
    return dvf


# ---------------------------------------------------------------------------
# Augmentation-aware DVF transformations
# ---------------------------------------------------------------------------


def flip_dvf_lr(dvf: np.ndarray) -> np.ndarray:
    """Flip a DVF left-right, consistent with a flip on spatial axis 0 (H).

    When the image is flipped along H (left-right), the DVF must be:
      1. Reversed along the H axis
      2. The H-component of the displacement negated

    DVF convention: (H, W, D, 3) where last dim = (d_H, d_W, d_D).

    Args:
        dvf: Displacement field of shape (H, W, D, 3).

    Returns:
        Flipped DVF of same shape.
    """
    dvf_flipped = dvf[::-1, :, :, :].copy()
    dvf_flipped[..., 0] = -dvf_flipped[..., 0]
    return dvf_flipped


def crop_dvf_z(dvf: np.ndarray, crop_head: int, crop_feet: int) -> np.ndarray:
    """Crop z-slices from a DVF and pad back to original size with zeros.

    Consistent with ZAxisFOVCropd applied to the images. Removed slices are
    replaced with zero displacement (identity transform).

    DVF convention: (H, W, D, 3).

    Args:
        dvf: Displacement field of shape (H, W, D, 3).
        crop_head: Number of slices removed from the superior (head) end.
        crop_feet: Number of slices removed from the inferior (feet) end.

    Returns:
        DVF of same shape as input with cropped regions zeroed out.
    """
    if crop_head == 0 and crop_feet == 0:
        return dvf

    d = dvf.shape[2]
    z_end = d - crop_feet if crop_feet > 0 else d
    cropped = dvf[:, :, crop_head:z_end, :]

    pad_head = np.zeros((dvf.shape[0], dvf.shape[1], crop_head, 3), dtype=dvf.dtype)
    pad_feet = np.zeros((dvf.shape[0], dvf.shape[1], crop_feet, 3), dtype=dvf.dtype)
    result = np.concatenate([pad_head, cropped, pad_feet], axis=2)

    return result


def apply_augmentation_to_dvf(
    dvf: np.ndarray,
    flipped: bool,
    crop_head: int,
    crop_feet: int,
) -> np.ndarray:
    """Apply the same augmentation that was applied to the images to a DVF.

    Call this after get_affine_dvf to ensure the DVF stays consistent with
    the augmented images. The augmentation parameters (flipped, crop_head,
    crop_feet) must match those used in ZAxisFOVCropd and RandFlipd for
    this sample.

    Args:
        dvf: Displacement field of shape (H, W, D, 3).
        flipped: Whether a left-right flip was applied to the images.
        crop_head: Slices removed from the superior end.
        crop_feet: Slices removed from the inferior end.

    Returns:
        Augmentation-consistent DVF of same shape.
    """
    if flipped:
        dvf = flip_dvf_lr(dvf)
    if crop_head > 0 or crop_feet > 0:
        dvf = crop_dvf_z(dvf, crop_head, crop_feet)
    return dvf


# ---------------------------------------------------------------------------
# Apply DVF to a tensor (for warping moving image with affine before LapIRN)
# ---------------------------------------------------------------------------


def dvf_to_tensor(dvf: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert a numpy DVF to a (1, 3, H, W, D) float32 tensor.

    Args:
        dvf: Displacement field of shape (H, W, D, 3).
        device: Target device.

    Returns:
        Tensor of shape (1, 3, H, W, D).
    """
    # (H, W, D, 3) -> (3, H, W, D) -> (1, 3, H, W, D)
    dvf_tensor = torch.from_numpy(dvf).permute(3, 0, 1, 2).unsqueeze(0).float()
    return dvf_tensor.to(device)


def create_affine_flow(
    config: TrainingConfig,
    device: torch.device,
    case_id: str,
    tp_x: str,
    tp_y: str,
    aug_flipped: bool,
    aug_crop_head: int,
    aug_crop_feet: int,
) -> torch.Tensor:
    """Load images/labels from a batch and apply the cached affine DVF.

    Computes (or loads) the ANTs affine DVF for the (case_id, tp_x, tp_y)
    pair, applies the same augmentation that was applied to the images,
    converts it to a grid_sample-compatible flow (reversed channel order,
    per-component normalization for align_corners=False), and warps the
    moving image X plus its labels into the affine-aligned space.

    Args:
        batch: Dict batch from the dataloader.
        config: Training configuration.
        device: Target device.
        transform: Bilinear spatial transform for the image.
        transform_nearest: Nearest spatial transform for the labels.
        grid_full: Full-resolution unit sampling grid.

    Returns:
        Tuple (X_affine, Y, X_lbl_ct, X_lbl_pet, Y_lbl_ct, Y_lbl_pet),
        all on device, with X and its labels affine-warped.
    """

    dvf = get_affine_dvf(
        case_id=case_id,
        tp_x=tp_x,
        tp_y=tp_y,
        fixed_ct_path=config.data_dir
        / "imagesTr"
        / f"PSMARegPSMA_{case_id}_0000_{tp_y}.nii.gz",
        moving_ct_path=config.data_dir
        / "imagesTr"
        / f"PSMARegPSMA_{case_id}_0000_{tp_x}.nii.gz",
        make_lowres_ants_image_fn=make_lowres_ants_image,
        preprocess_ct_fn=preprocess_ct,
        ants_affine_to_fullres_voxel_disp_fn=ants_affine_to_fullres_voxel_disp,
    )

    dvf = apply_augmentation_to_dvf(
        dvf=dvf,
        flipped=aug_flipped,
        crop_head=aug_crop_head,
        crop_feet=aug_crop_feet,
    )

    dvf_tensor = dvf_to_tensor(dvf, device)

    # grid_sample flow last-axis order is (along D, along W, along H) — reversed
    # vs numpy axes. align_corners=False -> normalize each component by size/2.
    H, W, D = config.img_shape
    d_h = dvf_tensor[:, 0] / (H / 2.0)
    d_w = dvf_tensor[:, 1] / (W / 2.0)
    d_d = dvf_tensor[:, 2] / (D / 2.0)
    flow_affine = torch.stack([d_d, d_w, d_h], dim=1)  # (1, 3, H, W, D)

    flow_perm = flow_affine.permute(0, 2, 3, 4, 1)

    return flow_perm
