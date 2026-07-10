from typing import List

import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
import torch


def save_sdt_like(sdt: np.ndarray, ref_label_path: str, out_path: str) -> None:
    """Save one SDT channel as NIfTI using a reference label's affine/header."""
    ref = nib.load(ref_label_path)
    img = nib.Nifti1Image(sdt.astype(np.float32), ref.affine, ref.header)
    nib.save(img, out_path)


def compute_clipped_sdt(mask: np.ndarray, clip_vox: float) -> np.ndarray:
    """SDT in VOXEL units, clipped to +/- clip_vox, normalized to [-1, 1].

    Negative inside, positive outside. Empty mask -> all +1.0.
    """
    if not mask.any():
        sentinel = np.ones(mask.shape, dtype=np.float32)
        return sentinel
    inside = ndi.distance_transform_edt(mask)
    outside = ndi.distance_transform_edt(~mask)
    sdt = outside - inside
    sdt = np.clip(sdt, -clip_vox, clip_vox)
    sdt = sdt / clip_vox
    return sdt.astype(np.float32)


def build_label_channels(
    seg: np.ndarray,
    label_groups: List[List[int]],
    clip_vox: float,
) -> np.ndarray:
    """One SDT channel per group; each group is a union of label ids.

    seg: (H, W, D) integer label map.
    Returns (len(label_groups), H, W, D) float32.
    """
    channels = []
    for group in label_groups:
        mask = np.isin(seg, group)
        sdt = compute_clipped_sdt(mask, clip_vox)
        channels.append(sdt)
    stacked = np.stack(channels, axis=0)
    return stacked


def concat_sdt_channels(
    image: torch.Tensor,
    seg: torch.Tensor,
    label_groups: List[List[int]],
    clip_vox: float,
) -> torch.Tensor:
    """Concat SDT channels onto a (C, H, W, D) image using its (1, H, W, D) seg."""
    seg_np = seg.squeeze(0).cpu().numpy().astype(np.int64)
    sdt_np = build_label_channels(seg_np, label_groups, clip_vox)
    sdt = torch.from_numpy(sdt_np).to(image.dtype)
    out = torch.cat([image, sdt], dim=0)
    return out
