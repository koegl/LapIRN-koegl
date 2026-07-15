"""Self-contained affine + polyaffine (bone-ICP) pre-registration for one pair.

Pipeline:
    1. affine pre-reg (reuse affine_reg, same as inference script)
    2. per-bone rigid ICP between affine-warped-moving and fixed bone surfaces
    3. log-Euclidean polyaffine fusion + scaling-and-squaring -> poly residual
    4. compose affine and poly, warp the ORIGINAL moving once
    5. save debug volumes (images + labels) for each stage

No model / no LapIRN here. Config vars are defined in main().
"""

from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import affine_reg
import config
import Functions
import miccai2020_model_stage
import my_data
import nibabel as nib
import numpy as np
import scipy.linalg
import scipy.ndimage
import scipy.spatial
import synthetic
import torch
import torch.nn.functional as F
import tqdm
from config import TrainingConfig

_POLY_DVF_CACHE: Dict[str, np.ndarray] = {}
POLY_CACHE_DIR = Path("/home/iml/fryderyk.koegl/data/PSMAReg/polyaffine_cache")


def _poly_cache_path(case_id: str, cache_dir: Path, tp_x: str, tp_y: str) -> Path:
    path = cache_dir / f"poly_{case_id}_{tp_x}_{tp_y}.npy"
    return path


def _load_seg_tensor(path: Path, device: torch.device) -> torch.Tensor:
    arr = my_data.nib.load(str(path)).get_fdata().astype(np.int16)
    seg = torch.from_numpy(arr)[None, None].to(device).float()
    return seg


def affine_dvf_to_unit_flow(
    affine_dvf: np.ndarray, cfg: TrainingConfig, device: torch.device
) -> torch.Tensor:
    """Canonical (un-augmented) affine DVF (H, W, D, 3) -> unit flow, used
    only to warp the moving bones into affine space for ICP."""
    dvf_tensor = affine_reg.dvf_to_tensor(affine_dvf, device)
    h, w, d = cfg.img_shape
    d_h = dvf_tensor[:, 0] / (h / 2.0)
    d_w = dvf_tensor[:, 1] / (w / 2.0)
    d_d = dvf_tensor[:, 2] / (d / 2.0)
    flow = torch.stack([d_d, d_w, d_h], dim=1).permute(0, 2, 3, 4, 1)
    return flow


def get_polyaffine_dvf(
    case_id: str,
    tp_x: str,
    tp_y: str,
    fixed_seg_path: Path,
    moving_seg_path: Path,
    get_affine_dvf_fn: Callable[[], np.ndarray],
    cfg: TrainingConfig,
    device: torch.device,
    bone_labels: Tuple[int, ...] = synthetic.BONE_LABEL_VALUES,
    sigma: float = 8.0,
    w_bg: float = 0.01,
    ss_steps: int = 6,
    icp_max_iter: int = 30,
    icp_tol: float = 1e-3,
    min_voxels: int = 200,
    max_points: int = 3000,
) -> np.ndarray:
    """Canonical polyaffine residual DVF (H, W, D, 3), voxel units, (i, j, k)
    order. Cached in memory and on disk. `get_affine_dvf_fn` returns the
    canonical affine DVF and is only called on a cache miss."""
    mem_key = f"{case_id}_{tp_x}_{tp_y}"
    if mem_key in _POLY_DVF_CACHE:
        return _POLY_DVF_CACHE[mem_key].astype(np.float32)

    cfg = config.TrainingConfig()
    cfg.cache_dir_poly.mkdir(parents=True, exist_ok=True)

    cache_path = _poly_cache_path(case_id, cfg.cache_dir_poly, tp_x, tp_y)
    if cache_path.exists():
        dvf = np.load(str(cache_path))
        _POLY_DVF_CACHE[mem_key] = dvf.astype(np.float16)
        return dvf.astype(np.float32)

    grid_full = Functions.generate_grid_unit(cfg.img_shape)
    grid_full = (
        torch.from_numpy(np.reshape(grid_full, (1,) + grid_full.shape))
        .to(device)
        .float()
    )
    transform_nearest = miccai2020_model_stage.SpatialTransformNearest_unit().to(device)

    affine_dvf = get_affine_dvf_fn()
    flow_affine_canon = affine_dvf_to_unit_flow(affine_dvf, cfg, device)

    seg_moving = _load_seg_tensor(moving_seg_path, device)
    seg_fixed = _load_seg_tensor(fixed_seg_path, device)
    seg_moving_affine = transform_nearest(seg_moving, flow_affine_canon, grid_full)

    seg_fixed_np = seg_fixed[0, 0].round().long().cpu().numpy().astype(np.int16)
    seg_moving_affine_np = (
        seg_moving_affine[0, 0].round().long().cpu().numpy().astype(np.int16)
    )

    velocity = build_polyaffine_velocity(
        seg_fixed_np=seg_fixed_np,
        seg_moving_np=seg_moving_affine_np,
        bone_labels=bone_labels,
        sigma=sigma,
        w_bg=w_bg,
        icp_max_iter=icp_max_iter,
        icp_tol=icp_tol,
        min_voxels=min_voxels,
        max_points=max_points,
    )
    disp = integrate_svf(velocity, ss_steps, device)
    poly_dvf = disp[0].permute(1, 2, 3, 0).contiguous().cpu().numpy().astype(np.float32)

    np.save(str(cache_path), poly_dvf)
    _POLY_DVF_CACHE[mem_key] = poly_dvf.astype(np.float16)
    return poly_dvf


def create_polyaffine_flow(
    poly_dvf: np.ndarray,
    aug_flipped: bool,
    aug_crop_head: int,
    aug_crop_feet: int,
    cfg: TrainingConfig,
    device: torch.device,
) -> torch.Tensor:
    """Canonical poly DVF -> augmented unit flow, transported with the same
    augmentation machinery as the affine flow."""
    dvf_aug = affine_reg.apply_augmentation_to_dvf(
        dvf=poly_dvf,
        flipped=aug_flipped,
        crop_head=aug_crop_head,
        crop_feet=aug_crop_feet,
    )
    dvf_tensor = affine_reg.dvf_to_tensor(dvf_aug, device)
    h, w, d = cfg.img_shape
    d_h = dvf_tensor[:, 0] / (h / 2.0)
    d_w = dvf_tensor[:, 1] / (w / 2.0)
    d_d = dvf_tensor[:, 2] / (d / 2.0)
    flow = torch.stack([d_d, d_w, d_h], dim=1).permute(0, 2, 3, 4, 1)
    return flow


def load_val_pair(
    val_image_dir: Path, case_id: str, tp_x: str, tp_y: str
) -> Dict[str, torch.Tensor]:
    """Load fixed (00) + moving (01) CT+PET pair, body-masked and normalized."""

    def load_ct(tp: str) -> np.ndarray:
        path = val_image_dir / f"PSMARegPSMA_{case_id}_0000_{tp}.nii.gz"
        arr = my_data.nib.load(str(path)).get_fdata().astype(np.float32)
        return arr

    def load_pet(tp: str) -> np.ndarray:
        path = val_image_dir / f"PSMARegPSMA_{case_id}_0001_{tp}.nii.gz"
        arr = my_data.nib.load(str(path)).get_fdata().astype(np.float32)
        return arr

    x_ct_raw = load_ct(tp_x)
    y_ct_raw = load_ct(tp_y)

    x_mask = my_data.get_body_mask(x_ct_raw)
    y_mask = my_data.get_body_mask(y_ct_raw)

    x_ct_raw = my_data.apply_body_mask(
        x_ct_raw, x_mask, fill_value=float(np.percentile(x_ct_raw, 0.5))
    )
    y_ct_raw = my_data.apply_body_mask(
        y_ct_raw, y_mask, fill_value=float(np.percentile(y_ct_raw, 0.5))
    )

    x_pet_raw = my_data.apply_body_mask(load_pet(tp_x), x_mask, fill_value=0.0)
    y_pet_raw = my_data.apply_body_mask(load_pet(tp_y), y_mask, fill_value=0.0)

    def t(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(arr).unsqueeze(0)

    x = torch.cat([t(my_data.norm_ct(x_ct_raw)), t(my_data.norm_pet(x_pet_raw))], dim=0)
    y = torch.cat([t(my_data.norm_ct(y_ct_raw)), t(my_data.norm_pet(y_pet_raw))], dim=0)
    pair = {"x": x.float(), "y": y.float()}
    return pair


def load_seg(
    seg_dir: Path, seg_template: str, case_id: str, tp: str, device: torch.device
) -> torch.Tensor:
    """Load a multi-label CT segmentation as (1, 1, H, W, D) float tensor."""
    stem = seg_template.format(case_id=case_id, tp=tp)
    path = seg_dir / f"{stem}.nii"
    if not path.exists():
        path = seg_dir / f"{stem}.nii.gz"
    arr = my_data.nib.load(str(path)).get_fdata().astype(np.int16)
    seg = torch.from_numpy(arr)[None, None].to(device).float()
    return seg


def create_affine_flow(
    case_id: str,
    val_image_dir: Path,
    cfg: TrainingConfig,
    device: torch.device,
    tp_x: str,
    tp_y: str,
) -> torch.Tensor:
    """Affine pre-reg flow in the LapIRN unit-flow convention (matches
    inference script)."""
    # tp_x is moving, tp_y is fixed (matches training / get_affine_dvf docstring)
    dvf = affine_reg.get_affine_dvf(
        case_id=case_id,
        tp_x=tp_x,
        tp_y=tp_y,
        fixed_ct_path=val_image_dir / f"PSMARegPSMA_{case_id}_0000_{tp_y}.nii.gz",
        moving_ct_path=val_image_dir / f"PSMARegPSMA_{case_id}_0000_{tp_x}.nii.gz",
        make_lowres_ants_image_fn=affine_reg.make_lowres_ants_image,
        preprocess_ct_fn=affine_reg.preprocess_ct,
        ants_affine_to_fullres_voxel_disp_fn=(
            affine_reg.ants_affine_to_fullres_voxel_disp
        ),
    )
    dvf = affine_reg.apply_augmentation_to_dvf(
        dvf=dvf, flipped=False, crop_head=0, crop_feet=0
    )
    dvf_tensor = affine_reg.dvf_to_tensor(dvf, device)
    h, w, d = cfg.img_shape
    d_h = dvf_tensor[:, 0] / (h / 2.0)
    d_w = dvf_tensor[:, 1] / (w / 2.0)
    d_d = dvf_tensor[:, 2] / (d / 2.0)
    flow = torch.stack([d_d, d_w, d_h], dim=1).permute(0, 2, 3, 4, 1)
    return flow


def surface_points(
    mask: np.ndarray,
    max_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Boundary voxel coords (N, 3) in (i, j, k), optionally subsampled."""
    eroded = scipy.ndimage.binary_erosion(mask)
    surf = mask & ~eroded
    pts = np.argwhere(surf).astype(np.float32)
    if len(pts) > max_points:
        idx = rng.choice(len(pts), size=max_points, replace=False)
        pts = pts[idx]
    return pts


def procrustes(p: np.ndarray, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Rigid R, t minimizing ||R p + t - q|| (Kabsch, no scale)."""
    cp = p.mean(axis=0)
    cq = q.mean(axis=0)
    pc = p - cp
    qc = q - cq
    h = pc.T @ qc
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    diag = np.diag([1.0, 1.0, d])
    r = vt.T @ diag @ u.T
    t = cq - r @ cp
    return r, t


def rigid_icp(
    src: np.ndarray,
    dst: np.ndarray,
    max_iter: int,
    tol: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Point-to-point rigid ICP mapping src -> dst. Centroid-initialised.
    Returns R, t, final mean nearest-neighbour distance."""
    r = np.eye(3)
    t = dst.mean(axis=0) - src.mean(axis=0)
    src_cur = src + t
    tree = scipy.spatial.cKDTree(dst)
    prev_err = np.inf
    err = np.inf
    for _ in range(max_iter):
        dist, idx = tree.query(src_cur)
        matched = dst[idx]
        r_it, t_it = procrustes(src_cur, matched)
        src_cur = (r_it @ src_cur.T).T + t_it
        r = r_it @ r
        t = r_it @ t + t_it
        err = float(dist.mean())
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    return r, t, err


def rigid_log_pullback(r: np.ndarray, t: np.ndarray) -> np.ndarray:
    """log of the pullback (fixed->moving) homogeneous matrix. r, t are the
    forward (moving->fixed) rigid params."""
    forward = np.eye(4)
    forward[:3, :3] = r
    forward[:3, 3] = t
    pullback = np.linalg.inv(forward)
    log_mat = scipy.linalg.logm(pullback)
    if np.iscomplexobj(log_mat):
        log_mat = log_mat.real
    return log_mat.astype(np.float32)


def build_polyaffine_velocity(
    seg_fixed_np: np.ndarray,
    seg_moving_np: np.ndarray,
    bone_labels: Sequence[int],
    sigma: float,
    w_bg: float,
    icp_max_iter: int,
    icp_tol: float,
    min_voxels: int,
    max_points: int,
) -> np.ndarray:
    """Fused stationary velocity field (3, H, W, D) in voxel (i, j, k) order.

    Per bone: ICP -> pullback log -> affine velocity, weighted by a Gaussian
    of the fixed-space distance transform. Background anchor keeps the field
    at identity (zero velocity) far from all bones.
    """
    shape = seg_fixed_np.shape
    h, w, d = shape
    grids = np.meshgrid(np.arange(h), np.arange(w), np.arange(d), indexing="ij")
    coords = np.stack(grids).astype(np.float32)  # (3, H, W, D)
    coords_flat = coords.reshape(3, -1)

    v_num = np.zeros((3, h, w, d), dtype=np.float32)
    w_den = np.full((h, w, d), float(w_bg), dtype=np.float32)

    fixed_ids = set(np.unique(seg_fixed_np).tolist())
    moving_ids = set(np.unique(seg_moving_np).tolist())
    common = [lbl for lbl in bone_labels if lbl in fixed_ids and lbl in moving_ids]

    rng = np.random.default_rng(0)
    used = 0
    two_sigma_sq = 2.0 * sigma * sigma
    for lbl in tqdm.tqdm(common, desc="bone ICP", leave=False):
        fixed_mask = seg_fixed_np == lbl
        moving_mask = seg_moving_np == lbl
        if fixed_mask.sum() < min_voxels or moving_mask.sum() < min_voxels:
            continue

        src = surface_points(moving_mask, max_points, rng)
        dst = surface_points(fixed_mask, max_points, rng)
        r, t, err = rigid_icp(src, dst, icp_max_iter, icp_tol)
        # tqdm.tqdm.write(f"  label {lbl:3d}: icp residual {err:.2f} vox")

        log_mat = rigid_log_pullback(r, t)
        vel_lbl = (log_mat[:3, :3] @ coords_flat + log_mat[:3, 3:4]).reshape(3, h, w, d)

        dist = scipy.ndimage.distance_transform_edt(~fixed_mask)
        weight = np.exp(-(dist**2) / two_sigma_sq).astype(np.float32)

        v_num += weight[None] * vel_lbl
        w_den += weight
        used += 1

    if used == 0:
        tqdm.tqdm.write("  WARNING: no common bones -> identity polyaffine")
    velocity = (v_num / w_den[None]).astype(np.float32)
    return velocity


def identity_grid_vox(
    shape: Tuple[int, int, int], device: torch.device
) -> torch.Tensor:
    """Identity voxel-coordinate grid (1, 3, H, W, D) in (i, j, k) order."""
    h, w, d = shape
    grids = torch.meshgrid(
        torch.arange(h), torch.arange(w), torch.arange(d), indexing="ij"
    )
    grid = torch.stack(grids)[None].float().to(device)
    return grid


def warp_voxel(
    field: torch.Tensor, disp: torch.Tensor, grid: torch.Tensor
) -> torch.Tensor:
    """Sample `field` at (grid + disp). All (1, 3, H, W, D), voxel units,
    (i, j, k) order. Internal align_corners=True convention."""
    h, w, d = field.shape[2:]
    loc = grid + disp
    li = 2.0 * loc[:, 0] / (h - 1) - 1.0
    lj = 2.0 * loc[:, 1] / (w - 1) - 1.0
    lk = 2.0 * loc[:, 2] / (d - 1) - 1.0
    samp = torch.stack([lk, lj, li], dim=-1)  # grid_sample wants (x, y, z)
    out = F.grid_sample(
        field, samp, mode="bilinear", padding_mode="border", align_corners=True
    )
    return out


def integrate_svf(
    velocity: np.ndarray, n_steps: int, device: torch.device
) -> torch.Tensor:
    """Scaling-and-squaring of a stationary velocity field. Returns the
    displacement (1, 3, H, W, D), voxel units, (i, j, k) order."""
    v = torch.from_numpy(velocity).float().to(device)[None]
    grid = identity_grid_vox(velocity.shape[1:], device)
    disp = v / (2**n_steps)
    for _ in range(n_steps):
        disp = disp + warp_voxel(disp, disp, grid)
    return disp


def voxel_disp_to_unit_flow(
    disp: torch.Tensor, shape: Tuple[int, int, int]
) -> torch.Tensor:
    """(1, 3, H, W, D) voxel disp (i, j, k) -> LapIRN unit flow
    (1, H, W, D, 3), matching the affine convention."""
    h, w, d = shape
    d_h = disp[:, 0] / (h / 2.0)
    d_w = disp[:, 1] / (w / 2.0)
    d_d = disp[:, 2] / (d / 2.0)
    flow = torch.stack([d_d, d_w, d_h], dim=1).permute(0, 2, 3, 4, 1)
    return flow


def compose_flows(
    flow_affine: torch.Tensor, flow_poly: torch.Tensor, grid_full: torch.Tensor
) -> torch.Tensor:
    """Compose affine (outer) and poly (inner) unit flows: sample the affine
    grid at the poly-displaced locations. Same pattern as the inference
    script's affine-deform composition."""
    poly_grid = grid_full + flow_poly
    affine_grid = grid_full + flow_affine
    affine_grid_ch = affine_grid.permute(0, 4, 1, 2, 3)
    composed = F.grid_sample(
        affine_grid_ch,
        poly_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    ).permute(0, 2, 3, 4, 1)
    total_flow = composed - grid_full
    return total_flow


def save_volume(vol: torch.Tensor, out_dir, name: str) -> None:
    """Save channel-0 of a (1, C, H, W, D) tensor as NIfTI (identity affine)."""
    arr = vol[0, 0].detach().cpu().numpy().astype(np.float32)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.nii.gz"
    nib.save(nib.Nifti1Image(arr, np.eye(4)), str(path))


def dice_per_label(
    pred: torch.Tensor, target: torch.Tensor, labels: List[int]
) -> Dict[int, float]:
    """Hard Dice per label. pred/target: (H, W, D) long."""
    scores: Dict[int, float] = {}
    for lbl in labels:
        p = pred == lbl
        t = target == lbl
        denom = (p.sum() + t.sum()).item()
        if denom == 0:
            continue
        scores[lbl] = (2.0 * (p & t).sum()).item() / denom
    return scores


def print_dice_comparison(dice_a: Dict[int, float], dice_b: Dict[int, float]) -> None:
    labels = sorted(set(dice_a) | set(dice_b))
    print(f"{'label':>6} {'affine':>8} {'aff+poly':>9} {'delta':>8}")
    for lbl in labels:
        a = dice_a.get(lbl, float("nan"))
        b = dice_b.get(lbl, float("nan"))
        print(f"{lbl:>6} {a:8.4f} {b:9.4f} {b - a:+8.4f}")
    mean_a = float(np.nanmean(list(dice_a.values())))
    mean_b = float(np.nanmean(list(dice_b.values())))
    print(f"{'mean':>6} {mean_a:8.4f} {mean_b:9.4f} {mean_b - mean_a:+8.4f}")


def run_pair(
    case_id: str,
    tp_x: str,
    tp_y: str,
    val_image_dir: Path,
    seg_dir: Path,
    seg_template: str,
    cfg: TrainingConfig,
    device: torch.device,
    grid_full: torch.Tensor,
    transform: "miccai2020_model_stage.SpatialTransform_unit",
    transform_nearest: "miccai2020_model_stage.SpatialTransformNearest_unit",
    out_dir: Path,
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
    """Compute prereg-only per-label dice for a single (case, tp_x->tp_y) pair.

    Returns (dice_ori, dice_affine, dice_poly): no-warp vs affine-only vs
    affine+polyaffine, each mapping label id -> hard dice against the fixed
    segmentation. No network.
    """
    # --- load pair + segmentations ---
    pair = load_val_pair(val_image_dir, case_id, tp_x, tp_y)
    x = pair["x"].unsqueeze(0).to(device).float()
    y = pair["y"].unsqueeze(0).to(device).float()
    seg_moving = load_seg(seg_dir, seg_template, case_id, tp_x, device)
    seg_fixed = load_seg(seg_dir, seg_template, case_id, tp_y, device)

    # --- affine pre-reg ---
    flow_affine = create_affine_flow(case_id, val_image_dir, cfg, device, tp_x, tp_y)
    x_affine = transform(x, flow_affine, grid_full)
    seg_moving_affine = transform_nearest(seg_moving, flow_affine, grid_full)

    # --- polyaffine (cached) ---
    def seg_path(tp: str) -> Path:
        stem = seg_template.format(case_id=case_id, tp=tp)
        p = seg_dir / f"{stem}.nii"
        if not p.exists():
            p = seg_dir / f"{stem}.nii.gz"
        return p

    def affine_dvf_fn() -> np.ndarray:
        # tp_x is moving, tp_y is fixed (matches training / get_affine_dvf)
        return affine_reg.get_affine_dvf(
            case_id=case_id,
            tp_x=tp_x,
            tp_y=tp_y,
            fixed_ct_path=val_image_dir / f"PSMARegPSMA_{case_id}_0000_{tp_y}.nii.gz",
            moving_ct_path=val_image_dir / f"PSMARegPSMA_{case_id}_0000_{tp_x}.nii.gz",
            make_lowres_ants_image_fn=affine_reg.make_lowres_ants_image,
            preprocess_ct_fn=affine_reg.preprocess_ct,
            ants_affine_to_fullres_voxel_disp_fn=(
                affine_reg.ants_affine_to_fullres_voxel_disp
            ),
        )

    poly_dvf = get_polyaffine_dvf(
        case_id=case_id,
        tp_x=tp_x,
        tp_y=tp_y,
        fixed_seg_path=seg_path(tp_y),
        moving_seg_path=seg_path(tp_x),
        get_affine_dvf_fn=affine_dvf_fn,
        cfg=cfg,
        device=device,
    )
    flow_poly = create_polyaffine_flow(
        poly_dvf=poly_dvf,
        aug_flipped=False,
        aug_crop_head=0,
        aug_crop_feet=0,
        cfg=cfg,
        device=device,
    )

    # --- compose and warp the ORIGINAL moving once ---
    total_flow = compose_flows(flow_affine, flow_poly, grid_full)
    x_prereg = transform(x, total_flow, grid_full)
    seg_prereg = transform_nearest(seg_moving, total_flow, grid_full)

    # --- dice: initial (no warp) vs affine vs affine+polyaffine ---
    seg_fixed_i = seg_fixed[0, 0].round().long()
    seg_moving_i = seg_moving[0, 0].round().long()
    seg_affine_i = seg_moving_affine[0, 0].round().long()
    seg_poly_i = seg_prereg[0, 0].round().long()
    present = torch.cat(
        [
            seg_fixed_i.unique(),
            seg_moving_i.unique(),
            seg_affine_i.unique(),
            seg_poly_i.unique(),
        ]
    ).unique()
    labels_present = [int(v) for v in present.tolist() if v != 0]
    dice_ori = dice_per_label(seg_moving_i, seg_fixed_i, labels_present)
    dice_affine = dice_per_label(seg_affine_i, seg_fixed_i, labels_present)
    dice_poly = dice_per_label(seg_poly_i, seg_fixed_i, labels_present)

    # --- debug volumes (commented out for the full-dataset sweep) ---
    # save_volume(y, out_dir, "0_fixed_image")
    # save_volume(x, out_dir, "1_moving_image")
    # save_volume(x_affine, out_dir, "2_affine_moving_image")
    # save_volume(x_prereg, out_dir, "3_polyaffine_moving_image")
    # save_volume(seg_fixed, out_dir, "0_fixed_label")
    # save_volume(seg_moving, out_dir, "1_moving_label")
    # save_volume(seg_moving_affine, out_dir, "2_affine_moving_label")
    # save_volume(seg_prereg, out_dir, "3_polyaffine_moving_label")

    return dice_ori, dice_affine, dice_poly


def main() -> None:
    # --- config: all paths come from TrainingConfig so this is HPC-portable ---
    cfg = TrainingConfig()
    cfg.in_channel = 4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    val_image_dir = cfg.data_dir / "imagesTr"
    seg_dir = cfg.data_dir / "labelsTr"
    seg_template = "PSMARegPSMA_{case_id}_0000_{tp}"
    out_dir = cfg.save_dir / "polyaffine_debug"

    # polyaffine params: left at get_polyaffine_dvf defaults to match training,
    # which also uses the defaults.

    grid_full = Functions.generate_grid_unit(cfg.img_shape)
    grid_full = (
        torch.from_numpy(np.reshape(grid_full, (1,) + grid_full.shape))
        .to(device)
        .float()
    )
    transform = miccai2020_model_stage.SpatialTransform_unit().to(device)
    transform_nearest = miccai2020_model_stage.SpatialTransformNearest_unit().to(device)

    # --- enumerate the training pairs exactly like the trainer does ---
    case_timepoints = my_data.list_case_timepoints(cfg.data_dir)
    train_ids, _ = my_data.get_train_val_split(cfg.data_dir, cfg.split_path)
    pairs = my_data.build_registration_pairs(
        case_timepoints, case_ids=train_ids, include_intermediate_pairs=True
    )
    print(f"train cases: {len(train_ids)}  registration pairs: {len(pairs)}")

    # per-pair means and the whole per-label pool (for a global mean)
    rows: List[Tuple[str, str, str, float, float, float]] = []
    all_ori: List[float] = []
    all_affine: List[float] = []
    all_poly: List[float] = []
    n_poly_worse = 0

    for case_id, tp_x, tp_y in tqdm.tqdm(pairs, desc="prereg dice sweep"):
        try:
            dice_ori, dice_affine, dice_poly = run_pair(
                case_id=case_id,
                tp_x=tp_x,
                tp_y=tp_y,
                val_image_dir=val_image_dir,
                seg_dir=seg_dir,
                seg_template=seg_template,
                cfg=cfg,
                device=device,
                grid_full=grid_full,
                transform=transform,
                transform_nearest=transform_nearest,
                out_dir=out_dir / f"{case_id}_{tp_x}_{tp_y}",
            )
        except Exception as exc:  # keep the sweep alive on HPC
            tqdm.tqdm.write(f"[WARN] {case_id} {tp_x}->{tp_y} failed: {exc}")
            continue

        labels = sorted(set(dice_ori) | set(dice_affine) | set(dice_poly))
        o_vals = [dice_ori[k] for k in labels if k in dice_ori]
        a_vals = [dice_affine[k] for k in labels if k in dice_affine]
        p_vals = [dice_poly[k] for k in labels if k in dice_poly]
        mean_o = float(np.mean(o_vals)) if o_vals else float("nan")
        mean_a = float(np.mean(a_vals)) if a_vals else float("nan")
        mean_p = float(np.mean(p_vals)) if p_vals else float("nan")
        all_ori.extend(o_vals)
        all_affine.extend(a_vals)
        all_poly.extend(p_vals)
        if mean_p < mean_a:  # remember: higher hard-dice is better here
            n_poly_worse += 1
        rows.append((case_id, tp_x, tp_y, mean_o, mean_a, mean_p))
        tqdm.tqdm.write(
            f"{case_id} {tp_x}->{tp_y}  "
            f"ori={mean_o:.4f}  affine={mean_a:.4f}  poly={mean_p:.4f}  "
            f"delta(poly-affine)={mean_p - mean_a:+.4f}"
        )

    # --- summary ---
    print("\n================ SUMMARY (hard dice, higher = better) ================")
    print(f"pairs evaluated:            {len(rows)}")
    if rows:
        pair_o = np.array([r[3] for r in rows])
        pair_a = np.array([r[4] for r in rows])
        pair_p = np.array([r[5] for r in rows])
        print(f"per-pair mean  ori:         {np.nanmean(pair_o):.4f}")
        print(f"per-pair mean  affine:      {np.nanmean(pair_a):.4f}")
        print(f"per-pair mean  poly:        {np.nanmean(pair_p):.4f}")
        print(f"per-label mean ori:         {np.mean(all_ori):.4f}")
        print(f"per-label mean affine:      {np.mean(all_affine):.4f}")
        print(f"per-label mean poly:        {np.mean(all_poly):.4f}")
        print(
            f"pairs where poly < affine:  {n_poly_worse} / {len(rows)} "
            f"({100.0 * n_poly_worse / len(rows):.1f}%)"
        )
        worst = sorted(rows, key=lambda r: r[5] - r[4])[:10]
        print("\nworst poly regressions (poly - affine):")
        for case_id, tp_x, tp_y, mean_o, mean_a, mean_p in worst:
            print(
                f"  {case_id} {tp_x}->{tp_y}  "
                f"ori={mean_o:.4f}  affine={mean_a:.4f}  poly={mean_p:.4f}  "
                f"delta={mean_p - mean_a:+.4f}"
            )


if __name__ == "__main__":
    main()
