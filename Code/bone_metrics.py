"""Bone / rigidity evaluation metrics (journal experiment 2 and the ablations).

Each rigidity metric is close to one method's objective, so they are reported
as a panel (see journal_extension_plan.md):

    BRE (best-fit rigid residual)        -> the Kabsch objective
    local Jacobian |det J - 1|, ||J^T J - I||  -> the Staring objective
    within-bone distance distortion      -> neither

The primary endpoints are outcomes no method optimises directly: per-bone and
soft-tissue Dice / HD95, Dice of soft tissue in a band around the bones, and
%NDV inside bone, in that band, and in the body.

Geometry: bones are taken from the FIXED ground-truth labels; the field maps a
fixed voxel p to p + u(p) in the moving scan. u is the displacement the
network's warp actually applies (jacobian.unit_flow_to_voxel_flow), in mm.
"""

from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from scipy import ndimage

import hd95_official
import jacobian
import ndv_official
import synthetic
import utils

BONE_LABELS: Tuple[int, ...] = tuple(synthetic.BONE_LABEL_VALUES)
SOFT_LABELS: Tuple[int, ...] = tuple(l for l in range(1, 118) if l not in BONE_LABELS)

# a bone is analysed if it has at least this many voxels in the fixed scan
BONE_MIN_VOXELS = 50
# soft-tissue band around the bones (fixed frame), in voxels
BAND_VOXELS = 3
# a soft-tissue label enters the near-bone Dice if it has this many voxels in the band
NEAR_BONE_MIN_VOXELS = 20
# distance distortion: pairs sampled in the whole bone (every bone with at
# least BONE_MIN_VOXELS voxels, boundary layers and ribs included), the same
# for every rigidity method; measuring only eroded interiors would ignore what
# the eroded Staring term leaves unconstrained. Partial-volume effects at the
# boundary are kept small by the 5 mm minimum pair distance.
# erosion retention: bones whose eroded interior keeps at least this many voxels
INTERIOR_MIN_VOXELS = 20
N_PAIRS = 2000
SHORT_RANGE_MM = (5.0, 15.0)
LONG_RANGE_MIN_MM = 15.0
SEED = 0


def per_label_hd95(
    fixed: np.ndarray,
    moving: np.ndarray,
    warped: np.ndarray,
    spacing_mm: Tuple[float, float, float],
    label_ids: Iterable[int] = range(1, 118),
) -> Dict[int, float]:
    """HD95 per label with the official rules: a label is scored only if present
    in both the fixed and the original moving labels; non-finite values are
    dropped. The mean over all labels equals compute_average_ct_label_hd95."""
    out = {}
    for label_id in label_ids:
        fixed_mask = fixed == label_id
        if not fixed_mask.any() or not (moving == label_id).any():
            continue
        distances = hd95_official.compute_surface_distances(
            fixed_mask, warped == label_id, spacing_mm
        )
        value = hd95_official.compute_robust_hausdorff(distances, 95.0)
        if np.isfinite(value):
            out[label_id] = float(value)
    return out


def mean_over(values: Dict[int, float], labels: Iterable[int]) -> float:
    selected = [values[l] for l in labels if l in values]
    return float(np.mean(selected)) if selected else float("nan")


def dice(a: torch.Tensor, b: torch.Tensor) -> float:
    total = a.sum() + b.sum()
    return float(2.0 * (a & b).sum() / total) if total > 0 else float("nan")


def regional_ndv(
    disp_eval: torch.Tensor, masks: Dict[str, np.ndarray]
) -> Dict[str, float]:
    """Official %NDV inside each mask. disp_eval: (1, 3, H, W, D) voxel
    displacement, channel i along array axis i (the field the scorer sees)."""
    disp = disp_eval[0].detach().cpu().numpy().astype(np.float32)
    jac_dets = ndv_official.calc_jac_dets(disp + ndv_official.get_identity_grid(disp))
    out = {}
    for name, mask in masks.items():
        inner = mask[1:-1, 1:-1, 1:-1]
        if inner.sum() == 0:
            out[name] = float("nan")
            continue
        _, _, volume, _ = ndv_official.calc_measurements(jac_dets, inner)
        out[name] = float(volume / inner.sum() * 100.0)
    return out


def rigid_residual_mm(p: torch.Tensor, q: torch.Tensor) -> float:
    """RMS distance (mm) between q and the best rigid motion (rotation +
    translation, Kabsch) of p. p, q: (N, 3) physical coordinates."""
    p = p.double()
    q = q.double()
    p_c = p - p.mean(0)
    q_c = q - q.mean(0)
    u, _, vt = torch.linalg.svd(p_c.T @ q_c)
    sign = torch.sign(torch.det(vt.T @ u.T))
    correction = torch.diag(
        torch.tensor([1.0, 1.0, float(sign)], dtype=p.dtype, device=p.device)
    )
    rotation = vt.T @ correction @ u.T
    residual = q_c - p_c @ rotation.T
    return float(residual.pow(2).sum(1).mean().sqrt())


def distance_distortion(
    idx: torch.Tensor,
    mapped_mm: torch.Tensor,
    interior: torch.Tensor,
    spacing: torch.Tensor,
    generator: torch.Generator,
) -> Tuple[float, float]:
    """Mean relative change |d1 - d0| / d0 of distances between point pairs in
    one bone, for short- and long-range pairs.

    idx: (N, 3) voxel indices of the bone; mapped_mm: (H, W, D, 3) mapped
    physical position of every voxel; interior: (H, W, D) bool mask of the
    bone (the region the pairs are sampled from)."""
    device = idx.device
    n = idx.shape[0]
    pos0 = idx.double() * spacing

    def distortion(a: torch.Tensor, b: torch.Tensor) -> float:
        d0 = (pos0[a] - pos0[b]).norm(dim=1)
        d1 = (
            (mapped_mm[tuple(idx[a].T)] - mapped_mm[tuple(idx[b].T)])
            .double()
            .norm(dim=1)
        )
        return float(((d1 - d0).abs() / d0).mean()) if d0.numel() else float("nan")

    # long range: random pairs, keep those at least LONG_RANGE_MIN_MM apart
    a = torch.randint(n, (4 * N_PAIRS,), device=device, generator=generator)
    b = torch.randint(n, (4 * N_PAIRS,), device=device, generator=generator)
    keep = (pos0[a] - pos0[b]).norm(dim=1) >= LONG_RANGE_MIN_MM
    long_range = distortion(a[keep][:N_PAIRS], b[keep][:N_PAIRS])

    # short range: a random offset of SHORT_RANGE_MM length from each point,
    # rounded to the voxel grid, kept if it lands in the interior
    a = torch.randint(n, (4 * N_PAIRS,), device=device, generator=generator)
    direction = torch.randn(
        a.numel(), 3, device=device, generator=generator, dtype=torch.float64
    )
    direction = direction / direction.norm(dim=1, keepdim=True)
    length = torch.empty(a.numel(), device=device, dtype=torch.float64).uniform_(
        *SHORT_RANGE_MM, generator=generator
    )
    target = (idx[a].double() + direction * length[:, None] / spacing).round().long()
    shape = torch.tensor(interior.shape, device=device)
    inside = ((target >= 0) & (target < shape)).all(1)
    target, a = target[inside], a[inside]
    inside = interior[tuple(target.T)]
    target, a = target[inside], a[inside]
    d0 = ((idx[a] - target).double() * spacing).norm(dim=1)
    valid = (d0 >= SHORT_RANGE_MM[0]) & (d0 <= SHORT_RANGE_MM[1])
    target, a = target[valid][:N_PAIRS], a[valid][:N_PAIRS]
    if a.numel() == 0:
        return float("nan"), long_range
    pos_t = target.double() * spacing
    d0 = (pos0[a] - pos_t).norm(dim=1)
    d1 = (mapped_mm[tuple(idx[a].T)] - mapped_mm[tuple(target.T)]).double().norm(dim=1)
    short_range = float(((d1 - d0).abs() / d0).mean())
    return short_range, long_range


def stencil_leakage(bone_labels: torch.Tensor, centres: torch.Tensor) -> float:
    """Share of the voxels read by the Staring stencils
    (utils.STARING_STENCIL_OFFSETS around each centre voxel) that do not belong
    to the centre voxel's own bone, i.e. lie in soft tissue or a neighbouring
    bone.

    bone_labels: (D, H, W) bone label map (non-bone = 0); centres: (D, H, W)
    bool, the voxels the loss is evaluated on."""
    centre_labels = bone_labels[centres]
    if centre_labels.numel() == 0:
        return float("nan")
    leaked = sum(
        int((neighbour[centres] != centre_labels).sum())
        for neighbour in utils.stencil_neighbours(bone_labels)
    )
    return leaked / (len(utils.STARING_STENCIL_OFFSETS) * centre_labels.numel())


def bone_pair_metrics(
    total_unit_flow: torch.Tensor,
    disp_eval: torch.Tensor,
    fixed_i: torch.Tensor,
    warped_i: torch.Tensor,
    hd95_per_label: Dict[int, float],
    body_mask: np.ndarray,
    spacing_mm: Tuple[float, float, float],
) -> Dict[str, float]:
    """Bone / soft-tissue / rigidity metrics of one registered pair.

    total_unit_flow: (1, 3, H, W, D) network unit flow (total transform).
    disp_eval: the same field as the scorer sees it (for %NDV).
    fixed_i / warped_i: (H, W, D) fixed and warped-moving ground-truth labels.
    """
    device = fixed_i.device
    spacing = torch.tensor(spacing_mm, device=device, dtype=torch.float64)

    # --- accuracy: bone and soft tissue ---
    bones_present = [
        l for l in BONE_LABELS if int((fixed_i == l).sum()) >= BONE_MIN_VOXELS
    ]
    bone_dices = [dice(warped_i == l, fixed_i == l) for l in bones_present]
    soft_dices = [
        dice(warped_i == l, fixed_i == l) for l in SOFT_LABELS if (fixed_i == l).any()
    ]

    bone_values = torch.tensor(BONE_LABELS, device=device)
    fixed_bone = torch.isin(fixed_i, bone_values)
    fixed_bone_np = fixed_bone.cpu().numpy()
    band_np = (
        ndimage.binary_dilation(fixed_bone_np, iterations=BAND_VOXELS) & ~fixed_bone_np
    )
    band = torch.from_numpy(band_np).to(device)
    near_bone_dices = [
        dice((warped_i == l) & band, (fixed_i == l) & band)
        for l in SOFT_LABELS
        if int(((fixed_i == l) & band).sum()) >= NEAR_BONE_MIN_VOXELS
    ]

    # --- regularity by region ---
    ndv = regional_ndv(
        disp_eval, {"bone": fixed_bone_np, "band": band_np, "body": body_mask}
    )

    # --- rigidity: geometry of the applied displacement, in mm ---
    u_vox = jacobian.unit_flow_to_voxel_flow(total_unit_flow.permute(0, 2, 3, 4, 1))
    shape = fixed_i.shape
    grid = torch.stack(
        torch.meshgrid(*[torch.arange(s, device=device) for s in shape], indexing="ij"),
        dim=-1,
    )
    mapped_mm = (grid + u_vox[0]).double() * spacing  # (H, W, D, 3)

    # local Jacobian in physical units: J_p = S J_vox S^-1
    det, jac = jacobian.jacobian_matrix(u_vox)
    # jac[0]: (axis, component, Hi, Wi, Di) -> (Hi, Wi, Di, component, axis)
    j_vox = jac[0].permute(2, 3, 4, 1, 0).double()
    j_phys = j_vox * spacing[:, None] / spacing[None, :]
    strain = j_phys.transpose(-1, -2) @ j_phys - torch.eye(
        3, device=device, dtype=torch.float64
    )
    ortho = strain.flatten(-2).norm(dim=-1)
    det_dev = (det[0, 0, 1:-1, 1:-1, 1:-1].double() - 1.0).abs()
    inner = (slice(1, -1),) * 3

    # one erosion for the retention after erosion and the eroded Staring mask
    bone_map = torch.where(fixed_bone, fixed_i, torch.zeros_like(fixed_i)).float()
    eroded = utils.erode_label_map(bone_map[None, None])[0, 0] > 0

    # the Staring penalty itself (properness + orthonormality + affinity, unit
    # sub-weights as in IO) on the whole bone mask: one number per pair for the
    # quantity the Staring term minimises (Staring with erosion minimises it on
    # the eroded mask only); the three terms are logged as well
    staring_total, staring_terms = utils.enforce_rigidity_loss(
        det, jac, u_vox, (bone_map[None, None] > 0).float()
    )
    staring_energy = float(staring_total)
    staring_det, staring_ortho, staring_affine = (float(t) for t in staring_terms)

    generator = torch.Generator(device=device).manual_seed(SEED)
    bre, det_errors, ortho_errors, dd_short, dd_long = [], [], [], [], []
    n_voxels = n_interior = n_retained = 0
    for label in bones_present:
        bone = fixed_i == label
        idx = bone.nonzero()
        pos = idx.double() * spacing
        bre.append(rigid_residual_mm(pos, mapped_mm[tuple(idx.T)]))

        bone_inner = bone[inner]
        det_errors.append(float(det_dev[bone_inner].mean()))
        ortho_errors.append(float(ortho[bone_inner].mean()))

        short, long = distance_distortion(idx, mapped_mm, bone, spacing, generator)
        dd_short.append(short)
        dd_long.append(long)

        interior = bone & eroded
        n_voxels += idx.shape[0]
        n_interior += int(interior.sum())
        n_retained += int(interior.sum()) >= INTERIOR_MIN_VOXELS

    def nanmean(values: List[float]) -> float:
        values = [v for v in values if np.isfinite(v)]
        return float(np.mean(values)) if values else float("nan")

    return {
        "n_bones": len(bones_present),
        "bone_dice": nanmean(bone_dices),
        "bone_hd95": mean_over(hd95_per_label, BONE_LABELS),
        "soft_dice": nanmean(soft_dices),
        "soft_hd95": mean_over(hd95_per_label, SOFT_LABELS),
        "near_bone_soft_dice": nanmean(near_bone_dices),
        "ndv_bone_percent": ndv["bone"],
        "ndv_band_percent": ndv["band"],
        "ndv_body_percent": ndv["body"],
        # direct stencil leakage of the Staring term on these bones: 0 for
        # Kabsch (no stencil) by construction
        "stencil_leakage_staring": stencil_leakage(bone_map, fixed_bone),
        "stencil_leakage_staring_eroded": stencil_leakage(bone_map, eroded),
        "bre_mm": nanmean(bre),
        "staring_energy_bone": staring_energy,
        "staring_properness_bone": staring_det,
        "staring_orthonormality_bone": staring_ortho,
        "staring_affinity_bone": staring_affine,
        "jac_det_dev_bone": nanmean(det_errors),
        "jac_ortho_dev_bone": nanmean(ortho_errors),
        "dist_distortion_short": nanmean(dd_short),
        "dist_distortion_long": nanmean(dd_long),
        "erosion_bones_retained": n_retained / len(bones_present)
        if bones_present
        else float("nan"),
        "erosion_voxels_retained": n_interior / n_voxels if n_voxels else float("nan"),
    }
