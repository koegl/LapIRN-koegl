"""Single source of truth for folding / Jacobian quantities used in training.

Everything here is differentiable torch and operates on a displacement field in
*voxel units* with shape ``(B, D, H, W, 3)``, where component ``c`` is the
displacement along spatial axis ``c`` (the identity is added on the matching
diagonal). This is the same convention the challenge scorer uses
(``ndv_official.py``).

Two folding views are provided and they are intentionally different things:

* ``non_diff_volume_loss`` / ``percent_ndv`` -- a differentiable reproduction of
  the challenge's non-diffeomorphic *volume* (``percent_ndv``). It sums the
  negative signed tetrahedron volumes over the 8 corner schemes + 2 Jstar
  schemes, exactly like ``ndv_official.calc_measurements``. Minimising the loss
  minimises the scored metric.
* ``jacobian_matrix`` -- the central-difference Jacobian (matrix + determinant),
  used by the bone-rigidity losses in ``utils``.

The old ``JacboianDet`` / ``neg_Jdet_loss`` (forward difference on the
*normalised* grid) and the central-difference count in ``utils.compute_ndv``
penalised/measured a single scheme and are superseded by this module.
"""

import torch
import torch.nn.functional as F

# 8 corner (tetrahedral) schemes: per-axis forward '+' / backward '-' difference,
# matching ndv_official's '+x+y+z' ... '-x-y-z'.
_CORNER_SCHEMES = [
    ("+", "+", "+"),
    ("+", "+", "-"),
    ("+", "-", "+"),
    ("+", "-", "-"),
    ("-", "+", "+"),
    ("-", "+", "-"),
    ("-", "-", "+"),
    ("-", "-", "-"),
]

_INNER = slice(1, -1)


def identity_grid(shape, device, dtype) -> torch.Tensor:
    """Identity coordinate grid of shape (1, D, H, W, 3), component c along axis c."""
    d, h, w = shape
    gd, gh, gw = torch.meshgrid(
        torch.arange(d, device=device, dtype=dtype),
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack([gd, gh, gw], dim=-1).unsqueeze(0)


def _diff_along(trans: torch.Tensor, dim: int, ktype: str) -> torch.Tensor:
    """Finite difference of ``trans`` (B, D, H, W, 3) along spatial ``dim`` in
    {1, 2, 3}, evaluated on the interior (all spatial axes cropped to [1:-1]).

    ktype: '0' central, '+' forward, '-' backward (see ndv_official kernels).
    """
    if dim == 1:
        plus = trans[:, 2:, _INNER, _INNER, :]
        mid = trans[:, _INNER, _INNER, _INNER, :]
        minus = trans[:, :-2, _INNER, _INNER, :]
    elif dim == 2:
        plus = trans[:, _INNER, 2:, _INNER, :]
        mid = trans[:, _INNER, _INNER, _INNER, :]
        minus = trans[:, _INNER, :-2, _INNER, :]
    else:
        plus = trans[:, _INNER, _INNER, 2:, :]
        mid = trans[:, _INNER, _INNER, _INNER, :]
        minus = trans[:, _INNER, _INNER, :-2, :]

    if ktype == "0":
        return (plus - minus) / 2
    if ktype == "+":
        return plus - mid
    return mid - minus  # "-"


def _det3(g0: torch.Tensor, g1: torch.Tensor, g2: torch.Tensor) -> torch.Tensor:
    """Determinant of the 3x3 Jacobian with rows g0, g1, g2 (each (..., 3)).

    Equals the scalar triple product g0 . (g1 x g2).
    """
    a0, a1, a2 = g0[..., 0], g0[..., 1], g0[..., 2]
    b0, b1, b2 = g1[..., 0], g1[..., 1], g1[..., 2]
    c0, c1, c2 = g2[..., 0], g2[..., 1], g2[..., 2]
    return (
        a0 * (b1 * c2 - b2 * c1)
        - a1 * (b0 * c2 - b2 * c0)
        + a2 * (b0 * c1 - b1 * c0)
    )


def _scheme_dets(trans: torch.Tensor):
    """Per-voxel det for the 10 volume schemes (8 corners + Jstar_1 + Jstar_2)."""
    dets = []

    # separable corner schemes: gradient along D/H/W with per-axis fwd/bwd diff
    for kd, kh, kw in _CORNER_SCHEMES:
        g0 = _diff_along(trans, 1, kd)
        g1 = _diff_along(trans, 2, kh)
        g2 = _diff_along(trans, 3, kw)
        dets.append(_det3(g0, g1, g2))

    c = trans[:, _INNER, _INNER, _INNER, :]
    # Jstar_1: backward diagonal diffs; planes (x,y,z) -> (D,H), (D,W), (H,W)
    dets.append(
        _det3(
            trans[:, :-2, :-2, _INNER, :] - c,  # (D, H) plane
            trans[:, :-2, _INNER, :-2, :] - c,  # (D, W) plane
            trans[:, _INNER, :-2, :-2, :] - c,  # (H, W) plane
        )
    )
    # Jstar_2: forward diagonal diffs; planes (x,y,z) -> (D,H), (H,W), (D,W)
    # (y/z planes are swapped relative to Jstar_1 -- matches ndv_official)
    dets.append(
        _det3(
            trans[:, 2:, 2:, _INNER, :] - c,  # (D, H) plane
            trans[:, _INNER, 2:, 2:, :] - c,  # (H, W) plane
            trans[:, 2:, _INNER, 2:, :] - c,  # (D, W) plane
        )
    )
    return dets


def _interior_mask(mask, flow_voxel):
    """Crop a (B, 1, D, H, W) or (B, D, H, W) mask to the interior (B, Di, Hi, Wi)."""
    if mask is None:
        return None
    if mask.dim() == 5:
        mask = mask[:, 0]
    return mask[:, _INNER, _INNER, _INNER]


def non_diff_volume_loss(
    flow_voxel: torch.Tensor,
    grid=None,  # accepted for drop-in compatibility with the old loss_Jdet
    mask: torch.Tensor = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Differentiable non-diffeomorphic volume, in the units of ``percent_ndv``.

    Reproduces ``ndv_official`` non_diff_volume / total_voxels * 100 using
    ``min(det, 0) = -relu(-det)`` so gradients flow to exactly the folded
    voxels/schemes. Minimising this minimises the scored NDV.

    Args:
        flow_voxel: (B, D, H, W, 3) displacement in voxel units.
        grid: ignored (kept so it can replace the old ``loss_Jdet`` in place).
        mask: optional (B, 1, D, H, W) body mask; if given, folding is only
            counted/normalised inside it (matching the scorer's body mask).
    """
    trans = flow_voxel + identity_grid(
        flow_voxel.shape[1:4], flow_voxel.device, flow_voxel.dtype
    )
    m = _interior_mask(mask, flow_voxel)

    non_diff_volume = flow_voxel.new_zeros(())
    for det in _scheme_dets(trans):
        neg = torch.relu(-det)  # -min(det, 0)
        if m is not None:
            neg = neg * m
        non_diff_volume = non_diff_volume + neg.sum()
    non_diff_volume = non_diff_volume * (0.5 / 6.0)

    if m is not None:
        total = m.sum()
    else:
        b, d, h, w = flow_voxel.shape[0], *flow_voxel.shape[1:4]
        total = flow_voxel.new_tensor(b * (d - 2) * (h - 2) * (w - 2))
    return non_diff_volume / (total + eps) * 100.0


def percent_ndv(flow_voxel: torch.Tensor, mask: torch.Tensor = None) -> float:
    """Scalar ``percent_ndv`` for logging (same computation as the loss)."""
    with torch.no_grad():
        return float(non_diff_volume_loss(flow_voxel, mask=mask).item())


def jacobian_matrix(flow_voxel: torch.Tensor):
    """Central-difference Jacobian of the deformation.

    Args:
        flow_voxel: (B, D, H, W, 3) displacement in voxel units.

    Returns:
        det: (B, 1, D, H, W) Jacobian determinant, boundary padded with 1.0.
        jac: (B, 3, 3, Di, Hi, Wi) Jacobian J = I + du/dx on the interior,
            indexed [:, axis, component].
    """
    trans = flow_voxel + identity_grid(
        flow_voxel.shape[1:4], flow_voxel.device, flow_voxel.dtype
    )
    g0 = _diff_along(trans, 1, "0")
    g1 = _diff_along(trans, 2, "0")
    g2 = _diff_along(trans, 3, "0")

    det = _det3(g0, g1, g2)  # (B, Di, Hi, Wi)
    det = F.pad(det, (1, 1, 1, 1, 1, 1), value=1.0).unsqueeze(1)

    # (B, axis, Di, Hi, Wi, component) -> (B, axis, component, Di, Hi, Wi)
    jac = torch.stack([g0, g1, g2], dim=1).permute(0, 1, 5, 2, 3, 4)
    return det, jac
