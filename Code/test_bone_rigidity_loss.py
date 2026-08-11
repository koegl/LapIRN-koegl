"""Behavioural test of the bone-rigidity loss on a real CT.

`utils.enforce_rigidity_loss` sums three terms over a bone mask:
    det    -- (det J - 1)^2                  (properness)
    ortho  -- ||J^T J - I||_F^2              (no shear / anisotropic scaling)
    affine -- ||d^2 u||^2                    (Jacobian constant => one transform)

Rigid motion must give zero on all three. This script feeds the loss fields
whose correct answer is known analytically, then fields built from the real
bone segmentation, and reports where the number comes from.

The tests are grouped:

  A. Global linear fields -- closed-form expected values. These validate the
     implementation itself. Translation / rotation / rigid must be ~0; scaling
     and shear must be positive. Shear is the interesting one: det stays
     exactly 1, so it is caught only by ortho.

  B. Real-mask fields -- the cases where the loss is supposed to be satisfied
     but where the mask boundary might make it fire anyway:
       B1  field exactly zero inside bone, non-zero outside
       B2  each bone label given its OWN rigid motion, soft tissue blended
     Under the intended semantics ("each bone moves rigidly") both should be
     ~0. Any large value is boundary leakage, not a violated constraint.

  C. Diagnostics -- how much of the loss lives in the boundary shell (erosion
     sweep), how much of the gradient lands outside bone (leakage in the same
     currency as the wandb gradient cosines), and what fraction of bone voxels
     are boundary voxels at all.

Every finite-difference stencil in the loss reaches one voxel beyond the centre
voxel, but the mask is applied only at the centre. So a bone voxel adjacent to
soft tissue is scored using soft-tissue displacements. Ribs and cortical shells
are a few voxels thick, so that shell can be most of the mask -- section C
measures whether it is.

Usage:
    python test_bone_rigidity_loss.py
    python test_bone_rigidity_loss.py --ct <path> --labels <path> --size 192
    python test_bone_rigidity_loss.py --size 0        # whole volume (slow, RAM)
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "Code"))

import jacobian  # noqa: E402
import synthetic  # noqa: E402
from utils import affine_loss, masked_jac_det_loss, orthonormality_loss  # noqa: E402

DEFAULT_CT = (
    "/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/imagesTr/"
    "PSMARegPSMA_0002_0000_00.nii.gz"
)
DEFAULT_LABELS = (
    "/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/labelsTr/"
    "PSMARegPSMA_0002_0000_00.nii.gz"
)

TOL = 1e-4  # what counts as "satisfied" for an exactly-rigid field


# --------------------------------------------------------------------------
# loss wrapper
# --------------------------------------------------------------------------
def rigidity_terms(flow: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    """flow: (1,D,H,W,3) voxel units. mask: (1,1,D,H,W) float. Returns the split."""
    jac_det, jac = jacobian.jacobian_matrix(flow)
    det = masked_jac_det_loss(jac_det, mask)
    ortho = orthonormality_loss(jac, mask)
    aff = affine_loss(flow, mask)
    return {
        "det": float(det),
        "ortho": float(ortho),
        "affine": float(aff),
        "total": float(det + ortho + aff),
    }


# --------------------------------------------------------------------------
# field constructors (voxel units, component c along axis c)
# --------------------------------------------------------------------------
def voxel_coords(shape, device) -> torch.Tensor:
    """(1,D,H,W,3) grid of voxel indices."""
    return jacobian.identity_grid(shape, device, torch.float64)


def linear_field(shape, A: np.ndarray, t: np.ndarray, device) -> torch.Tensor:
    """u(x) = (A - I)(x - centre) + t, i.e. the deformation x -> A(x-c)+c+t."""
    x = voxel_coords(shape, device)
    centre = torch.tensor(
        [(s - 1) / 2 for s in shape], dtype=torch.float64, device=device
    )
    m = torch.tensor(A - np.eye(3), dtype=torch.float64, device=device)
    off = torch.tensor(t, dtype=torch.float64, device=device)
    return torch.einsum("ij,bdhwj->bdhwi", m, x - centre) + off


def rotation(axis: str, deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _blur(field: torch.Tensor, sigma: float) -> torch.Tensor:
    """Simple separable Gaussian; explicit loop, no shape trickery."""
    radius = max(1, int(3 * sigma))
    k = torch.arange(-radius, radius + 1, dtype=field.dtype, device=field.device)
    k = torch.exp(-(k**2) / (2 * sigma**2))
    k = k / k.sum()
    out = field.permute(0, 4, 1, 2, 3).contiguous()  # (1,3,D,H,W)
    for dim in (2, 3, 4):
        shape = [1, 1, 1, 1, 1]
        shape[dim] = k.numel()
        weight = k.view(shape).expand(3, 1, *shape[2:]).contiguous()
        pad = [0, 0, 0, 0, 0, 0]
        pad[2 * (4 - dim)] = radius
        pad[2 * (4 - dim) + 1] = radius
        out = torch.nn.functional.conv3d(
            torch.nn.functional.pad(out, pad, mode="replicate"), weight, groups=3
        )
    return out.permute(0, 2, 3, 4, 1).contiguous()


def per_bone_rigid_field(
    labels: torch.Tensor, bone_values, shape, device, max_deg=3.0, max_shift=2.0, seed=0
) -> torch.Tensor:
    """Each bone label gets its own small rigid motion; soft tissue is a smooth
    blend of them. Bone voxels keep their exact rigid displacement.

    This is the field the loss is *supposed* to accept: rigid within every bone.
    """
    rng = np.random.default_rng(seed)
    x = voxel_coords(shape, device)
    flow = torch.zeros(x.shape, dtype=torch.float64, device=device)
    present = [v for v in bone_values if bool((labels == v).any())]

    for value in present:
        sel = labels == value  # (D,H,W)
        idx = torch.nonzero(sel, as_tuple=False).to(torch.float64)
        centre = idx.mean(dim=0)
        axis = "xyz"[int(rng.integers(3))]
        rot = rotation(axis, float(rng.uniform(-max_deg, max_deg)))
        shift = rng.uniform(-max_shift, max_shift, size=3)
        m = torch.tensor(rot - np.eye(3), dtype=torch.float64, device=device)
        off = torch.tensor(shift, dtype=torch.float64, device=device)
        local = torch.einsum("ij,bdhwj->bdhwi", m, x - centre) + off
        flow[0][sel] = local[0][sel]

    # continuous soft tissue: blur everything, then restore the exact rigid
    # values inside bone. The field is then C0 with a kink at the interface --
    # which is what a physically correct bone/soft-tissue field looks like.
    bone = torch.isin(labels, torch.tensor(list(bone_values), device=device))
    blurred = _blur(flow, sigma=3.0)
    blurred[0][bone] = flow[0][bone]
    return blurred, present


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def report(name: str, terms: dict[str, float], expect: str = "") -> None:
    verdict = ""
    if expect == "zero":
        verdict = "  PASS" if terms["total"] < TOL else "  <-- FAIL (should be ~0)"
    elif expect == "nonzero":
        verdict = "  PASS" if terms["total"] > TOL else "  <-- FAIL (should be > 0)"
    print(
        f"  {name:<44s} det={terms['det']:>11.3e}  ortho={terms['ortho']:>11.3e}  "
        f"affine={terms['affine']:>11.3e}  total={terms['total']:>11.3e}{verdict}"
    )


def analytic(name: str, got: float, want: float) -> None:
    ok = "PASS" if abs(got - want) <= max(1e-6, 1e-3 * abs(want)) else "MISMATCH"
    print(f"    analytic {name:<12s} expected {want:>11.3e}  got {got:>11.3e}  {ok}")


# --------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ct", default=DEFAULT_CT)
    p.add_argument("--labels", default=DEFAULT_LABELS)
    p.add_argument(
        "--size",
        type=int,
        default=160,
        help="cube side cropped around the bone centroid (0 = whole volume)",
    )
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    import nibabel as nib
    from scipy import ndimage

    device = torch.device(args.device)
    torch.set_grad_enabled(False)

    lbl_img = nib.load(args.labels)
    labels_np = np.asarray(lbl_img.dataobj)
    print(f"labels {args.labels}")
    print(f"  full shape {labels_np.shape}, {len(np.unique(labels_np))} unique values")

    bone_values = list(synthetic.BONE_LABEL_VALUES)
    bone_np = np.isin(labels_np, bone_values)
    if not bone_np.any():
        sys.exit("no bone labels found in this segmentation")

    # Sanity check the label mapping itself: if BONE_LABEL_VALUES does not
    # actually select bone, every rigidity number below is meaningless. Bone is
    # roughly >200 HU; soft tissue is roughly -100..100 HU.
    ct_np = np.asarray(nib.load(args.ct).dataobj)
    if ct_np.shape == labels_np.shape:
        inside = float(np.median(ct_np[bone_np]))
        outside = float(np.median(ct_np[~bone_np]))
        flag = "" if inside > outside + 150 else "   <-- SUSPICIOUS, check the mapping"
        print(
            f"  median intensity inside bone mask {inside:8.1f} vs "
            f"outside {outside:8.1f}{flag}"
        )
    else:
        print(f"  (CT shape {ct_np.shape} != labels, skipping the HU sanity check)")
    del ct_np

    # crop around the bone centroid so no mask voxel sits on the volume border
    # (the loss pads border voxels with 0 / 1.0, which would bias every number)
    if args.size:
        centre = np.array(ndimage.center_of_mass(bone_np)).astype(int)
        half = args.size // 2
        sl = tuple(
            slice(
                max(0, c - half), min(s, max(0, c - half) + args.size)
            )
            for c, s in zip(centre, labels_np.shape)
        )
        labels_np = labels_np[sl]
        bone_np = bone_np[sl]
        print(f"  cropped to {labels_np.shape} around the bone centroid")

    labels = torch.as_tensor(np.ascontiguousarray(labels_np).astype(np.int64), device=device)
    mask = torch.as_tensor(bone_np.astype(np.float64), device=device)[None, None]
    shape = tuple(labels.shape)
    n_bone = int(bone_np.sum())
    print(f"  bone voxels: {n_bone} ({100 * n_bone / bone_np.size:.2f}% of volume)")

    # ---------------------------------------------------------------- geometry
    print("\n=== C0. mask geometry (how much of the mask is boundary) ===")
    print("  A stencil at a bone voxel reads its 6+ neighbours. If those are not")
    print("  bone, the term is scored using soft-tissue displacement.")
    for it in (1, 2, 3):
        eroded = ndimage.binary_erosion(bone_np, iterations=it)
        frac = 1.0 - eroded.sum() / max(n_bone, 1)
        print(
            f"  within {it} voxel(s) of the bone surface: "
            f"{100 * frac:5.1f}% of bone voxels"
        )
    print(
        "  -> if this is large, most of the rigidity loss is computed across the\n"
        "     bone/soft-tissue interface rather than inside bone."
    )

    # ------------------------------------------------- A. global linear fields
    print("\n=== A. global linear fields (closed-form expectations) ===")

    zero = torch.zeros((1, *shape, 3), dtype=torch.float64, device=device)
    report("A1 identity (zero field)", rigidity_terms(zero, mask), "zero")

    f = linear_field(shape, np.eye(3), np.array([5.0, -3.0, 2.0]), device)
    report("A2 pure translation (5,-3,2)", rigidity_terms(f, mask), "zero")

    f = linear_field(shape, rotation("z", 5.0), np.zeros(3), device)
    report("A3 rotation 5 deg about z", rigidity_terms(f, mask), "zero")

    R = rotation("z", 5.0) @ rotation("x", 3.0)
    f = linear_field(shape, R, np.array([4.0, 1.0, -2.0]), device)
    report("A4 rotation + translation", rigidity_terms(f, mask), "zero")

    s = 1.05
    f = linear_field(shape, s * np.eye(3), np.zeros(3), device)
    t = rigidity_terms(f, mask)
    report(f"A5 isotropic expansion x{s}", t, "nonzero")
    analytic("det", t["det"], (s**3 - 1) ** 2)
    analytic("ortho", t["ortho"], 3 * (s**2 - 1) ** 2)
    analytic("affine", t["affine"], 0.0)

    s = 0.95
    f = linear_field(shape, s * np.eye(3), np.zeros(3), device)
    t = rigidity_terms(f, mask)
    report(f"A6 isotropic squeeze x{s}", t, "nonzero")
    analytic("det", t["det"], (s**3 - 1) ** 2)
    analytic("ortho", t["ortho"], 3 * (s**2 - 1) ** 2)

    A = np.diag([1.10, 0.90, 1.00])
    f_squeeze = linear_field(shape, A, np.zeros(3), device)
    t = rigidity_terms(f_squeeze, mask)
    report("A7 anisotropic squeeze (1.10,0.90,1.0)", t, "nonzero")
    analytic("det", t["det"], (np.prod(np.diag(A)) - 1) ** 2)
    analytic("ortho", t["ortho"], float(((A.T @ A - np.eye(3)) ** 2).sum()))

    k = 0.05
    A = np.array([[1.0, k, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    f = linear_field(shape, A, np.zeros(3), device)
    t = rigidity_terms(f, mask)
    report(f"A8 pure shear k={k} (det stays 1!)", t, "nonzero")
    analytic("det", t["det"], 0.0)
    analytic("ortho", t["ortho"], 2 * k**2 + k**4)
    print("    -> det is blind to shear; only ortho catches it. Confirms the")
    print("       det term is redundant whenever ortho is active.")

    # --------------------------------------------- B. real-mask, should be ~0
    print("\n=== B. fields that respect bone rigidity on the real mask ===")

    # B1: exactly zero inside bone, smooth non-zero outside
    x = voxel_coords(shape, device)
    centre = torch.tensor([(s - 1) / 2 for s in shape], dtype=torch.float64, device=device)
    r = ((x - centre) ** 2).sum(-1, keepdim=True).sqrt()
    bump = 6.0 * torch.exp(-((r / (0.35 * max(shape))) ** 2))
    outside = (1.0 - mask[0].permute(1, 2, 3, 0))[None]
    f_b1 = bump * outside * torch.tensor([1.0, 0.5, -0.5], dtype=torch.float64, device=device)
    report("B1 field == 0 inside bone, bump outside", rigidity_terms(f_b1, mask), "zero")

    # B2: each bone its own rigid motion, soft tissue blended
    f_b2, present = per_bone_rigid_field(labels, bone_values, shape, device)
    print(f"     ({len(present)} distinct bone labels in this crop)")
    report("B2 per-bone independent rigid motion", rigidity_terms(f_b2, mask), "zero")

    # B3: control -- ONE rigid motion for all bones, soft tissue blended.
    # If B2 fails but B3 passes, the loss is enforcing a single global bone
    # transform, i.e. it forbids articulation between bones.
    R = rotation("z", 3.0)
    rigid_all = linear_field(shape, R, np.array([2.0, 0.0, -1.0]), device)
    bone_b = mask[0].permute(1, 2, 3, 0)[None].bool().expand_as(rigid_all)
    f_b3 = _blur(rigid_all * bone_b, sigma=3.0)
    f_b3 = torch.where(bone_b, rigid_all, f_b3)
    report("B3 ONE rigid motion for all bones", rigidity_terms(f_b3, mask), "zero")

    # ----------------------------------------------------- C. where it comes from
    print("\n=== C1. erosion sweep: how much of the loss is the boundary shell ===")
    print("  Same field, mask eroded by n voxels. A steep drop means the loss is")
    print("  dominated by the interface, not by the bone interior.")
    for name, field in (("B1", f_b1), ("B2", f_b2), ("B3", f_b3)):
        row = []
        for it in (0, 1, 2, 3):
            m = bone_np if it == 0 else ndimage.binary_erosion(bone_np, iterations=it)
            if m.sum() == 0:
                row.append("  (empty)")
                continue
            mt = torch.as_tensor(m.astype(np.float64), device=device)[None, None]
            row.append(f"{rigidity_terms(field, mt)['total']:.3e}")
        print(f"  {name}: erode 0..3 -> " + "  ".join(f"{v:>11s}" for v in row))

    print("\n=== C2. gradient leakage: where does dL/du actually land? ===")
    print("  The wandb cosines are gradients w.r.t. network weights. If the")
    print("  rigidity gradient is large outside bone, it steers soft tissue too,")
    print("  which is what a negative cos(rigidity, dice_soft) would look like.")
    shell1 = ndimage.binary_dilation(bone_np, iterations=2) & ~bone_np
    shell2 = ndimage.binary_dilation(bone_np, iterations=6) & ~ndimage.binary_dilation(
        bone_np, iterations=2
    )
    far = ~ndimage.binary_dilation(bone_np, iterations=6)

    with torch.enable_grad():
        for name, field in (("B2", f_b2), ("B3", f_b3)):
            flow = field.clone().requires_grad_(True)
            jac_det, jac = jacobian.jacobian_matrix(flow)
            total = (
                masked_jac_det_loss(jac_det, mask)
                + orthonormality_loss(jac, mask)
                + affine_loss(flow, mask)
            )
            total.backward()
            g = flow.grad[0].norm(dim=-1)  # (D,H,W)
            tot = float(g.sum())
            parts = {
                "inside bone": bone_np,
                "0-2 vox outside": shell1,
                "2-6 vox outside": shell2,
                "far from bone": far,
            }
            print(f"  {name}: total |dL/du| = {tot:.3e}")
            for label, region in parts.items():
                rt = torch.as_tensor(region, device=device)
                frac = float(g[rt].sum()) / tot if tot > 0 else 0.0
                print(f"       {label:<18s} {100 * frac:5.1f}% of the gradient")

    print("\n=== C3. term scale (are det / ortho / affine comparable?) ===")
    print("  They are summed with equal weight, so if one is orders of magnitude")
    print("  larger it is the only one w_bone_rigidity actually controls.")
    for name, field in (("B2", f_b2), ("B3", f_b3), ("A7 squeeze", f_squeeze)):
        t = rigidity_terms(field, mask)
        share = {
            k: (t[k] / t["total"] if t["total"] > 0 else 0.0)
            for k in ("det", "ortho", "affine")
        }
        print(
            f"  {name:<12s} det {100 * share['det']:5.1f}%  "
            f"ortho {100 * share['ortho']:5.1f}%  affine {100 * share['affine']:5.1f}%"
        )

    print("\nHow to read this:")
    print("  A1-A4 non-zero          -> the loss is wrong for rigid motion itself.")
    print("  A1-A4 zero, B1/B2 large -> the loss is correct pointwise but the mask")
    print("                             boundary makes it fire on fields that do")
    print("                             respect per-bone rigidity.")
    print("  B2 large, B3 small      -> it enforces ONE transform for all bones,")
    print("                             i.e. it forbids articulation. That would")
    print("                             explain cos(rigidity, dice_bone) = -0.45.")
    print("  C2 gradient mostly      -> the masked loss is not spatially masked in")
    print("     outside bone            effect; explains cos(rigidity,dice_soft)<0.")


if __name__ == "__main__":
    main()
