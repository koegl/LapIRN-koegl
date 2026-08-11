"""Behavioural test of the bone-rigidity loss on real CTs.

`utils.enforce_rigidity_loss` sums three terms over a bone mask:
    det    -- (det J - 1)^2                  (properness)
    ortho  -- ||J^T J - I||_F^2              (no shear / anisotropic scaling)
    affine -- ||d^2 u||^2                    (Jacobian constant => one transform)

Rigid motion must give zero on all three. This script feeds the loss fields
whose correct answer is known analytically, then fields built from the real bone
segmentation, and reports where the number comes from.

Two modes:

  single case (default) -- every test, verbose.
      A. Global linear fields, closed-form expectations. Validates the maths.
         These do not depend on the case at all (a linear field has the same
         Jacobian everywhere), which is why the sweep skips them.
      B. Real-mask fields that DO respect per-bone rigidity and must score ~0:
         B1 zero inside bone, bump outside;  B2 each bone its own rigid motion;
         B3 control, one shared rigid motion for all bones.
      C. Where the number comes from: erosion sweep, gradient leakage, term
         split, and the mask geometry behind all of it.

  sweep (--sweep N) -- N random cases, only the numbers that vary per case:
      the geometry, B1 and B2, how much of each survives erosion, and how much
      of the gradient lands outside bone. Aggregated as median [min, max].

The mechanism under test: every finite-difference stencil reaches one voxel past
the centre voxel, but the mask is applied only at the centre. So a bone voxel
adjacent to soft tissue is scored using soft-tissue displacements, and a bone
voxel adjacent to a different bone is scored across that joint.

Usage:
    python test_bone_rigidity_loss.py
    python test_bone_rigidity_loss.py --ct <path> --labels <path> --size 192
    python test_bone_rigidity_loss.py --sweep 10
    python test_bone_rigidity_loss.py --sweep 10 --seed 1 --size 128
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import traceback

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jacobian  # noqa: E402
import synthetic  # noqa: E402
from utils import (  # noqa: E402
    affine_loss,
    masked_jac_det_loss,
    orthonormality_loss,
    per_label_rigid_loss,
)

DATA_ROOT = "/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset"
DEFAULT_CT = f"{DATA_ROOT}/imagesTr/PSMARegPSMA_0002_0000_00.nii.gz"
DEFAULT_LABELS = f"{DATA_ROOT}/labelsTr/PSMARegPSMA_0002_0000_00.nii.gz"

TOL = 1e-4  # what counts as "satisfied" for an exactly-rigid field

# The neighbourhood the loss actually reads: 18-connected. Central differences
# use the 6 face neighbours; the mixed second derivatives in affine_loss add the
# 12 edge neighbours (i+-1, j+-1, k). The 8 corners are never touched. Every
# geometry number must use this same element or the numbers are not comparable
# to each other -- scipy's default is 6-connected.
STENCIL = ndimage.generate_binary_structure(3, 2)


# --------------------------------------------------------------------------
# loss wrapper
# --------------------------------------------------------------------------
def per_label_value(flow: torch.Tensor, case, min_voxels: int = 50) -> float:
    """The replacement loss: per-label rigid fit residual, in voxel^2.

    The label map is restricted to `bone_np` first. That matters only inside
    this test: load_case drops the outer voxel shell of the crop from the mask
    (the old loss pads the volume border and would otherwise bias its means),
    while the label map still carries those voxels. The B-fields are built from
    the mask, so the dropped shell holds blurred / bump values rather than the
    exact rigid ones -- selecting by raw label value would score those and
    report a failure that is an artefact of the crop, not of the loss. In
    training no such drop happens and mask and labels coincide.
    """
    return per_label_detail(flow, case, min_voxels)[0]


def per_label_detail(flow: torch.Tensor, case, min_voxels: int = 50):
    """(loss, worst residual, worst label value, labels used)."""
    lab = case["labels"].clone()
    lab[~torch.as_tensor(case["bone_np"], device=lab.device)] = 0  # 0 is not a bone
    loss, info = per_label_rigid_loss(
        flow.float(),
        lab.reshape(1, 1, *case["shape"]),
        torch.as_tensor(case["bone_values"], device=flow.device),
        min_voxels=min_voxels,
    )
    return (
        float(loss),
        float(info["worst"]),
        int(info["worst_label"]),
        int(info["n_labels"]),
    )


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
    """Separable Gaussian over (1,D,H,W,3), per component."""
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


def field_zero_in_bone(case, device) -> torch.Tensor:
    """B1: exactly zero inside bone, a smooth bump everywhere else."""
    shape = case["shape"]
    x = voxel_coords(shape, device)
    centre = torch.tensor(
        [(s - 1) / 2 for s in shape], dtype=torch.float64, device=device
    )
    r = ((x - centre) ** 2).sum(-1, keepdim=True).sqrt()
    bump = 6.0 * torch.exp(-((r / (0.35 * max(shape))) ** 2))
    outside = (1.0 - case["mask"][0].permute(1, 2, 3, 0))[None]
    direction = torch.tensor([1.0, 0.5, -0.5], dtype=torch.float64, device=device)
    return bump * outside * direction


def field_per_bone_rigid(case, device, max_deg=3.0, max_shift=2.0, seed=0):
    """B2: each bone label gets its own rigid motion, soft tissue blended.

    This is the field the loss is *supposed* to accept: rigid within every bone.
    Bone voxels keep their exact rigid displacement, so the field is C0 with a
    kink at the interface -- which is what a correct bone/soft field looks like.
    """
    labels, shape = case["labels"], case["shape"]
    rng = np.random.default_rng(seed)
    x = voxel_coords(shape, device)
    flow = torch.zeros(x.shape, dtype=torch.float64, device=device)
    present = [v for v in case["bone_values"] if bool((labels == v).any())]

    for value in present:
        sel = labels == value
        idx = torch.nonzero(sel, as_tuple=False).to(torch.float64)
        centre = idx.mean(dim=0)
        axis = "xyz"[int(rng.integers(3))]
        rot = rotation(axis, float(rng.uniform(-max_deg, max_deg)))
        shift = rng.uniform(-max_shift, max_shift, size=3)
        m = torch.tensor(rot - np.eye(3), dtype=torch.float64, device=device)
        off = torch.tensor(shift, dtype=torch.float64, device=device)
        local = torch.einsum("ij,bdhwj->bdhwi", m, x - centre) + off
        flow[0][sel] = local[0][sel]

    bone = torch.as_tensor(case["bone_np"], device=device)
    blurred = _blur(flow, sigma=3.0)
    blurred[0][bone] = flow[0][bone]
    return blurred, present


def field_one_rigid(case, device) -> torch.Tensor:
    """B3 control: ONE rigid motion shared by all bones, soft tissue blended."""
    R = rotation("z", 3.0)
    rigid_all = linear_field(case["shape"], R, np.array([2.0, 0.0, -1.0]), device)
    bone_b = case["mask"][0].permute(1, 2, 3, 0)[None].bool().expand_as(rigid_all)
    blurred = _blur(rigid_all * bone_b, sigma=3.0)
    return torch.where(bone_b, rigid_all, blurred)


# --------------------------------------------------------------------------
# case loading
# --------------------------------------------------------------------------
def load_case(ct_path, labels_path, size, device, check_hu=True, verbose=True):
    """Crop a cube around the bone centroid and build the mask + label tensors."""
    labels_np = np.asarray(nib.load(labels_path).dataobj)
    bone_values = list(synthetic.BONE_LABEL_VALUES)
    bone_np = np.isin(labels_np, bone_values)
    if not bone_np.any():
        raise ValueError("no bone labels in this segmentation")

    hu_ok = None
    if check_hu and ct_path:
        ct_np = np.asarray(nib.load(ct_path).dataobj)
        if ct_np.shape == labels_np.shape:
            inside = float(np.median(ct_np[bone_np]))
            outside = float(np.median(ct_np[~bone_np]))
            hu_ok = inside > outside + 150
            if verbose:
                flag = "" if hu_ok else "   <-- SUSPICIOUS, check the mapping"
                print(
                    f"  median intensity inside bone {inside:8.1f} vs "
                    f"outside {outside:8.1f}{flag}"
                )
        del ct_np

    if size:
        centre = np.array(ndimage.center_of_mass(bone_np)).astype(int)
        half = size // 2
        sl = tuple(
            slice(max(0, c - half), min(s, max(0, c - half) + size))
            for c, s in zip(centre, labels_np.shape)
        )
        labels_np = labels_np[sl]
        bone_np = bone_np[sl]

    # Every term pads the volume border (0 for ortho/affine, 1.0 for det), so a
    # mask voxel sitting on it contributes exactly 0 and drags every mean down.
    # Drop those so the closed-form checks in section A are exact.
    edge = np.ones_like(bone_np)
    edge[1:-1, 1:-1, 1:-1] = False
    bone_np = bone_np & ~edge

    labels_np = np.ascontiguousarray(labels_np).astype(np.int64)
    bone_np = np.ascontiguousarray(bone_np)
    if not bone_np.any():
        raise ValueError("bone mask empty after cropping")

    return {
        "labels_np": labels_np,
        "bone_np": bone_np,
        "labels": torch.as_tensor(labels_np, device=device),
        "mask": torch.as_tensor(bone_np.astype(np.float64), device=device)[None, None],
        "shape": tuple(labels_np.shape),
        "n_bone": int(bone_np.sum()),
        "bone_values": bone_values,
        "hu_ok": hu_ok,
    }


def discover_cases(root: str) -> list[tuple[str, str]]:
    """(ct, labels) pairs: CT channel _0000_ with a matching file in labelsTr."""
    pairs = []
    for ct in sorted(glob.glob(os.path.join(root, "imagesTr", "*_0000_*.nii.gz"))):
        lbl = os.path.join(root, "labelsTr", os.path.basename(ct))
        if os.path.exists(lbl):
            pairs.append((ct, lbl))
    return pairs


# --------------------------------------------------------------------------
# the per-case measurements that actually vary between patients
# --------------------------------------------------------------------------
def geometry_stats(case) -> dict[str, float]:
    """How much of the mask is boundary, and how much soft tissue leaks in."""
    bone_np, lab = case["bone_np"], case["labels_np"]
    n_bone = case["n_bone"]

    interior = ndimage.binary_erosion(bone_np, structure=STENCIL)
    n_surface = n_bone - int(interior.sum())

    rind = ndimage.binary_dilation(bone_np, structure=STENCIL) & ~bone_np
    n_rind = int(rind.sum())

    # bone voxels whose neighbourhood holds a SECOND bone label: where the loss
    # couples one bone to the next and so forbids articulation
    lab_hi = np.where(bone_np, lab, -1)
    lab_lo = np.where(bone_np, lab, lab.max() + 1)
    coupled = bone_np & (
        ndimage.maximum_filter(lab_hi, footprint=STENCIL)
        != ndimage.minimum_filter(lab_lo, footprint=STENCIL)
    )

    return {
        "n_bone": float(n_bone),
        "n_labels": float(len(np.unique(lab[bone_np]))),
        "pct_surface": 100.0 * n_surface / max(n_bone, 1),
        "pct_soft_in_domain": 100.0 * n_rind / max(n_bone + n_rind, 1),
        "pct_bone_bone": 100.0 * int(coupled.sum()) / max(n_bone, 1),
        "_coupled": coupled,
    }


def eroded_mask(bone_np, iterations, device):
    m = (
        bone_np
        if iterations == 0
        else ndimage.binary_erosion(bone_np, structure=STENCIL, iterations=iterations)
    )
    if not m.any():
        return None
    return torch.as_tensor(m.astype(np.float64), device=device)[None, None]


def gradient_outside_fraction(field, case, device) -> float:
    """Fraction of |dL/du| that lands on non-bone voxels."""
    bone_np = case["bone_np"]
    with torch.enable_grad():
        flow = field.detach().clone().requires_grad_(True)
        jac_det, jac = jacobian.jacobian_matrix(flow)
        total = (
            masked_jac_det_loss(jac_det, case["mask"])
            + orthonormality_loss(jac, case["mask"])
            + affine_loss(flow, case["mask"])
        )
        total.backward()
    g = flow.grad[0].norm(dim=-1)
    tot = float(g.sum())
    if tot <= 0:
        return float("nan")
    inside = float(g[torch.as_tensor(bone_np, device=device)].sum())
    return 100.0 * (1.0 - inside / tot)


def leak_stats(case, device) -> dict[str, float]:
    """B1 and B2 plus how much of each survives erosion of the mask."""
    bone_np = case["bone_np"]
    out: dict[str, float] = {}

    f_b1 = field_zero_in_bone(case, device)
    f_b2, present = field_per_bone_rigid(case, device)
    out["n_present"] = float(len(present))

    for name, field in (("b1", f_b1), ("b2", f_b2)):
        terms = rigidity_terms(field, case["mask"])
        out[name] = terms["total"]
        for key in ("det", "ortho", "affine"):
            out[f"{name}_{key}_pct"] = (
                100.0 * terms[key] / terms["total"] if terms["total"] > 0 else 0.0
            )
        for it in (1, 2):
            m = eroded_mask(bone_np, it, device)
            val = rigidity_terms(field, m)["total"] if m is not None else float("nan")
            # ratio of means over different masks: "is the interior clean?"
            out[f"{name}_erode{it}_pct"] = (
                100.0 * val / terms["total"] if terms["total"] > 0 else float("nan")
            )
    out["b2_grad_outside_pct"] = gradient_outside_fraction(f_b2, case, device)

    # the replacement loss on the same two fields: both must now be ~0
    out["b1_new"] = per_label_value(f_b1, case)
    out["b2_new"] = per_label_value(f_b2, case)
    # and it must still reject a non-rigid field
    squeeze = linear_field(
        case["shape"], np.diag([1.10, 0.90, 1.00]), np.zeros(3), device
    )
    out["squeeze_new"] = per_label_value(squeeze, case)
    return out


# --------------------------------------------------------------------------
# reporting helpers
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


def summarise(rows: list[dict], key: str, fmt: str = "{:.1f}") -> str:
    vals = [r[key] for r in rows if key in r and np.isfinite(r[key])]
    if not vals:
        return "n/a"
    a = np.asarray(vals)
    return (
        f"{fmt.format(float(np.median(a)))} "
        f"[{fmt.format(float(a.min()))}, {fmt.format(float(a.max()))}]"
    )


# --------------------------------------------------------------------------
# single-case full report
# --------------------------------------------------------------------------
def run_single(args, device) -> None:
    print(f"labels {args.labels}")
    case = load_case(args.ct, args.labels, args.size, device, verbose=True)
    shape, mask, n_bone = case["shape"], case["mask"], case["n_bone"]
    bone_np = case["bone_np"]
    print(f"  shape {shape}, bone voxels {n_bone}")

    # ---------------------------------------------------------------- geometry
    geo = geometry_stats(case)
    print("\n=== C0. mask geometry (how much of the mask is boundary) ===")
    print("  A stencil at a bone voxel reads its 18 face+edge neighbours. If")
    print("  those are not bone, the term is scored on soft-tissue displacement.")
    print("  (cumulative: 'within 2' includes 'within 1', so it must increase)")
    prev = 0.0
    for it in (1, 2, 3):
        eroded = ndimage.binary_erosion(bone_np, structure=STENCIL, iterations=it)
        frac = 1.0 - eroded.sum() / max(n_bone, 1)
        print(
            f"  within {it} voxel(s) of the surface: {100 * frac:5.1f}% cumulative"
            f"   ({100 * (frac - prev):5.1f}% in this shell alone,"
            f" {100 * (1 - frac):5.1f}% still deeper)"
        )
        prev = frac

    print(
        f"\n  soft-tissue voxels read by the stencils: "
        f"{geo['pct_soft_in_domain']:.1f}% of the loss domain is NOT bone"
    )
    print(
        f"  bone voxels touching a DIFFERENT bone label: "
        f"{geo['pct_bone_bone']:.1f}% -- where the loss forbids two bones\n"
        f"     from moving relative to each other"
    )
    print(
        f"  sanity: {geo['pct_surface']:.1f}% of bone touches non-bone. A voxel can "
        "be in\n     both sets, but if bone-to-bone greatly exceeds bone-to-soft, "
        "suspect\n     the label map (adjacent labels tiling one structure), not "
        "anatomy."
    )
    per_label = []
    for value in np.unique(case["labels_np"][bone_np]):
        sel = bone_np & (case["labels_np"] == value)
        n_sel = int(sel.sum())
        if n_sel < 50:
            continue
        per_label.append(
            (100 * int((geo["_coupled"] & sel).sum()) / n_sel, int(value), n_sel)
        )
    per_label.sort(reverse=True)
    if per_label:
        top = ", ".join(f"{v}:{p:.0f}%" for p, v, _ in per_label[:6])
        med = float(np.median([p for p, _, _ in per_label]))
        print(f"  per-label contact: median {med:.0f}%, worst -> {top}")

    # ------------------------------------------------- A. global linear fields
    print("\n=== A. global linear fields (closed-form expectations) ===")
    print("  Case-independent: a linear field has the same Jacobian everywhere,")
    print("  so these validate the maths, not the anatomy. The sweep skips them.")

    zero = torch.zeros((1, *shape, 3), dtype=torch.float64, device=device)
    report("A1 identity (zero field)", rigidity_terms(zero, mask), "zero")

    f = linear_field(shape, np.eye(3), np.array([5.0, -3.0, 2.0]), device)
    report("A2 pure translation (5,-3,2)", rigidity_terms(f, mask), "zero")

    f = linear_field(shape, rotation("z", 5.0), np.zeros(3), device)
    report("A3 rotation 5 deg about z", rigidity_terms(f, mask), "zero")

    R = rotation("z", 5.0) @ rotation("x", 3.0)
    f = linear_field(shape, R, np.array([4.0, 1.0, -2.0]), device)
    report("A4 rotation + translation", rigidity_terms(f, mask), "zero")

    for s in (1.05, 0.95):
        f = linear_field(shape, s * np.eye(3), np.zeros(3), device)
        t = rigidity_terms(f, mask)
        report(f"A5/6 isotropic scale x{s}", t, "nonzero")
        analytic("det", t["det"], (s**3 - 1) ** 2)
        analytic("ortho", t["ortho"], 3 * (s**2 - 1) ** 2)
        analytic("affine", t["affine"], 0.0)

    A = np.diag([1.10, 0.90, 1.00])
    f_squeeze = linear_field(shape, A, np.zeros(3), device)
    t = rigidity_terms(f_squeeze, mask)
    report("A7 anisotropic squeeze (1.10,0.90,1.0)", t, "nonzero")
    analytic("det", t["det"], (np.prod(np.diag(A)) - 1) ** 2)
    analytic("ortho", t["ortho"], float(((A.T @ A - np.eye(3)) ** 2).sum()))
    analytic("affine", t["affine"], 0.0)

    k = 0.05
    A = np.array([[1.0, k, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    f = linear_field(shape, A, np.zeros(3), device)
    t = rigidity_terms(f, mask)
    report(f"A8 pure shear k={k} (det stays 1!)", t, "nonzero")
    analytic("det", t["det"], 0.0)
    analytic("ortho", t["ortho"], 2 * k**2 + k**4)
    analytic("affine", t["affine"], 0.0)
    print("    -> det is blind to shear; only ortho catches it. The two terms")
    print("       are not interchangeable.")

    # --------------------------------------------- B. real-mask, should be ~0
    print("\n=== B. fields that respect bone rigidity on the real mask ===")
    f_b1 = field_zero_in_bone(case, device)
    report(
        "B1 field == 0 inside bone, bump outside", rigidity_terms(f_b1, mask), "zero"
    )

    f_b2, present = field_per_bone_rigid(case, device)
    print(f"     ({len(present)} distinct bone labels in this crop)")
    report("B2 per-bone independent rigid motion", rigidity_terms(f_b2, mask), "zero")

    f_b3 = field_one_rigid(case, device)
    report("B3 ONE rigid motion for all bones", rigidity_terms(f_b3, mask), "zero")

    # ------------------------------------------------- C. where it comes from
    print("\n=== C1. erosion sweep: how much of the loss is the boundary shell ===")
    print("  Same field, mask eroded by n voxels. A steep drop means the loss is")
    print("  dominated by the interface, not by the bone interior.")
    for name, field in (("B1", f_b1), ("B2", f_b2), ("B3", f_b3)):
        row = []
        for it in (0, 1, 2, 3):
            m = eroded_mask(bone_np, it, device)
            row.append(
                "  (empty)" if m is None else f"{rigidity_terms(field, m)['total']:.3e}"
            )
        print(f"  {name}: erode 0..3 -> " + "  ".join(f"{v:>11s}" for v in row))

    print("\n=== C2. gradient leakage: where does dL/du actually land? ===")
    print("  The wandb cosines are gradients w.r.t. network weights. If the")
    print("  rigidity gradient is large outside bone, it steers soft tissue too,")
    print("  which is what a negative cos(rigidity, dice_soft) would look like.")
    dil2 = ndimage.binary_dilation(bone_np, structure=STENCIL, iterations=2)
    dil6 = ndimage.binary_dilation(bone_np, structure=STENCIL, iterations=6)
    # the shells EXCLUDE bone: shell1 is the rind of soft tissue immediately
    # outside the mask, not a band starting inside it
    parts = {
        "inside bone": bone_np,
        "1-2 vox outside (soft)": dil2 & ~bone_np,
        "3-6 vox outside (soft)": dil6 & ~dil2,
        "far from bone (soft)": ~dil6,
    }
    for name, field in (("B2", f_b2), ("B3", f_b3)):
        with torch.enable_grad():
            flow = field.detach().clone().requires_grad_(True)
            jac_det, jac = jacobian.jacobian_matrix(flow)
            total = (
                masked_jac_det_loss(jac_det, mask)
                + orthonormality_loss(jac, mask)
                + affine_loss(flow, mask)
            )
            total.backward()
        g = flow.grad[0].norm(dim=-1)
        tot = float(g.sum())
        print(f"  {name}: total |dL/du| = {tot:.3e}")
        for label, region in parts.items():
            rt = torch.as_tensor(region, device=device)
            frac = float(g[rt].sum()) / tot if tot > 0 else 0.0
            print(f"       {label:<24s} {100 * frac:5.1f}% of the gradient")

    print("\n=== C3. term scale (are det / ortho / affine comparable?) ===")
    print("  They are summed with equal weight, so if one is orders of magnitude")
    print("  larger it is the only one w_bone_rigidity actually controls.")
    for name, field in (("B2", f_b2), ("B3", f_b3), ("A7 squeeze", f_squeeze)):
        t = rigidity_terms(field, mask)
        sh = {
            key: (t[key] / t["total"] if t["total"] > 0 else 0.0)
            for key in ("det", "ortho", "affine")
        }
        print(
            f"  {name:<12s} det {100 * sh['det']:5.1f}%  "
            f"ortho {100 * sh['ortho']:5.1f}%  affine {100 * sh['affine']:5.1f}%"
        )

    # ------------------------------- D. the replacement: per-label rigid fit
    print("\n=== D. per-label rigid fit (utils.per_label_rigid_loss) ===")
    print("  Same fields, new loss. Rigid-per-bone fields must now score ~0,")
    print("  including B1/B2/B3, which the old term rejected. Non-rigid fields")
    print("  must still be caught. Units are voxel^2, so magnitudes differ.")
    d_cases = [
        ("D1 identity (zero field)", zero, "zero"),
        (
            "D2 pure translation",
            linear_field(shape, np.eye(3), np.array([5.0, -3.0, 2.0]), device),
            "zero",
        ),
        (
            "D3 rotation 5 deg about z",
            linear_field(shape, rotation("z", 5.0), np.zeros(3), device),
            "zero",
        ),
        (
            "D4 rotation + translation",
            linear_field(
                shape,
                rotation("z", 5.0) @ rotation("x", 3.0),
                np.array([4.0, 1.0, -2.0]),
                device,
            ),
            "zero",
        ),
        (
            "D5 isotropic expansion x1.05",
            linear_field(shape, 1.05 * np.eye(3), np.zeros(3), device),
            "nonzero",
        ),
        ("D6 anisotropic squeeze", f_squeeze, "nonzero"),
        (
            "D7 pure shear k=0.05",
            linear_field(
                shape,
                np.array([[1.0, 0.05, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
                np.zeros(3),
                device,
            ),
            "nonzero",
        ),
        ("D8 = B1 zero inside bone", f_b1, "zero"),
        ("D9 = B2 per-bone rigid motion", f_b2, "zero"),
        ("D10 = B3 one rigid motion", f_b3, "zero"),
    ]
    for name, field, expect in d_cases:
        val, worst, worst_lbl, n_used = per_label_detail(field, case)
        if expect == "zero":
            ok = val < TOL
            verdict = "  PASS" if ok else "  <-- FAIL (should be ~0)"
        else:
            ok = val > TOL
            verdict = "  PASS" if ok else "  <-- FAIL (should be > 0)"
        print(f"  {name:<44s} residual={val:>11.3e}{verdict}")
        if not ok and expect == "zero":
            # the loss is a mean over labels, so one bad structure hides in it.
            # ratio >> 1 means a single label is driving the whole number --
            # usually a near-planar or near-collinear one whose rigid fit is
            # ill-conditioned, in which case raise --min-voxels.
            print(
                f"       worst label {worst_lbl} at {worst:.3e} "
                f"({worst / max(val, 1e-30):.0f}x the mean, {n_used} labels used)"
            )
    print("  -> D8/D9/D10 are exactly the three the old term failed. If they")
    print("     pass here, the leakage and the bone-fusing are both gone.")

    print("\nHow to read this:")
    print("  A1-A4 non-zero          -> the loss is wrong for rigid motion itself.")
    print("  A1-A4 zero, B1/B2 large -> the loss is correct pointwise but the mask")
    print("                             boundary makes it fire on fields that do")
    print("                             respect per-bone rigidity.")
    print("  B2 large, B3 small      -> it enforces ONE transform for all bones,")
    print("                             i.e. it forbids articulation.")
    print("  C2 gradient mostly      -> the masked loss is not spatially masked in")
    print("     outside bone            effect; explains cos(rigidity,dice_soft)<0.")


# --------------------------------------------------------------------------
# sweep over N random cases
# --------------------------------------------------------------------------
def run_sweep(args, device) -> None:
    pairs = discover_cases(args.data_root)
    if not pairs:
        sys.exit(f"no (image, label) pairs found under {args.data_root}")
    rng = np.random.default_rng(args.seed)
    n = min(args.sweep, len(pairs))
    picked = [pairs[i] for i in rng.choice(len(pairs), size=n, replace=False)]

    print(f"sweep over {n} of {len(pairs)} cases (seed {args.seed}, size {args.size})")
    print("skipping section A: linear fields give the same answer on every case.\n")

    hdr = (
        f"{'case':<28s} {'bone vox':>9s} {'lbls':>5s} {'surf%':>6s} "
        f"{'soft%':>6s} {'b-b%':>6s} {'B1':>10s} {'B1e1%':>7s} "
        f"{'B2':>10s} {'B2e2%':>7s} {'grad_out%':>10s} "
        f"{'newB1':>10s} {'newB2':>10s} {'newSqz':>10s}"
    )
    print(hdr)
    print("-" * len(hdr))

    rows: list[dict] = []
    for ct, lbl in picked:
        name = os.path.basename(lbl).replace(".nii.gz", "")
        try:
            case = load_case(
                ct, lbl, args.size, device, check_hu=not args.no_hu, verbose=False
            )
            row = geometry_stats(case)
            row.pop("_coupled", None)
            row.update(leak_stats(case, device))
            row["name"] = name
            row["hu_ok"] = case["hu_ok"]
            rows.append(row)
            hu_flag = "" if case["hu_ok"] in (None, True) else "  <-- HU suspicious"
            print(
                f"{name:<28s} {row['n_bone']:>9.0f} {row['n_labels']:>5.0f} "
                f"{row['pct_surface']:>6.1f} {row['pct_soft_in_domain']:>6.1f} "
                f"{row['pct_bone_bone']:>6.1f} {row['b1']:>10.3e} "
                f"{row['b1_erode1_pct']:>7.1f} {row['b2']:>10.3e} "
                f"{row['b2_erode2_pct']:>7.1f} "
                f"{row['b2_grad_outside_pct']:>10.1f} "
                f"{row['b1_new']:>10.3e} {row['b2_new']:>10.3e} "
                f"{row['squeeze_new']:>10.3e}{hu_flag}"
            )
        except Exception as exc:  # keep the sweep going, report at the end
            print(f"{name:<28s}  FAILED: {exc}")
            if args.traceback:
                traceback.print_exc()

    if not rows:
        sys.exit("every case failed")

    print(f"\n=== aggregate over {len(rows)} cases: median [min, max] ===")
    print("  geometry")
    print(f"    bone voxels in crop          {summarise(rows, 'n_bone', '{:.0f}')}")
    print(f"    distinct bone labels         {summarise(rows, 'n_labels', '{:.0f}')}")
    print(f"    bone touching non-bone   %   {summarise(rows, 'pct_surface')}")
    print(f"    loss domain that is NOT bone {summarise(rows, 'pct_soft_in_domain')}")
    print(f"    bone touching another bone % {summarise(rows, 'pct_bone_bone')}")
    print("  behaviour (both B1 and B2 SHOULD be ~0)")
    print(f"    B1  zero-in-bone field       {summarise(rows, 'b1', '{:.3e}')}")
    print(f"    B2  per-bone rigid field     {summarise(rows, 'b2', '{:.3e}')}")
    print("  where it comes from")
    print(f"    B1 left after 1 erosion  %   {summarise(rows, 'b1_erode1_pct')}")
    print(f"    B2 left after 2 erosions %   {summarise(rows, 'b2_erode2_pct')}")
    print(f"    B2 gradient outside bone %   {summarise(rows, 'b2_grad_outside_pct')}")
    print("  term split on B2")
    print(f"    det   %                      {summarise(rows, 'b2_det_pct')}")
    print(f"    ortho %                      {summarise(rows, 'b2_ortho_pct')}")
    print(f"    affine%                      {summarise(rows, 'b2_affine_pct')}")
    print("  replacement loss (per-label rigid fit), voxel^2")
    print(f"    B1  must be ~0               {summarise(rows, 'b1_new', '{:.3e}')}")
    print(f"    B2  must be ~0               {summarise(rows, 'b2_new', '{:.3e}')}")
    print(
        f"    squeeze must be > 0          {summarise(rows, 'squeeze_new', '{:.3e}')}"
    )
    n_ok = sum(
        1
        for r in rows
        if r["b1_new"] < TOL and r["b2_new"] < TOL and r["squeeze_new"] > TOL
    )
    print(f"    -> replacement passes on {n_ok}/{len(rows)} cases")

    print("\nHow to read this:")
    print("  B1/B2 far from 0 on every case -> not a quirk of one patient.")
    print("  'left after erosion' near 0    -> the value is boundary leakage;")
    print("                                    what remains is bone-to-bone")
    print("                                    coupling plus interior error.")
    print("  'gradient outside bone' high   -> the masked loss steers soft tissue,")
    print("                                    i.e. cos(rigidity, dice_soft) < 0.")


# --------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ct", default=DEFAULT_CT)
    p.add_argument("--labels", default=DEFAULT_LABELS)
    p.add_argument("--data-root", default=DATA_ROOT, help="holds imagesTr/ labelsTr/")
    p.add_argument(
        "--sweep",
        type=int,
        default=0,
        help="run the per-case subset on N random cases instead of one full report",
    )
    p.add_argument("--seed", type=int, default=0, help="which random cases to draw")
    p.add_argument(
        "--size",
        type=int,
        default=160,
        help="cube side cropped around the bone centroid (0 = whole volume)",
    )
    p.add_argument("--no-hu", action="store_true", help="skip the CT sanity check")
    p.add_argument("--traceback", action="store_true")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    torch.set_grad_enabled(False)

    if args.sweep:
        run_sweep(args, device)
    else:
        run_single(args, device)


if __name__ == "__main__":
    main()
