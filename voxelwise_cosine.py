"""Voxelwise cosine between IO loss gradients w.r.t. the displacement field.

Registers one training pair exactly as the submission container does (ANTs
affine + LapIRN + full-resolution IO), but with the ORIGINAL organ and tumour
labels from labelsTr and no time budget. At IO steps 0, 25, 50, 75, 100
(step i = the field after i updates) and at the field IO returns (best_disp),
each loss term is backpropagated on its own to the full-resolution
displacement. Each field gets its own subfolder (step_XXX/, best_step_XXX/),
and per pair of terms it holds

    grad_<a>_norm       |g_a|
    grad_<b>_norm       |g_b|
    <a>_<b>_cos         cos(g_a, g_b), 0 where either gradient is exactly 0
    <a>_<b>_mag         sqrt(|g_a| |g_b|)  -- large only where both are active
    <a>_<b>_cosmag      cos * mag

Gradients are taken w.r.t. the displacement in mm (the unit flow is rescaled
per axis by voxel count and spacing), so the cosine is a physical angle on the
anisotropic grid. Terms carry their IO weights; the cosine is scale-free, the
magnitudes are what the optimiser sees.

The field lives on the fixed grid, so all maps are in the fixed frame. The
warped moving CT / PET / tumour / organ labels are saved next to them, since
with a tumour-poor fixed scan the warped moving is the informative overlay.
Everything is written with the fixed CT's affine and header (open in Slicer).

Voxelwise gradients of interpolated labels and finite-difference terms flip
sign between neighbouring voxels, so the raw maps are salt-and-pepper. Two
ways to read them at a coarser scale:

  --sigmas-mm   each gradient is Gaussian-smoothed (per channel, sigma in mm,
                converted per axis) before the maps are formed; every sigma gets
                its own folder (<field>/ for 0, <field>_sigma<S>mm/ otherwise).
  region_cosine.csv
                one cosine per region: the gradients inside the region are
                flattened and compared as whole vectors, i.e.
                sum(g_a . g_b) / (||g_a||_R ||g_b||_R). Regions are each lesion
                (connected component of the moving tumour, warped to the fixed
                frame with nearest, dilated by --lesion-dilation voxels so the
                boundary band where Dice / MTV act is inside), the union of the
                lesions per location group (bone / organs / background, by
                majority vote over the moving organ labels, as in
                tumour_in_bone.py), all lesions, and the whole volume. Computed
                for every saved field, sigma and pair.

Usage:
    python voxelwise_cosine.py                      # 0006: 02 -> 00
    python voxelwise_cosine.py --weights <no-PET LapIRN>.pth
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict

REPO = Path(__file__).resolve().parent
os.environ.setdefault("LAPIRN_CODE", str(REPO / "Code"))
sys.path.insert(0, str(REPO / "submission"))

import infer  # noqa: E402  (puts Code/ on sys.path)
import instance_opt  # noqa: E402
import jacobian  # noqa: E402
import miccai2020_model_stage  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import utils  # noqa: E402
import pandas as pd  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from miccai2020_model_stage import NCC  # noqa: E402
from scipy import ndimage  # noqa: E402
from synthetic import BONE_LABEL_VALUES  # noqa: E402

DATA_DIR = Path("/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset")
PAIRS = (("ncc", "dice"), ("mtv", "dice"), ("jactum", "dice"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default="0122")
    p.add_argument("--moving-tp", default="01")
    p.add_argument("--fixed-tp", default="00")
    p.add_argument(
        "--weights", type=Path, default=REPO / "submission/weights/model.pth"
    )
    p.add_argument("--io-it", type=int, default=100)
    p.add_argument(
        "--save-steps",
        type=int,
        nargs="+",
        default=[0, 25, 50, 75, 100],
        help="IO steps to save (step i = field after i updates); best is always saved",
    )
    p.add_argument(
        "--sigmas-mm",
        type=float,
        nargs="+",
        default=[0.0, 6.0],
        help="Gaussian sigma(s) in mm for smoothing the gradients; 0 = raw",
    )
    p.add_argument(
        "--lesion-dilation",
        type=int,
        default=2,
        help="voxels by which each warped lesion is dilated for its region",
    )
    p.add_argument(
        "--min-lesion-voxels",
        type=int,
        default=None,
        help="moving-frame components smaller than this are not their own region "
        "(default: cfg.io_cc_min_voxels, the size filter of IO's per-lesion terms)",
    )
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def image_path(data_dir: Path, case: str, modality: str, tp: str) -> Path:
    return data_dir / "imagesTr" / f"PSMARegPSMA_{case}_{modality}_{tp}.nii.gz"


def label_path(data_dir: Path, case: str, modality: str, tp: str) -> Path:
    return data_dir / "labelsTr" / f"PSMARegPSMA_{case}_{modality}_{tp}.nii.gz"


def load_volume(path: Path, device: torch.device) -> torch.Tensor:
    arr = nib.load(str(path)).get_fdata().astype(np.float32)
    return torch.from_numpy(arr)[None, None].to(device)


def term_losses(
    disp_unit: torch.Tensor,
    X: torch.Tensor,
    Y: torch.Tensor,
    x_lbl_ct: torch.Tensor,
    y_lbl_ct: torch.Tensor,
    moving_pet_mask: torch.Tensor,
    transform: torch.nn.Module,
    grid: torch.Tensor,
    cfg,
) -> Dict[str, torch.Tensor]:
    """The IO terms, weighted and computed exactly as in
    instance_opt.io_objective (container settings: ncc_weight = cfg.w_ct, no
    class weights, soft squared MTV, tumour det(J) term)."""
    x_y = transform(X, disp_unit.permute(0, 2, 3, 4, 1), grid)
    ncc = NCC(cfg.lvl3_ncc_win)(x_y[:, 0:1], Y[:, 0:1])
    dice = instance_opt.dice_loss_with_grad(
        x_lbl_ct, y_lbl_ct, disp_unit, grid, transform, class_weights=None
    )
    warped_mask = utils.warp_binary_mask(moving_pet_mask, disp_unit, grid, transform)
    mtv = utils.mtv_bias_loss(warped_mask, moving_pet_mask)
    # det(J) lives on the fixed grid, so it is masked with the warped lesion,
    # detached -- same as io_objective
    disp_voxel = infer.Functions.transform_unit_flow_to_flow_cuda(
        disp_unit.permute(0, 2, 3, 4, 1).clone()
    )
    jac_det, _ = jacobian.jacobian_matrix(disp_voxel)
    jactum = utils.masked_jac_det_loss(jac_det, warped_mask.detach())
    return {
        "ncc": cfg.w_ct * ncc,
        "dice": cfg.w_io_dice * dice,
        "mtv": cfg.w_io_mtv * mtv**2,
        "jactum": cfg.w_io_jacobian_tumor * jactum,
    }


def unit_to_mm_scale(shape, spacing, device: torch.device) -> torch.Tensor:
    """Per-channel factor m_c = u_c * scale_c from unit flow to mm.

    Unit-flow channel 0 is the last array axis, channel 2 the first (see
    Functions.transform_unit_flow_to_flow_cuda)."""
    h, w, d = shape
    voxel = [(d - 1) / 2, (w - 1) / 2, (h - 1) / 2]
    mm = [voxel[0] * spacing[2], voxel[1] * spacing[1], voxel[2] * spacing[0]]
    return torch.tensor(mm, device=device).view(1, 3, 1, 1, 1)


def gaussian_smooth(g: torch.Tensor, sigma_vox) -> torch.Tensor:
    """Separable Gaussian over the spatial axes of (1, C, H, W, D), each
    channel on its own; sigma_vox is one sigma per spatial axis, in voxels."""
    out = g.transpose(0, 1).contiguous()  # (C, 1, H, W, D)
    for axis, sigma in enumerate(sigma_vox):
        if sigma <= 0:
            continue
        r = int(math.ceil(3 * sigma))
        x = torch.arange(-r, r + 1, device=g.device, dtype=g.dtype)
        k = torch.exp(-(x**2) / (2 * sigma**2))
        shape = [1, 1, 1, 1, 1]
        shape[2 + axis] = 2 * r + 1
        padding = [0, 0, 0]
        padding[axis] = r
        out = F.conv3d(out, (k / k.sum()).view(shape), padding=tuple(padding))
    return out.transpose(0, 1).contiguous()


def lesion_components(tumour: np.ndarray, organs: np.ndarray, voxel_ml: float, min_voxels: int):
    """Connected components of the moving tumour, relabelled 1..n after
    dropping those below min_voxels, with their size and location group."""
    lab, n = ndimage.label(tumour)
    bone = np.isin(organs, BONE_LABEL_VALUES)
    out = np.zeros(lab.shape, dtype=np.int32)
    info = []
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        m = lab[sl] == k
        n_vox = int(m.sum())
        if n_vox < min_voxels:
            continue
        lesion = len(info) + 1
        out[sl][m] = lesion
        n_bone = int(bone[sl][m].sum())
        n_organs = int(((organs[sl][m] != 0) & ~bone[sl][m]).sum())
        counts = {"bone": n_bone, "organs": n_organs, "background": n_vox - n_bone - n_organs}
        info.append(
            {
                "lesion": lesion,
                "lesion_voxels": n_vox,
                "lesion_ml": n_vox * voxel_ml,
                "location": max(counts, key=counts.get),
                **{f"fraction_in_{g}": c / n_vox for g, c in counts.items()},
            }
        )
    print(f"{n} tumour components, {len(info)} with >= {min_voxels} voxels")
    return out, info


def region_cosine(ga: torch.Tensor, gb: torch.Tensor, mask: torch.Tensor):
    """cos between the two gradients flattened over the region, and their norms."""
    a, b = ga[0][:, mask], gb[0][:, mask]
    na, nb = a.norm().item(), b.norm().item()
    cos = (a * b).sum().item() / (na * nb) if na > 0 and nb > 0 else float("nan")
    return cos, na, nb


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device(args.device)
    cfg = infer.build_config()
    cfg.io_it = args.io_it

    case, mtp, ftp = args.case, args.moving_tp, args.fixed_tp
    out_dir = args.out_dir or (
        REPO / "saved/voxelwise_cosine" / f"PSMARegPSMA_{case}_{mtp}_to_{ftp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    fixed_ct = image_path(args.data_dir, case, "0000", ftp)
    fixed_pet = image_path(args.data_dir, case, "0001", ftp)
    moving_ct = image_path(args.data_dir, case, "0000", mtp)
    moving_pet = image_path(args.data_dir, case, "0001", mtp)
    ref = nib.load(str(fixed_ct))
    spacing = tuple(float(z) for z in ref.header.get_zooms()[:3])

    def save(arr: torch.Tensor, directory: Path, name: str, dtype=np.float32) -> None:
        data = arr.detach().squeeze().cpu().numpy().astype(dtype)
        header = ref.header.copy()
        header.set_data_dtype(dtype)
        nib.save(
            nib.Nifti1Image(data, ref.affine, header), str(directory / f"{name}.nii.gz")
        )

    # --- registration, identical to the container ---
    grid = infer.Functions.generate_grid_unit(cfg.img_shape)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).to(device).float()
    transform = miccai2020_model_stage.SpatialTransform_unit().to(device)
    transform_nearest = miccai2020_model_stage.SpatialTransformNearest_unit().to(device)

    model = infer.create_model(device, cfg, args.weights)
    X, Y = infer.load_pair(fixed_ct, fixed_pet, moving_ct, moving_pet)
    X, Y = X.to(device), Y.to(device)

    flow_affine = infer.affine_flow(fixed_ct, moving_ct, cfg, device)
    X_affine = transform(X, flow_affine, grid)
    with torch.no_grad():
        F_X_Y, _, _, _, _, _, _ = model(X_affine, Y)
    total_unit_flow = infer.compose(flow_affine, F_X_Y.permute(0, 2, 3, 4, 1), grid)
    del model

    # --- original labels: organs (0000) and tumour (0001) ---
    x_lbl_ct = load_volume(label_path(args.data_dir, case, "0000", mtp), device)
    y_lbl_ct = load_volume(label_path(args.data_dir, case, "0000", ftp), device)
    x_lbl_pet = (
        load_volume(label_path(args.data_dir, case, "0001", mtp), device) == 1
    ).float()
    print(f"moving tumour voxels: {int(x_lbl_pet.sum())}")
    lesion_lbl_np, lesions = lesion_components(
        x_lbl_pet[0, 0].cpu().numpy() > 0,
        x_lbl_ct[0, 0].cpu().numpy().round().astype(np.int32),
        float(np.prod(spacing)) / 1000.0,
        args.min_lesion_voxels or cfg.io_cc_min_voxels,
    )
    lesion_lbl = torch.from_numpy(lesion_lbl_np.astype(np.float32))[None, None].to(device)

    # --- IO: container settings, full resolution, no deadline ---
    save_steps = set(args.save_steps)
    snapshots: Dict[int, torch.Tensor] = {}

    def keep(i: int, field: torch.Tensor) -> None:
        if i in save_steps:
            snapshots[i] = field.clone()

    best = instance_opt.run_io(
        Y,
        total_unit_flow,
        X,
        x_lbl_ct,
        x_lbl_pet,
        y_lbl_ct,
        transform,
        transform_nearest,
        grid,
        cfg,
        device,
        include_pet=True,
        include_rigidity=False,
        include_dice=True,
        log_hard_dice=True,
        use_class_weights=False,
        deadline=None,
        step_callback=keep,
    )
    best_step = instance_opt.run_io.best_step
    print(f"IO ran {instance_opt.run_io.steps_taken} steps, best step {best_step}")
    missing = sorted(save_steps - snapshots.keys())
    if missing:
        print(f"WARNING: steps {missing} never reached, not saved")

    scale = unit_to_mm_scale(cfg.img_shape, spacing, device)

    rows = []

    def fixed_frame_regions(flow: torch.Tensor):
        """(name, lesion info or None, bool mask) for every region in the fixed
        frame under `flow`."""
        warped = transform_nearest(lesion_lbl, flow, grid)[0, 0].round().long()
        r = args.lesion_dilation
        regions = []
        for info in lesions:
            m = (warped == info["lesion"]).float()[None, None]
            if r > 0:
                m = F.max_pool3d(m, 2 * r + 1, stride=1, padding=r)
            regions.append((f"lesion_{info['lesion']}", info, m[0, 0] > 0))
        lesion_regions = list(regions)
        for group in ("bone", "organs", "background"):
            ms = [m for _, info, m in lesion_regions if info["location"] == group]
            if ms:
                regions.append((f"group_{group}", None, torch.stack(ms).any(0)))
        if lesion_regions:
            all_masks = torch.stack([m for _, _, m in lesion_regions])
            regions.append(("all_lesions", None, all_masks.any(0)))
        regions.append(("whole_volume", None, torch.ones_like(warped, dtype=torch.bool)))
        return regions

    def save_set(field: torch.Tensor, name: str, step: int, is_best: bool) -> None:
        """Per-term gradients at `field` (in mm), smoothed per sigma; for each
        sigma the pair maps, the moving images / labels warped by `field`, and
        the region cosines."""
        disp = field.detach().clone().requires_grad_(True)
        losses = term_losses(
            disp, X, Y, x_lbl_ct, y_lbl_ct, x_lbl_pet, transform, grid, cfg
        )
        raw: Dict[str, torch.Tensor] = {}
        for term, loss in losses.items():
            (g_unit,) = torch.autograd.grad(loss, disp, retain_graph=True)
            raw[term] = g_unit / scale  # dL/dm_c = dL/du_c / scale_c
        print(f"--- {name}: " + ", ".join(f"{t} {l.item():.4f}" for t, l in losses.items()))
        del losses

        with torch.no_grad():
            flow = field.permute(0, 2, 3, 4, 1)
            warped = {
                f"warped_moving_ct_{mtp}": (transform(moving_ct_raw, flow, grid), np.float32),
                f"warped_moving_pet_{mtp}": (transform(moving_pet_raw, flow, grid), np.float32),
                f"warped_moving_tumour_{mtp}": (
                    transform_nearest(x_lbl_pet, flow, grid), np.uint8
                ),
                f"warped_moving_organs_{mtp}": (
                    transform_nearest(x_lbl_ct, flow, grid), np.int16
                ),
            }
            regions = fixed_frame_regions(flow)

        for sigma in args.sigmas_mm:
            directory = out_dir / (name if sigma == 0 else f"{name}_sigma{sigma:g}mm")
            directory.mkdir(parents=True, exist_ok=True)
            grads = {
                t: gaussian_smooth(g, [sigma / sp for sp in spacing]) if sigma > 0 else g
                for t, g in raw.items()
            }

            norms = {t: g.norm(dim=1, keepdim=True) for t, g in grads.items()}
            for t, n in norms.items():
                save(n, directory, f"grad_{t}_norm")

            for a, b in PAIRS:
                denom = norms[a] * norms[b]
                dot = (grads[a] * grads[b]).sum(dim=1, keepdim=True)
                cos = torch.where(
                    denom > 0, dot / denom.clamp_min(1e-30), torch.zeros_like(dot)
                )
                mag = denom.sqrt()
                save(cos, directory, f"{a}_{b}_cos")
                save(mag, directory, f"{a}_{b}_mag")
                save(cos * mag, directory, f"{a}_{b}_cosmag")

                summary = []
                for region, info, mask in regions:
                    c, na, nb = region_cosine(grads[a], grads[b], mask)
                    rows.append(
                        {
                            "field": name,
                            "step": step,
                            "is_best": is_best,
                            "sigma_mm": sigma,
                            "pair": f"{a}_{b}",
                            "region": region,
                            "region_voxels": int(mask.sum()),
                            "cos": c,
                            "norm_a": na,
                            "norm_b": nb,
                            **(info or {}),
                        }
                    )
                    if not region.startswith("lesion_"):
                        summary.append(f"{region} {c:+.3f}")
                print(f"  sigma {sigma:g}mm {a} vs {b}: " + ", ".join(summary))

            for fname, (arr, dtype) in warped.items():
                save(arr, directory, fname, dtype)

    moving_ct_raw = load_volume(moving_ct, device)
    moving_pet_raw = load_volume(moving_pet, device)
    for i in sorted(snapshots):
        save_set(snapshots.pop(i), f"step_{i:03d}", i, i == best_step)
    save_set(best, f"best_step_{best_step:03d}", best_step, True)

    pd.DataFrame(rows).to_csv(out_dir / "region_cosine.csv", index=False)
    pd.DataFrame(lesions).to_csv(out_dir / "lesions.csv", index=False)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
