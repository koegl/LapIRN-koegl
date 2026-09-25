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

Usage:
    python voxelwise_cosine.py                      # 0006: 02 -> 00
    python voxelwise_cosine.py --weights <no-PET LapIRN>.pth
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict

REPO = Path(__file__).resolve().parent
os.environ.setdefault("LAPIRN_CODE", str(REPO / "Code"))
sys.path.insert(0, str(REPO / "submission"))

import infer  # noqa: E402  (puts Code/ on sys.path)
import instance_opt  # noqa: E402
import miccai2020_model_stage  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import utils  # noqa: E402
from miccai2020_model_stage import NCC  # noqa: E402

DATA_DIR = Path("/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset")
PAIRS = (("ncc", "dice"), ("mtv", "dice"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default="0331")
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
    """The three IO terms, weighted and computed exactly as in
    instance_opt.io_objective (container settings: ncc_weight = cfg.w_ct, no
    class weights, soft squared MTV)."""
    x_y = transform(X, disp_unit.permute(0, 2, 3, 4, 1), grid)
    ncc = NCC(cfg.lvl3_ncc_win)(x_y[:, 0:1], Y[:, 0:1])
    dice = instance_opt.dice_loss_with_grad(
        x_lbl_ct, y_lbl_ct, disp_unit, grid, transform, class_weights=None
    )
    warped_mask = utils.warp_binary_mask(moving_pet_mask, disp_unit, grid, transform)
    mtv = utils.mtv_bias_loss(warped_mask, moving_pet_mask)
    return {
        "ncc": cfg.w_ct * ncc,
        "dice": cfg.w_io_dice * dice,
        "mtv": cfg.w_io_mtv * mtv**2,
    }


def unit_to_mm_scale(shape, spacing, device: torch.device) -> torch.Tensor:
    """Per-channel factor m_c = u_c * scale_c from unit flow to mm.

    Unit-flow channel 0 is the last array axis, channel 2 the first (see
    Functions.transform_unit_flow_to_flow_cuda)."""
    h, w, d = shape
    voxel = [(d - 1) / 2, (w - 1) / 2, (h - 1) / 2]
    mm = [voxel[0] * spacing[2], voxel[1] * spacing[1], voxel[2] * spacing[0]]
    return torch.tensor(mm, device=device).view(1, 3, 1, 1, 1)


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

    def save_set(field: torch.Tensor, directory: Path) -> None:
        """Per-term gradients at `field` (in mm), the pair maps, and the moving
        images / labels warped by `field`."""
        directory.mkdir(parents=True, exist_ok=True)
        print(f"--- {directory.name}")
        disp = field.detach().clone().requires_grad_(True)
        losses = term_losses(
            disp, X, Y, x_lbl_ct, y_lbl_ct, x_lbl_pet, transform, grid, cfg
        )
        grads: Dict[str, torch.Tensor] = {}
        for name, loss in losses.items():
            (g_unit,) = torch.autograd.grad(loss, disp, retain_graph=True)
            grads[name] = g_unit / scale  # dL/dm_c = dL/du_c / scale_c
            print(f"{name}: loss {loss.item():.6f}")
        del losses

        norms = {name: g.norm(dim=1, keepdim=True) for name, g in grads.items()}
        for name, n in norms.items():
            save(n, directory, f"grad_{name}_norm")

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
            active = denom > 0
            print(
                f"{a} vs {b}: {int(active.sum())} voxels with both gradients, "
                f"mean cos {cos[active].mean().item():+.3f}, "
                f"mag-weighted cos {(cos * mag).sum().item() / mag.sum().item():+.3f}"
            )

        # warped moving images / labels in the fixed frame, raw intensities
        with torch.no_grad():
            flow = field.permute(0, 2, 3, 4, 1)
            save(
                transform(moving_ct_raw, flow, grid),
                directory,
                f"warped_moving_ct_{mtp}",
            )
            save(
                transform(moving_pet_raw, flow, grid),
                directory,
                f"warped_moving_pet_{mtp}",
            )
            save(
                transform_nearest(x_lbl_pet, flow, grid),
                directory,
                f"warped_moving_tumour_{mtp}",
                np.uint8,
            )
            save(
                transform_nearest(x_lbl_ct, flow, grid),
                directory,
                f"warped_moving_organs_{mtp}",
                np.int16,
            )

    moving_ct_raw = load_volume(moving_ct, device)
    moving_pet_raw = load_volume(moving_pet, device)
    for i in sorted(snapshots):
        save_set(snapshots.pop(i), out_dir / f"step_{i:03d}")
    save_set(best, out_dir / f"best_step_{best_step:03d}")

    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
