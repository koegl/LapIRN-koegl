"""PSMAReg (Learn2Reg 2026) test-phase entrypoint.

Registers ONE moving/fixed PET+CT set per run:

    infer.py <fixed_ct> <fixed_pet> <moving_ct> <moving_pet> <output_disp>

fixed  = baseline  (timepoint 00), moving = follow-up (timepoint 01).

Pipeline (mirrors Code/inference.py, chase_best_model, use_io=False):
  1. body-mask + normalise CT/PET -> 2-channel moving (X) / fixed (Y) volumes
  2. ANTs affine prereg on half-res windowed CT -> full-res voxel DVF
  3. warp X by the affine flow, run LapIRN lvl3 (X_affine, Y) -> deformable flow
  4. compose affine (outer) with the deformable field (inner)
  5. convert the composed unit flow to full-res voxel displacements and save

Output: NIfTI, channel-first (3, X, Y, Z), float32, voxel displacements on the
fixed-image grid, identity affine -- the convention of
Code/inference.py:save_disp.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Tuple

# Everything the container writes must land in a writable place: /app and the
# repo copy are read-only for the evaluation user, and $HOME may not exist.
for _var, _val in (
    ("MPLCONFIGDIR", "/tmp/mpl"),
    ("TORCH_HOME", "/tmp/torch"),
    ("XDG_CACHE_HOME", "/tmp/cache"),
    ("HOME", "/tmp"),
):
    os.environ.setdefault(_var, _val)

sys.path.insert(0, os.environ.get("LAPIRN_CODE", "/app/lapirn/Code"))

import affine_reg  # noqa: E402
import Functions  # noqa: E402
import miccai2020_model_stage  # noqa: E402
import my_data  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from config import TrainingConfig  # noqa: E402

DEFAULT_WEIGHTS = Path(os.environ.get("LAPIRN_WEIGHTS", "/app/weights/model.pth"))


def build_config() -> TrainingConfig:
    """The model hyper-parameters of the submitted checkpoint.

    These mirror the overrides at the top of Code/inference.py:main; every other
    field keeps its TrainingConfig default. They must match the checkpoint or
    load_state_dict rejects it.
    """
    cfg = TrainingConfig()
    cfg.in_channel = 4
    cfg.start_channel = 7
    cfg.resblock_expansion = 1
    cfg.n_resblocks = 5
    return cfg


def create_model(
    device: torch.device, cfg: TrainingConfig, model_path: Path
) -> torch.nn.Module:
    """Same three-level construction as Code/inference.py:create_model."""
    model_lvl1 = (
        miccai2020_model_stage.Miccai2020_LDR_laplacian_unit_add_lvl1(
            in_channel=cfg.in_channel,
            n_classes=cfg.n_classes,
            start_channel=cfg.start_channel,
            is_train=True,
            imgshape=cfg.img_shape_4,
            range_flow=cfg.range_flow,
            cost_volume_mode=cfg.cost_volume_mode,
            cost_volume_radius=cfg.cost_volume_radius,
            cost_volume_dilation=cfg.cost_volume_dilation,
            cost_volume_feat_channels=cfg.cost_volume_feat_channels,
            cost_volume_out_channels=cfg.cost_volume_out_channels,
            n_resblocks=cfg.n_resblocks,
            resblock_expansion=cfg.resblock_expansion,
        ).to(device)
        if cfg.use_lvl1
        else None
    )

    model_lvl2 = miccai2020_model_stage.Miccai2020_LDR_laplacian_unit_add_lvl2(
        in_channel=cfg.in_channel,
        n_classes=cfg.n_classes,
        start_channel=cfg.start_channel,
        is_train=True,
        imgshape=cfg.img_shape_2,
        range_flow=cfg.range_flow,
        model_lvl1=model_lvl1,
        n_resblocks=cfg.n_resblocks,
        resblock_expansion=cfg.resblock_expansion,
    ).to(device)

    model = miccai2020_model_stage.Miccai2020_LDR_laplacian_unit_add_lvl3(
        in_channel=cfg.in_channel,
        n_classes=cfg.n_classes,
        start_channel=cfg.start_channel,
        is_train=True,
        imgshape=cfg.img_shape,
        range_flow=cfg.range_flow,
        model_lvl2=model_lvl2,
        n_resblocks=cfg.n_resblocks,
        resblock_expansion=cfg.resblock_expansion,
        use_seg_pet_head=cfg.use_seg_pet_head,
        seg_pet_head_channels=cfg.seg_pet_head_channels,
        use_bone_head=cfg.use_seg_bone_head,
        bone_head_channels=cfg.seg_bone_head_channels,
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def load_pair(
    fixed_ct_path: Path,
    fixed_pet_path: Path,
    moving_ct_path: Path,
    moving_pet_path: Path,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Read one pair and normalise it exactly as Code/inference.py:load_val_pair.

    Returns (X, Y) of shape (1, 2, H, W, D): X = moving (follow-up),
    Y = fixed (baseline); channel 0 = CT, channel 1 = PET.
    """

    def load(path: Path) -> np.ndarray:
        return nib.load(str(path)).get_fdata().astype(np.float32)

    x_ct_raw, y_ct_raw = load(moving_ct_path), load(fixed_ct_path)

    x_mask = my_data.get_body_mask(x_ct_raw)
    y_mask = my_data.get_body_mask(y_ct_raw)

    x_ct_raw = my_data.apply_body_mask(x_ct_raw, x_mask, fill_value=my_data.CT_AIR_HU)
    y_ct_raw = my_data.apply_body_mask(y_ct_raw, y_mask, fill_value=my_data.CT_AIR_HU)

    x_pet_raw = my_data.apply_body_mask(load(moving_pet_path), x_mask, fill_value=0.0)
    y_pet_raw = my_data.apply_body_mask(load(fixed_pet_path), y_mask, fill_value=0.0)

    def t(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(arr).unsqueeze(0)

    x = torch.cat([t(my_data.norm_ct(x_ct_raw)), t(my_data.norm_pet(x_pet_raw))], dim=0)
    y = torch.cat([t(my_data.norm_ct(y_ct_raw)), t(my_data.norm_pet(y_pet_raw))], dim=0)
    return x.unsqueeze(0).float(), y.unsqueeze(0).float()


def affine_flow(
    fixed_ct_path: Path,
    moving_ct_path: Path,
    cfg: TrainingConfig,
    device: torch.device,
) -> torch.Tensor:
    """ANTs affine prereg as a unit flow of shape (1, H, W, D, 3).

    compute_affine_dvf is called directly rather than affine_reg.get_affine_dvf:
    the latter reads/writes a cache under the repo, which is read-only here and
    meaningless for a one-shot run.
    """
    dvf = affine_reg.compute_affine_dvf(
        fixed_ct_path=fixed_ct_path,
        moving_ct_path=moving_ct_path,
        make_lowres_ants_image_fn=affine_reg.make_lowres_ants_image,
        preprocess_ct_fn=affine_reg.preprocess_ct,
        ants_affine_to_fullres_voxel_disp_fn=affine_reg.ants_affine_to_fullres_voxel_disp,
    )
    dvf_tensor = affine_reg.dvf_to_tensor(dvf, device)
    h, w, d = cfg.img_shape
    d_h = dvf_tensor[:, 0] / (h / 2.0)
    d_w = dvf_tensor[:, 1] / (w / 2.0)
    d_d = dvf_tensor[:, 2] / (d / 2.0)
    return torch.stack([d_d, d_w, d_h], dim=1).permute(0, 2, 3, 4, 1)


def compose(
    flow_outer: torch.Tensor, flow_inner: torch.Tensor, grid: torch.Tensor
) -> torch.Tensor:
    """Compose two unit flows (..., 3) into one; returns channel-first (1,3,...).

    `flow_outer` is applied after `flow_inner` -- the affine goes outside the
    network field, so the result is the TOTAL transform from the original moving
    image onto the fixed grid.
    """
    deform_grid = grid + flow_inner
    affine_grid_ch = (grid + flow_outer).permute(0, 4, 1, 2, 3)
    composed = torch.nn.functional.grid_sample(
        affine_grid_ch,
        deform_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    ).permute(0, 2, 3, 4, 1)
    return (composed - grid).permute(0, 4, 1, 2, 3)


def save_disp(total_unit_flow: torch.Tensor, out_path: Path) -> None:
    """Write the submission field: (3, X, Y, Z) float32 voxel displacements.

    Mirrors Code/inference.py:save_disp -- unit flow -> full-res voxel units,
    then the channel axis is reversed so channel 0 corresponds to the first
    spatial axis, which is the order the organizers' warper expects.
    """
    voxel = Functions.transform_unit_flow_to_flow_cuda(
        total_unit_flow.permute(0, 2, 3, 4, 1).clone()
    ).permute(0, 4, 1, 2, 3)
    disp_np = voxel[0].detach().cpu().numpy().astype(np.float32)[::-1].copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(disp_np, np.eye(4)), str(out_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PSMAReg LapIRN inference: one PET+CT set -> displacement field."
    )
    parser.add_argument("fixed_ct", type=Path, help="baseline CT  (..._0000_00.nii.gz)")
    parser.add_argument("fixed_pet", type=Path, help="baseline PET (..._0001_00.nii.gz)")
    parser.add_argument("moving_ct", type=Path, help="follow-up CT  (..._0000_01.nii.gz)")
    parser.add_argument("moving_pet", type=Path, help="follow-up PET (..._0001_01.nii.gz)")
    parser.add_argument("output_disp", type=Path, help="where to write the field")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.time()

    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device(args.device)
    cfg = build_config()

    grid = Functions.generate_grid_unit(cfg.img_shape)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).to(device).float()
    transform = miccai2020_model_stage.SpatialTransform_unit().to(device)

    model = create_model(device, cfg, args.weights)

    X, Y = load_pair(args.fixed_ct, args.fixed_pet, args.moving_ct, args.moving_pet)
    X, Y = X.to(device), Y.to(device)

    flow_affine = affine_flow(args.fixed_ct, args.moving_ct, cfg, device)
    X_affine = transform(X, flow_affine, grid)

    with torch.no_grad():
        F_X_Y, _, _, _, _, _, _ = model(X_affine, Y)

    total_unit_flow = compose(flow_affine, F_X_Y.permute(0, 2, 3, 4, 1), grid)
    save_disp(total_unit_flow, args.output_disp)

    print(f"wrote {args.output_disp} in {time.time() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
