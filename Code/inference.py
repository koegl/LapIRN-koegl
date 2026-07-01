import shutil
from pathlib import Path

import affine_reg
import Functions
import miccai2020_model_stage
import my_data
import numpy as np
import torch
import tqdm
from config import TrainingConfig

# --- variables (define here, no argparse) ---
val_image_dir = Path("/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/imagesVal")
val_cache_dir = Path("/home/iml/fryderyk.koegl/data/PSMAReg/affine_cache_val")
val_subjects = [
    "0001",
    "0003",
    "0005",
    "0007",
    "0008",
    "0009",
    "0013",
    "0021",
    "0024",
    "0029",
    "0031",
    "0033",
    "0034",
    "0035",
    "0036",
    "0038",
    "0039",
    "0042",
    "0047",
    "0048",
]


def load_val_pair(val_image_dir: Path, case_id: str) -> torch.Tensor:
    """Load fixed (00) + moving (01) val pair, same preprocessing as training,
    no labels, no augmentation. Returns dict with x (moving) and y (fixed)."""

    def load_ct(tp: str) -> np.ndarray:
        path = val_image_dir / f"PSMARegPSMA_{case_id}_0000_{tp}.nii.gz"
        return my_data.nib.load(str(path)).get_fdata().astype(np.float32)

    def load_pet(tp: str) -> np.ndarray:
        path = val_image_dir / f"PSMARegPSMA_{case_id}_0001_{tp}.nii.gz"
        return my_data.nib.load(str(path)).get_fdata().astype(np.float32)

    x_ct_raw = load_ct("01")  # moving = follow-up
    y_ct_raw = load_ct("00")  # fixed = baseline

    x_mask = my_data.get_body_mask(x_ct_raw)
    y_mask = my_data.get_body_mask(y_ct_raw)

    x_ct_raw = my_data.apply_body_mask(
        x_ct_raw, x_mask, fill_value=float(np.percentile(x_ct_raw, 0.5))
    )
    y_ct_raw = my_data.apply_body_mask(
        y_ct_raw, y_mask, fill_value=float(np.percentile(y_ct_raw, 0.5))
    )

    x_pet_raw = my_data.apply_body_mask(load_pet("01"), x_mask, fill_value=0.0)
    y_pet_raw = my_data.apply_body_mask(load_pet("00"), y_mask, fill_value=0.0)

    def t(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(arr).unsqueeze(0)

    x = torch.cat([t(my_data.norm_ct(x_ct_raw)), t(my_data.norm_pet(x_pet_raw))], dim=0)
    y = torch.cat([t(my_data.norm_ct(y_ct_raw)), t(my_data.norm_pet(y_pet_raw))], dim=0)
    pair = {"x": x.float(), "y": y.float()}
    return pair


def create_model(device, cfg: TrainingConfig):
    model_path = Path(
        "/home/iml/fryderyk.koegl/data/PSMAReg/models/PSMAReg_LapIRN_nimble-perch-653_stagelvl3_best.pth"
    )

    model_lvl1 = miccai2020_model_stage.Miccai2020_LDR_laplacian_unit_add_lvl1(
        in_channel=cfg.in_channel,
        n_classes=cfg.n_classes,
        start_channel=cfg.start_channel,
        is_train=True,
        imgshape=cfg.img_shape_4,
        range_flow=cfg.range_flow,
    ).to(device)
    model_lvl2 = miccai2020_model_stage.Miccai2020_LDR_laplacian_unit_add_lvl2(
        in_channel=cfg.in_channel,
        n_classes=cfg.n_classes,
        start_channel=cfg.start_channel,
        is_train=True,
        imgshape=cfg.img_shape_2,
        range_flow=cfg.range_flow,
        model_lvl1=model_lvl1,
    ).to(device)
    model = miccai2020_model_stage.Miccai2020_LDR_laplacian_unit_add_lvl3(
        in_channel=cfg.in_channel,
        n_classes=cfg.n_classes,
        start_channel=cfg.start_channel,
        is_train=True,
        imgshape=cfg.img_shape,
        range_flow=cfg.range_flow,
        model_lvl2=model_lvl2,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model


def save_disp(disp_half: torch.Tensor, out_dir: Path, case_id: str) -> None:
    disp_np = disp_half.squeeze(0).cpu().numpy().astype(np.float32)  # (3, X, Y, Z)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"disp_{case_id}_00_{case_id}_01.nii.gz"
    my_data.nib.save(my_data.nib.Nifti1Image(disp_np, np.eye(4)), str(out_path))


def main() -> None:
    cfg = TrainingConfig()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # point affine cache at a val-only dir to avoid train collisions
    cfg.cache_dir = val_cache_dir
    out_dir = Path("/home/iml/fryderyk.koegl/data/PSMAReg/submission_and_debug")

    model = create_model(device, cfg)

    grid_full = Functions.generate_grid_unit(cfg.img_shape)
    grid_full = (
        torch.from_numpy(np.reshape(grid_full, (1,) + grid_full.shape))
        .to(device)
        .float()
    )

    transform = miccai2020_model_stage.SpatialTransform_unit().to(device)

    for case_id in tqdm.tqdm(val_subjects, desc="inference"):
        with torch.no_grad():
            process_subject(
                case_id,
                val_image_dir,
                out_dir,
                model,
                transform,
                grid_full,
                cfg,
                device,
            )

    shutil.make_archive(str(out_dir / "submission"), "zip", root_dir=out_dir)


def process_subject(
    case_id, val_image_dir, out_dir, model, transform, grid_full, cfg, device
):
    pair = load_val_pair(val_image_dir, case_id)
    X = pair["x"].unsqueeze(0).to(device).float()
    Y = pair["y"].unsqueeze(0).to(device).float()

    dvf = affine_reg.get_affine_dvf(
        case_id=case_id,
        tp_x="01",
        tp_y="00",
        fixed_ct_path=val_image_dir / f"PSMARegPSMA_{case_id}_0000_00.nii.gz",
        moving_ct_path=val_image_dir / f"PSMARegPSMA_{case_id}_0000_01.nii.gz",
        make_lowres_ants_image_fn=affine_reg.make_lowres_ants_image,
        preprocess_ct_fn=affine_reg.preprocess_ct,
        ants_affine_to_fullres_voxel_disp_fn=affine_reg.ants_affine_to_fullres_voxel_disp,
    )
    dvf = affine_reg.apply_augmentation_to_dvf(
        dvf=dvf, flipped=False, crop_head=0, crop_feet=0
    )
    dvf_tensor = affine_reg.dvf_to_tensor(dvf, device)
    h, w, d = cfg.img_shape
    d_h = dvf_tensor[:, 0] / (h / 2.0)
    d_w = dvf_tensor[:, 1] / (w / 2.0)
    d_d = dvf_tensor[:, 2] / (d / 2.0)
    flow_affine = torch.stack([d_d, d_w, d_h], dim=1).permute(0, 2, 3, 4, 1)

    X_affine = transform(X, flow_affine, grid_full)

    with torch.no_grad():
        F_X_Y, warped, _, _, _, _, _ = model(X_affine, Y)

    if False:
        my_data.save_volume(
            volume=X[:, 0:1, ...],
            out_dir=out_dir / "images",
            epoch=0,
            name=f"{case_id}_01",
        )
        my_data.save_volume(
            volume=Y[:, 0:1, ...],
            out_dir=out_dir / "images",
            epoch=0,
            name=f"{case_id}_00",
        )
        my_data.save_volume(
            volume=X_affine[:, 0:1, ...],
            out_dir=out_dir / "warped",
            epoch=0,
            name=f"{case_id}_affine",
        )
        my_data.save_volume(
            volume=warped[:, 0:1, ...],
            out_dir=out_dir / "warped",
            epoch=0,
            name=f"{case_id}_warped",
        )

    deform_grid = grid_full + F_X_Y.permute(0, 2, 3, 4, 1)  # (1, H, W, D, 3)
    affine_grid = grid_full + flow_affine  # (1, H, W, D, 3)

    # TODO: verify composition order — warp X by total_unit_flow, overlay on Y,
    # confirm alignment improves. If it warps away, swap sampling order.
    # sample affine_grid (as 3 channels) at deform_grid locations
    affine_grid_ch = affine_grid.permute(0, 4, 1, 2, 3)  # (1, 3, H, W, D)
    composed_grid = torch.nn.functional.grid_sample(
        affine_grid_ch,
        deform_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    ).permute(0, 2, 3, 4, 1)  # (1, H, W, D, 3)

    total_unit_flow = (composed_grid - grid_full).permute(0, 4, 1, 2, 3)  # (1,3,H,W,D)

    if False:
        # --- DEBUG: verify composition order ---
        debug_dir = out_dir / "debug_compose"
        X_moving = X[:, 0:1]  # original moving
        X_affine_only = transform(X, flow_affine, grid_full)[:, 0:1]  # affine only
        X_composed = transform(X, total_unit_flow.permute(0, 2, 3, 4, 1), grid_full)[
            :, 0:1
        ]  # full composed
        Y_fixed = Y[:, 0:1]  # target

        my_data.save_volume(
            X_moving, out_dir=debug_dir, epoch=0, name=f"{case_id}_0_moving"
        )
        my_data.save_volume(
            X_affine_only, out_dir=debug_dir, epoch=0, name=f"{case_id}_1_affine"
        )
        my_data.save_volume(
            X_composed, out_dir=debug_dir, epoch=0, name=f"{case_id}_2_composed"
        )
        my_data.save_volume(
            Y_fixed, out_dir=debug_dir, epoch=0, name=f"{case_id}_3_fixed"
        )
        # --- END DEBUG ---

    total_voxel = Functions.transform_unit_flow_to_flow_cuda(
        total_unit_flow.permute(0, 2, 3, 4, 1).clone()
    ).permute(0, 4, 1, 2, 3)  # (1, 3, H, W, D)

    disp_half = torch.nn.functional.interpolate(
        total_voxel, scale_factor=0.5, mode="trilinear", align_corners=False
    )  # (1, 3, 96, 96, 144)

    save_disp(disp_half, out_dir / "submission", case_id)


if __name__ == "__main__":
    main()
