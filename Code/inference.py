import zipfile
from pathlib import Path
from typing import Dict

import affine_reg
import Functions
import miccai2020_model_stage
import my_data
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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
results_csv = Path("/home/iml/fryderyk.koegl/data/PSMAReg/results.csv")


def compress_to_zip(source_dir: Path, output_zip: Path) -> None:
    files = list(source_dir.rglob("*"))
    files = [f for f in files if f.is_file()]

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in files:
            arcname = file.relative_to(source_dir)
            zf.write(file, arcname)
            print(f"Added: {arcname}")

    print(f"\nDone! Zip saved to: {output_zip}")


class SpatialTransformer(torch.nn.Module):
    """
    N-D Spatial Transformer
    Obtained from https://github.com/voxelmorph/voxelmorph
    """

    def __init__(self, size, mode="bilinear"):
        super().__init__()
        self.mode = mode
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(*vectors, indexing="ij")
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        self.register_buffer("grid", grid)

    def forward(self, src: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        new_locs = self.grid + flow
        shape = flow.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]
        return F.grid_sample(src, new_locs, align_corners=False, mode=self.mode)


def load_val_pair(val_image_dir: Path, case_id: str) -> Dict[str, torch.Tensor]:
    """Load fixed (00) + moving (01) val pair."""

    def load_ct(tp: str) -> np.ndarray:
        path = val_image_dir / f"PSMARegPSMA_{case_id}_0000_{tp}.nii.gz"
        return my_data.nib.load(str(path)).get_fdata().astype(np.float32)

    def load_pet(tp: str) -> np.ndarray:
        path = val_image_dir / f"PSMARegPSMA_{case_id}_0001_{tp}.nii.gz"
        return my_data.nib.load(str(path)).get_fdata().astype(np.float32)

    x_ct_raw = load_ct("01")
    y_ct_raw = load_ct("00")

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


def create_model(
    device: torch.device, cfg: TrainingConfig, model_path: Path
) -> torch.nn.Module:
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
    disp_np = disp_half.squeeze(0).cpu().numpy().astype(np.float32)
    disp_np = disp_np[::-1].copy()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"disp_{case_id}_00_{case_id}_01.nii.gz"
    my_data.nib.save(my_data.nib.Nifti1Image(disp_np, np.eye(4)), str(out_path))


def multilabel_dice(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean Dice over shared nonzero labels. pred/target: (H, W, D) int."""
    labels = torch.unique(target)
    labels = labels[labels != 0]
    dices = []
    for lbl in labels:
        p = (pred == lbl).float()
        t = (target == lbl).float()
        denom = p.sum() + t.sum()
        if denom == 0:
            continue
        dices.append((2.0 * (p * t).sum() / denom).item())
    return float(np.mean(dices)) if dices else float("nan")


def append_results_to_csv(
    csv_path: Path,
    model_name: str,
    dices: Dict[str, float],
) -> None:
    """Append one experiment row to the results CSV, creating it if needed."""
    avg_dice = float(np.nanmean(list(dices.values())))
    row: Dict[str, object] = {"model": model_name, "avg_dice": avg_dice}
    row.update({f"dice_{k}": v for k, v in dices.items()})

    new_row_df = pd.DataFrame([row])

    if csv_path.exists():
        existing_df = pd.read_csv(csv_path)
        combined_df = pd.concat([existing_df, new_row_df], ignore_index=True)
    else:
        combined_df = new_row_df

    combined_df.to_csv(csv_path, index=False)
    tqdm.tqdm.write(f"results saved → {csv_path}")


def main() -> None:
    cfg = TrainingConfig()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # cfg.start_channel = 7
    cfg.cache_dir = val_cache_dir

    model_path = Path(
        "/home/iml/fryderyk.koegl/data/PSMAReg/models/PSMAReg_LapIRN_intelligent-bug-730_stagelvl3_best.pth"
    )
    model_name = model_path.stem

    out_dir = Path("/home/iml/fryderyk.koegl/data/PSMAReg/submission_and_debug")
    seg_dir = Path(
        "/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/segmentations"
    )

    model = create_model(device, cfg, model_path)

    grid_full = Functions.generate_grid_unit(cfg.img_shape)
    grid_full = (
        torch.from_numpy(np.reshape(grid_full, (1,) + grid_full.shape))
        .to(device)
        .float()
    )

    transform = miccai2020_model_stage.SpatialTransform_unit().to(device)
    transform_nearest = miccai2020_model_stage.SpatialTransformNearest_unit().to(device)

    dices: Dict[str, float] = {}
    for case_id in tqdm.tqdm(val_subjects, desc="inference"):
        with torch.no_grad():
            dices[case_id] = process_subject(
                case_id,
                val_image_dir,
                out_dir,
                model,
                transform,
                grid_full,
                cfg,
                device,
                transform_nearest,
                seg_dir,
            )

    avg = float(np.nanmean(list(dices.values())))
    tqdm.tqdm.write(f"average dice: {avg:.4f}")

    append_results_to_csv(results_csv, model_name, dices)

    # compress_to_zip(out_dir / "submission", out_dir / "submission.zip")


def process_subject(
    case_id: str,
    val_image_dir: Path,
    out_dir: Path,
    model: torch.nn.Module,
    transform: torch.nn.Module,
    grid_full: torch.Tensor,
    cfg: TrainingConfig,
    device: torch.device,
    transform_nearest: torch.nn.Module,
    seg_dir: Path,
) -> float:
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

    deform_grid = grid_full + F_X_Y.permute(0, 2, 3, 4, 1)
    affine_grid = grid_full + flow_affine

    affine_grid_ch = affine_grid.permute(0, 4, 1, 2, 3)
    composed_grid = torch.nn.functional.grid_sample(
        affine_grid_ch,
        deform_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    ).permute(0, 2, 3, 4, 1)

    total_unit_flow = (composed_grid - grid_full).permute(0, 4, 1, 2, 3)

    total_voxel = Functions.transform_unit_flow_to_flow_cuda(
        total_unit_flow.permute(0, 2, 3, 4, 1).clone()
    ).permute(0, 4, 1, 2, 3)

    disp_half = torch.nn.functional.interpolate(
        total_voxel, scale_factor=0.5, mode="trilinear", align_corners=False
    )

    disp_up = torch.nn.functional.interpolate(
        disp_half, scale_factor=2, mode="trilinear", align_corners=False
    )

    dice_their = np.nan

    if case_id != "0024":
        _, _, hh, ww, dd = disp_up.shape
        disp_up_unit = disp_up.clone()
        disp_up_unit[:, 0] = disp_up_unit[:, 0] / ((dd - 1) / 2.0)
        disp_up_unit[:, 1] = disp_up_unit[:, 1] / ((ww - 1) / 2.0)
        disp_up_unit[:, 2] = disp_up_unit[:, 2] / ((hh - 1) / 2.0)

        def load_seg(tp: str) -> torch.Tensor:
            path = seg_dir / f"{case_id}_{tp}.nii"
            arr = my_data.nib.load(str(path)).get_fdata().astype(np.int16)
            return torch.from_numpy(arr)[None, None].to(device).float()

        seg_moving = load_seg("01")
        seg_fixed = load_seg("00")
        seg_fixed_i = seg_fixed[0, 0].round().long()

        their_st = SpatialTransformer(size=cfg.img_shape, mode="nearest").to(device)
        disp_their = disp_up.flip(1)
        seg_their = their_st(seg_moving, disp_their)
        dice_their = multilabel_dice(seg_their[0, 0].round().long(), seg_fixed_i)

        dice_before = multilabel_dice(seg_moving[0, 0].round().long(), seg_fixed_i)
        tqdm.tqdm.write(f"{case_id}: before={dice_before:.4f};\tafter={dice_their:.4f}")

    if False:
        # --- DEBUG: verify submitted field via evaluator pipeline ---
        debug_dir = out_dir / "debug_compose"
        X_moving = X[:, 0:1]
        X_submitted = transform(X, disp_up_unit.permute(0, 2, 3, 4, 1), grid_full)[
            :, 0:1
        ]
        Y_fixed = Y[:, 0:1]
        my_data.save_volume(
            X_moving, out_dir=debug_dir, epoch=0, name=f"{case_id}_0_moving"
        )
        my_data.save_volume(
            X_submitted, out_dir=debug_dir, epoch=0, name=f"{case_id}_1_submitted"
        )
        my_data.save_volume(
            Y_fixed, out_dir=debug_dir, epoch=0, name=f"{case_id}_2_fixed"
        )
        # --- END DEBUG ---

    save_disp(disp_half, out_dir / "submission", case_id)

    return dice_their


if __name__ == "__main__":
    main()
