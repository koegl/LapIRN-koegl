import os
from pathlib import Path
from typing import Callable, Dict, Optional

import affine_reg
import mlflow
import my_data
import numpy as np
import poly_affine_reg
import psutil
import synthetic
import torch
import torch.nn.functional as F
import tqdm
import utils
from config import TrainingConfig
from Functions import (
    generate_grid,
    generate_grid_unit,
    transform_unit_flow_to_flow_cuda,
)
from miccai2020_model_stage import (
    Miccai2020_LDR_laplacian_unit_add_lvl1,
    Miccai2020_LDR_laplacian_unit_add_lvl2,
    SpatialTransform_unit,
    SpatialTransformNearest_unit,
    jacobian_determinant,
    multi_resolution_NCC,
    neg_Jdet_loss,
    smoothloss,
)
from torch.utils import data as torch_data


def evaluate_lvl2(
    model: Miccai2020_LDR_laplacian_unit_add_lvl2,
    valid_generator: torch_data.DataLoader,
    config: TrainingConfig,
    device: torch.device,
    loss_similarity_ct: multi_resolution_NCC,
    loss_similarity_pet: multi_resolution_NCC,
    loss_smooth: Callable,
    loss_Jdet: Callable,
    transform: SpatialTransform_unit,
    grid_2: torch.Tensor,
    epoch: int,
    val_interval: int,
    saved_initial: bool,
    is_last: bool,
) -> Dict[str, float]:

    model.eval()
    val_losses: Dict[str, float] = {
        "loss": 0.0,
        "ncc_ct": 0.0,
        "ncc_pet": 0.0,
        "smooth": 0.0,
        "dice_ct": 0.0,
        "dice_pet": 0.0,
        "jacobian": 0.0,
        "ndv": 0.0,
    }
    n_batches = 0

    transform_nearest = SpatialTransformNearest_unit().to(device)
    for param in transform_nearest.parameters():
        param.requires_grad = False

    grid_full = generate_grid_unit(config.img_shape)
    grid_full = (
        torch.from_numpy(np.reshape(grid_full, (1,) + grid_full.shape))
        .to(device)
        .float()
    )

    n_dice_ct = 0
    n_dice_pet = 0

    with torch.no_grad():
        saved = False
        for batch in valid_generator:
            X = batch["x"].to(device).float()
            Y = batch["y"].to(device).float()
            X_lbl_ct = batch["x_label_ct"].to(device)
            X_lbl_pet = batch["x_label_pet"].to(device)
            Y_lbl_ct = batch["y_label_ct"].to(device)
            Y_lbl_pet = batch["y_label_pet"].to(device)

            case_id: str = batch["case_id"][0]
            tp_y: str = batch["tp_y"][0]
            tp_x: str = batch["tp_x"][0]

            flow_affine = affine_reg.create_affine_flow(
                config=config,
                device=device,
                case_id=case_id,
                tp_x=tp_x,
                tp_y=tp_y,
                aug_flipped=batch["aug_flipped"],
                aug_crop_head=batch["aug_crop_head"],
                aug_crop_feet=batch["aug_crop_feet"],
            )

            if config.use_poly_affine is False:
                X_prereg = transform(X, flow_affine, grid_full)
            else:
                poly_dvf = poly_affine_reg.get_polyaffine_dvf(
                    case_id=case_id,
                    tp_x=tp_x,
                    tp_y=tp_y,
                    fixed_seg_path=config.data_dir
                    / "labelsTr"
                    / f"{case_id}_{tp_y}.nii.gz",
                    moving_seg_path=config.data_dir
                    / "labelsTr"
                    / f"{case_id}_{tp_x}.nii.gz",
                    get_affine_dvf_fn=lambda: affine_reg.get_affine_dvf(
                        case_id=case_id,
                        tp_x=tp_x,
                        tp_y=tp_y,
                        fixed_ct_path=config.data_dir
                        / "imagesTr"
                        / f"PSMARegPSMA_{case_id}_0000_{tp_y}.nii.gz",
                        moving_ct_path=config.data_dir
                        / "imagesTr"
                        / f"PSMARegPSMA_{case_id}_0000_{tp_x}.nii.gz",
                        make_lowres_ants_image_fn=affine_reg.make_lowres_ants_image,
                        preprocess_ct_fn=affine_reg.preprocess_ct,
                        ants_affine_to_fullres_voxel_disp_fn=(
                            affine_reg.ants_affine_to_fullres_voxel_disp
                        ),
                    ),
                    cfg=config,
                    device=device,
                )
                flow_poly = poly_affine_reg.create_polyaffine_flow(
                    poly_dvf=poly_dvf,
                    aug_flipped=batch["aug_flipped"],
                    aug_crop_head=batch["aug_crop_head"],
                    aug_crop_feet=batch["aug_crop_feet"],
                    cfg=config,
                    device=device,
                )
                flow_total = poly_affine_reg.compose_flows(
                    flow_affine, flow_poly, grid_full
                )

                X_prereg = transform(X, flow_total, grid_full)

            F_X_Y, X_Y, Y_4x, F_xy, _, _ = model(X_prereg, Y)
            if epoch % (val_interval * 25) == 0 or is_last:
                if not saved_initial:
                    zero_disp = torch.zeros_like(F_X_Y)
                    x_ref = model.transform(
                        X, zero_disp.permute(0, 2, 3, 4, 1), model.grid_1
                    )
                    y_ref = model.transform(
                        Y, zero_disp.permute(0, 2, 3, 4, 1), model.grid_1
                    )
                    my_data.save_volume(
                        volume=x_ref[:, 0:1, ...],
                        out_dir=config.save_dir / "initial",
                        epoch=epoch,
                        name="x_ref_ct_lvl2",
                    )
                    my_data.save_volume(
                        volume=y_ref[:, 0:1, ...],
                        out_dir=config.save_dir / "initial",
                        epoch=epoch,
                        name="y_ref_ct_lvl2",
                    )
                    saved_initial = True

                if saved is False:
                    ct = X_Y[:, 0:1, :, :, :]
                    my_data.save_volume(
                        volume=ct,
                        out_dir=config.save_dir / "warped",
                        epoch=epoch,
                        name="warped_ct_lvl2",
                    )
                    saved = True

            X_Y_ct = X_Y[:, 0:1, ...]
            X_Y_pet = X_Y[:, 1:2, ...]
            Y_4x_ct = Y_4x[:, 0:1, ...]
            Y_4x_pet = Y_4x[:, 1:2, ...]

            loss_ncc_ct = loss_similarity_ct(X_Y_ct, Y_4x_ct)
            loss_ncc_pet = loss_similarity_pet(X_Y_pet, Y_4x_pet)
            loss_multiNCC = config.w_ct * loss_ncc_ct + config.w_pet * loss_ncc_pet

            F_X_Y_norm = transform_unit_flow_to_flow_cuda(
                F_X_Y.permute(0, 2, 3, 4, 1).clone()
            )
            jac_det = jacobian_determinant(F_X_Y_norm)
            ndv = utils.compute_ndv(jac_det)
            loss_jacobian = loss_Jdet(F_X_Y_norm, grid_2)

            _, _, x, y, z = F_xy.shape
            F_xy[:, 0, :, :, :] = F_xy[:, 0, :, :, :] * (z - 1)
            F_xy[:, 1, :, :, :] = F_xy[:, 1, :, :, :] * (y - 1)
            F_xy[:, 2, :, :, :] = F_xy[:, 2, :, :, :] * (x - 1)
            loss_regulation = loss_smooth(F_xy)

            X_lbl_ct = transform_nearest(X_lbl_ct.float(), flow_affine, grid_full)
            X_lbl_pet = transform_nearest(X_lbl_pet.float(), flow_affine, grid_full)

            X_lbl_ct_down = utils.downsample_label(
                X_lbl_ct.to(device), scale_factor=0.5
            )
            Y_lbl_ct_down = utils.downsample_label(
                Y_lbl_ct.to(device), scale_factor=0.5
            )
            X_lbl_pet_down = utils.downsample_label(
                X_lbl_pet.to(device), scale_factor=0.5
            )
            Y_lbl_pet_down = utils.downsample_label(
                Y_lbl_pet.to(device), scale_factor=0.5
            )

            loss_dice_ct = utils.dice_loss_with_grad(
                X_lbl_ct_down, Y_lbl_ct_down, F_X_Y, model.grid_1, transform
            )
            loss_dice_pet = utils.dice_loss_with_grad(
                X_lbl_pet_down, Y_lbl_pet_down, F_X_Y, model.grid_1, transform
            )

            loss = (
                loss_multiNCC
                + config.w_jacobian * loss_jacobian
                + config.w_smooth * loss_regulation
            )
            if loss_dice_ct is not None:
                loss = loss + config.w_dice_ct * loss_dice_ct
            if loss_dice_pet is not None:
                loss = loss + config.w_dice_pet * loss_dice_pet

            val_losses["loss"] += loss.item()
            val_losses["ncc_ct"] += loss_ncc_ct.item()
            val_losses["ncc_pet"] += loss_ncc_pet.item()
            val_losses["smooth"] += loss_regulation.item()
            val_losses["jacobian"] += loss_jacobian.item()
            val_losses["ndv"] += ndv

            if loss_dice_ct is not None:
                val_losses["dice_ct"] += loss_dice_ct.item()
                n_dice_ct += 1
            if loss_dice_pet is not None:
                val_losses["dice_pet"] += loss_dice_pet.item()
                n_dice_pet += 1
            n_batches += 1

    model.train()

    averaged = {
        key: value / n_batches
        for key, value in val_losses.items()
        if key not in ("dice_ct", "dice_pet")
    }
    averaged["dice_ct"] = (
        val_losses["dice_ct"] / n_dice_ct if n_dice_ct > 0 else float("nan")
    )
    averaged["dice_pet"] = (
        val_losses["dice_pet"] / n_dice_pet if n_dice_pet > 0 else float("nan")
    )
    return averaged


def train_lvl2(
    config: TrainingConfig,
    path_model_level1: Path,
    train_generator: torch_data.DataLoader,
    valid_generator: torch_data.DataLoader,
    resume_model_path: Optional[Path] = None,
    resume_optimizer_path: Optional[Path] = None,
) -> Dict[str, Path]:
    print("Training lvl2...")
    proc = psutil.Process(os.getpid())

    steps_per_epoch = len(train_generator)
    total_steps = config.total_steps_lvl2
    val_step_interval = config.val_interval * steps_per_epoch
    unfreeze_step = config.unfreeze_epoch_in_lvl2 * steps_per_epoch

    run_name = utils.get_run_name()

    best_dice_ct = float("inf")
    config.model_save_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = (
        config.model_save_dir
        / f"{config.mlflow_experiment}_{run_name}_stagelvl2_best.pth"
    )
    final_model_path = (
        config.model_save_dir
        / f"{config.mlflow_experiment}_{run_name}_stagelvl2_{total_steps}.pth"
    )
    if (resume_model_path is None) != (resume_optimizer_path is None):
        raise ValueError(
            "For resuming, provide both resume_model_path and "
            "resume_optimizer_path (or neither)."
        )
    best_optimizer_path = (
        config.model_save_dir
        / f"{config.mlflow_experiment}_{run_name}_stagelvl2_best_optimizer.pth"
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_lvl1 = Miccai2020_LDR_laplacian_unit_add_lvl1(
        in_channel=config.in_channel,
        n_classes=config.n_classes,
        start_channel=config.start_channel,
        is_train=True,
        imgshape=config.img_shape_4,
        range_flow=config.range_flow,
    ).to(device)

    print("Loading weight for model_lvl1...", path_model_level1)
    model_lvl1.load_state_dict(torch.load(path_model_level1))

    for param in model_lvl1.parameters():
        param.requires_grad = False

    model = Miccai2020_LDR_laplacian_unit_add_lvl2(
        in_channel=config.in_channel,
        n_classes=config.n_classes,
        start_channel=config.start_channel,
        is_train=True,
        imgshape=config.img_shape_2,
        range_flow=config.range_flow,
        model_lvl1=model_lvl1,
    ).to(device)

    loss_similarity_ct = multi_resolution_NCC(win=config.lvl2_ncc_win, scale=2)
    loss_similarity_pet = multi_resolution_NCC(win=config.lvl2_ncc_win, scale=2)
    loss_smooth = smoothloss
    loss_Jdet = neg_Jdet_loss

    transform = SpatialTransform_unit().to(device)
    transform_nearest = SpatialTransformNearest_unit().to(device)

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True
    for param in transform_nearest.parameters():
        param.requires_grad = False

    grid_2 = generate_grid(config.img_shape_2)
    grid_2 = (
        torch.from_numpy(np.reshape(grid_2, (1,) + grid_2.shape)).to(device).float()
    )

    grid_full = generate_grid_unit(config.img_shape)
    grid_full = (
        torch.from_numpy(np.reshape(grid_full, (1,) + grid_full.shape))
        .to(device)
        .float()
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr_lvl2)

    config.save_dir.mkdir(parents=True, exist_ok=True)

    lossall = np.zeros((4, total_steps))

    start_global_step = 0
    if resume_model_path is not None:
        print("Resuming lvl2 from...", resume_model_path)
        model.load_state_dict(torch.load(resume_model_path, map_location=device))
        opt_ckpt = torch.load(resume_optimizer_path, map_location=device)
        optimizer.load_state_dict(opt_ckpt["optimizer"])
        start_global_step = opt_ckpt["global_step"]
        best_dice_ct = opt_ckpt["best_dice_ct"]

    train_iter = utils.cycle(train_generator)

    if config.overfit is False:
        pbar = tqdm.tqdm(
            total=total_steps, initial=start_global_step, desc="lvl2 training"
        )

    saved_initial: bool = False

    epoch_metrics: Dict[str, float] = {}
    n_gated = 0

    # re-apply unfreeze if resuming past the threshold
    if start_global_step >= unfreeze_step:
        model.unfreeze_modellvl1()

    for global_step in range(start_global_step, total_steps):
        epoch = global_step // steps_per_epoch
        is_epoch_start = global_step % steps_per_epoch == 0
        is_epoch_end = global_step % steps_per_epoch == steps_per_epoch - 1
        is_last_step = global_step == total_steps - 1

        if is_epoch_start:
            epoch_metrics = {}
            n_gated = 0

        if global_step == unfreeze_step:
            model.unfreeze_modellvl1()

        batch = next(train_iter)

        is_synthetic = bool(batch["is_synthetic"][0])

        Y = batch["y"].to(device).float()
        Y_lbl_ct = batch["y_label_ct"].to(device)
        Y_lbl_pet = batch["y_label_pet"].to(device)

        if is_synthetic:
            if config.overfit_synthetic:
                if config.frozen_synthetic_path.exists():
                    frozen = synthetic.load_frozen_pair(
                        config.frozen_synthetic_path, device
                    )
                else:
                    X_full, X_lbl_ct, X_lbl_pet, gt_unit = (
                        synthetic.generate_synthetic_moving_cached(
                            source=Y,
                            source_label_ct=Y_lbl_ct,
                            source_label_pet=Y_lbl_pet,
                            body_map=batch["body_map"].to(device),
                            device=device,
                        )
                    )
                    frozen = {
                        "x": X_full,
                        "x_label_ct": X_lbl_ct,
                        "x_label_pet": X_lbl_pet,
                        "gt_unit": gt_unit,
                    }
                    synthetic.save_frozen_pair(frozen, config.frozen_synthetic_path)
                X_full = frozen["x"]
                X_lbl_ct = frozen["x_label_ct"]
                X_lbl_pet = frozen["x_label_pet"]
                gt_unit = frozen["gt_unit"]
            else:
                X_full, X_lbl_ct, X_lbl_pet, gt_unit = (
                    synthetic.generate_synthetic_moving_cached(
                        source=Y,
                        source_label_ct=Y_lbl_ct,
                        source_label_pet=Y_lbl_pet,
                        body_map=batch["body_map"].to(device),
                        device=device,
                    )
                )
            X_prereg = X_full
            X = X_prereg
        else:
            X = batch["x"].to(device).float()
            X_lbl_ct = batch["x_label_ct"].to(device)
            X_lbl_pet = batch["x_label_pet"].to(device)
            gt_unit = None

            case_id: str = batch["case_id"][0]
            tp_y: str = batch["tp_y"][0]
            tp_x: str = batch["tp_x"][0]

            flow_affine = affine_reg.create_affine_flow(
                config=config,
                device=device,
                case_id=case_id,
                tp_x=tp_x,
                tp_y=tp_y,
                aug_flipped=batch["aug_flipped"],
                aug_crop_head=batch["aug_crop_head"],
                aug_crop_feet=batch["aug_crop_feet"],
            )

            if config.use_poly_affine is False:
                X_prereg = transform(X, flow_affine, grid_full)
            else:
                poly_dvf = poly_affine_reg.get_polyaffine_dvf(
                    case_id=case_id,
                    tp_x=tp_x,
                    tp_y=tp_y,
                    fixed_seg_path=config.data_dir
                    / "labelsTr"
                    / f"PSMARegPSMA_{case_id}_0000_{tp_y}.nii.gz",
                    moving_seg_path=config.data_dir
                    / "labelsTr"
                    / f"PSMARegPSMA_{case_id}_0000_{tp_x}.nii.gz",
                    get_affine_dvf_fn=lambda: affine_reg.get_affine_dvf(
                        case_id=case_id,
                        tp_x=tp_x,
                        tp_y=tp_y,
                        fixed_ct_path=config.data_dir
                        / "imagesTr"
                        / f"PSMARegPSMA_{case_id}_0000_{tp_y}.nii.gz",
                        moving_ct_path=config.data_dir
                        / "imagesTr"
                        / f"PSMARegPSMA_{case_id}_0000_{tp_x}.nii.gz",
                        make_lowres_ants_image_fn=affine_reg.make_lowres_ants_image,
                        preprocess_ct_fn=affine_reg.preprocess_ct,
                        ants_affine_to_fullres_voxel_disp_fn=(
                            affine_reg.ants_affine_to_fullres_voxel_disp
                        ),
                    ),
                    cfg=config,
                    device=device,
                )
                flow_poly = poly_affine_reg.create_polyaffine_flow(
                    poly_dvf=poly_dvf,
                    aug_flipped=batch["aug_flipped"],
                    aug_crop_head=batch["aug_crop_head"],
                    aug_crop_feet=batch["aug_crop_feet"],
                    cfg=config,
                    device=device,
                )
                flow_total = poly_affine_reg.compose_flows(
                    flow_affine, flow_poly, grid_full
                )

                X_prereg = transform(X, flow_total, grid_full)

        F_X_Y, X_Y, Y_4x, F_xy, _, _ = model(X_prereg, Y)

        if config.overfit is True and saved_initial is False:
            my_data.save_initial(
                model,
                X,
                X_prereg,
                Y_4x,
                F_X_Y,
                config,
                epoch,
                run_name,
                batch,
                level=2,
            )
            saved_initial = True

        if config.overfit is True and (
            (is_epoch_start and (epoch == 0 or epoch % 20 == 0)) or is_last_step
        ):
            ct = X_Y[:, 0:1, :, :, :]
            my_data.save_volume(
                volume=ct,
                out_dir=config.save_dir / "warped",
                epoch=epoch,
                name=f"warped_ct_lvl2_{run_name}_{batch['case_id'][0]}",
            )
        F_X_Y = F_X_Y.float()
        X_Y = X_Y.float()
        Y_4x = Y_4x.float()
        F_xy = F_xy.float()

        X_Y_ct = X_Y[:, 0:1, ...]
        X_Y_pet = X_Y[:, 1:2, ...]
        Y_4x_ct = Y_4x[:, 0:1, ...]
        Y_4x_pet = Y_4x[:, 1:2, ...]

        F_X_Y_norm = transform_unit_flow_to_flow_cuda(
            F_X_Y.permute(0, 2, 3, 4, 1).clone()
        )
        jac_det = jacobian_determinant(F_X_Y_norm)
        ndv = utils.compute_ndv(jac_det)

        loss_jacobian = loss_Jdet(F_X_Y_norm, grid_2)

        # reg2 - use velocity
        _, _, x, y, z = F_xy.shape
        F_xy[:, 0, :, :, :] = F_xy[:, 0, :, :, :] * (z - 1)
        F_xy[:, 1, :, :, :] = F_xy[:, 1, :, :, :] * (y - 1)
        F_xy[:, 2, :, :, :] = F_xy[:, 2, :, :, :] * (x - 1)
        loss_regulation = loss_smooth(F_xy)

        # synthetic labels are already in the (deformed) moving frame; the
        # real branch needs the affine applied first
        if not is_synthetic:
            X_lbl_ct = transform_nearest(X_lbl_ct.float(), flow_affine, grid_full)
            X_lbl_pet = transform_nearest(X_lbl_pet.float(), flow_affine, grid_full)

        if is_synthetic:
            use_dice_pet = True
            use_ncc_pet = True
        else:
            pet_iou = utils.affine_pet_iou(
                batch["x_label_pet"].to(device),
                Y_lbl_pet,
                flow_affine,
                grid_full,
                transform_nearest,
            )
            use_dice_pet = pet_iou >= config.dice_pet_iou_threshold
            use_ncc_pet = pet_iou >= config.dice_pet_iou_threshold
        if not use_dice_pet:
            n_gated += 1

        loss_ncc_ct = loss_similarity_ct(X_Y_ct, Y_4x_ct)
        loss_ncc_pet = loss_similarity_pet(X_Y_pet, Y_4x_pet)
        if use_ncc_pet:
            loss_multiNCC = config.w_ct * loss_ncc_ct + config.w_pet * loss_ncc_pet
        else:
            loss_multiNCC = config.w_ct * loss_ncc_ct

        X_lbl_ct_down = utils.downsample_label(X_lbl_ct.to(device), scale_factor=0.5)
        Y_lbl_ct_down = utils.downsample_label(Y_lbl_ct.to(device), scale_factor=0.5)
        X_lbl_pet_down = utils.downsample_label(X_lbl_pet.to(device), scale_factor=0.5)
        Y_lbl_pet_down = utils.downsample_label(Y_lbl_pet.to(device), scale_factor=0.5)

        if is_last_step and False:
            warped_seg_ct = transform_nearest(
                X_lbl_ct_down.float(),
                F_X_Y.permute(0, 2, 3, 4, 1),
                model.grid_1,
            )

            my_data.save_volume(
                volume=warped_seg_ct.to(torch.int16),
                out_dir=config.save_dir / "warped",
                epoch=epoch,
                name="x_warped_lvl2",
            )
            my_data.save_volume(
                volume=Y_lbl_ct_down.to(torch.int16),
                out_dir=config.save_dir / "warped",
                epoch=epoch,
                name="y_lvl2",
            )

        loss_dice_ct = utils.dice_loss_with_grad(
            X_lbl_ct_down, Y_lbl_ct_down, F_X_Y, model.grid_1, transform
        )
        loss_dice_pet = utils.dice_loss_with_grad(
            X_lbl_pet_down, Y_lbl_pet_down, F_X_Y, model.grid_1, transform
        )

        # update total loss
        loss = (
            loss_multiNCC
            + config.w_jacobian * loss_jacobian
            + config.w_smooth * loss_regulation
        )
        if loss_dice_ct is not None:
            loss = loss + config.w_dice_ct * loss_dice_ct
        if loss_dice_pet is not None and use_dice_pet:
            loss = loss + config.w_dice_pet * loss_dice_pet

        if is_synthetic:
            gt_unit_ds = F.interpolate(
                gt_unit,
                size=F_X_Y.shape[2:],
                mode="trilinear",
                align_corners=True,
            )
            loss_dvf = ((F_X_Y - gt_unit_ds) ** 2).mean()
        else:
            loss_dvf = torch.zeros((), device=device)
        loss = loss + config.w_dvf * loss_dvf

        loss_scaled = loss / config.accumulation_steps
        is_step = (global_step + 1) % config.accumulation_steps == 0 or is_last_step

        utils.optimizer_step_with_guard(
            loss, loss_scaled, optimizer, model, is_step, global_step, level=2
        )

        lossall[:, global_step] = np.array(
            [
                loss.item(),
                loss_multiNCC.item(),
                loss_jacobian.item(),
                loss_regulation.item(),
            ]
        )
        if config.overfit is False:
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                ncc=f"{loss_multiNCC.item():.4f}",
                dice_ct=f"{loss_dice_ct.item():.4f}"
                if loss_dice_ct is not None
                else "n/a",
                dice_pet=f"{loss_dice_pet.item():.4f}"
                if loss_dice_pet is not None
                else "n/a",
                Jdet=f"{loss_jacobian.item():.6f}",
                smo=f"{loss_regulation.item():.4f}",
                dvf=f"{loss_dvf.item():.8f}",
            )
        train_metrics = {
            "train_lvl2/loss": loss.item(),
            "train_lvl2/ncc_ct": loss_ncc_ct.item(),
            "train_lvl2/ncc_pet": loss_ncc_pet.item(),
            "train_lvl2/smooth": loss_regulation.item(),
            "train_lvl2/jacob": loss_jacobian.item(),
            "train_lvl2/ndv": ndv,
            "train_lvl2/dvf": loss_dvf.item(),
        }
        if loss_dice_ct is not None:
            train_metrics["train_lvl2/dice_ct"] = loss_dice_ct.item()
        if loss_dice_pet is not None:
            train_metrics["train_lvl2/dice_pet"] = loss_dice_pet.item()
        mlflow.log_metrics(train_metrics, step=global_step)

        for key, value in train_metrics.items():
            epoch_metrics[key] = epoch_metrics.get(key, 0.0) + value

        if is_epoch_end:
            mlflow.log_metrics(
                {
                    f"{key}_epoch": value / steps_per_epoch
                    for key, value in epoch_metrics.items()
                },
                step=global_step,
            )
            if config.overfit:
                print(
                    f"ep: {epoch} "
                    f"ncc={epoch_metrics['train_lvl2/ncc_ct'] / steps_per_epoch:.4f} "
                    f"dice={epoch_metrics['train_lvl2/dice_ct'] / steps_per_epoch:.4f}"
                )

        if config.overfit is False and (
            global_step % val_step_interval == 0 or is_last_step
        ):
            val_losses = evaluate_lvl2(
                model=model,
                valid_generator=valid_generator,
                config=config,
                device=device,
                loss_similarity_ct=loss_similarity_ct,
                loss_similarity_pet=loss_similarity_pet,
                loss_smooth=loss_smooth,
                loss_Jdet=loss_Jdet,
                transform=transform,
                grid_2=grid_2,
                epoch=epoch,
                val_interval=config.val_interval,
                saved_initial=saved_initial,
                is_last=is_last_step,
            )
            saved_initial = True
            mlflow.log_metrics(
                {
                    f"valid_lvl2/val_{key}": value
                    for key, value in val_losses.items()
                    if not (isinstance(value, float) and np.isnan(value))
                },
                step=global_step,
            )
            tqdm.tqdm.write(
                f"step {global_step} (ep {epoch}) -> val loss {val_losses['loss']:.4f} "
                f"- ncc_ct {val_losses['ncc_ct']:.4f} "
                f"- ncc_pet {val_losses['ncc_pet']:.4f} "
                f"- dice_ct {val_losses['dice_ct']:.4f} "
                f"- dice_pet {val_losses['dice_pet']:.4f}"
            )

            if val_losses["dice_ct"] < best_dice_ct:
                best_dice_ct = val_losses["dice_ct"]
                torch.save(model.state_dict(), best_model_path)
                opt_ckpt = {
                    "optimizer": optimizer.state_dict(),
                    "global_step": global_step,
                    "best_dice_ct": best_dice_ct,
                }
                torch.save(opt_ckpt, best_optimizer_path)
                tqdm.tqdm.write(
                    f"step {global_step}: new best dice_ct {best_dice_ct:.4f} -> saved best"
                )
                print(
                    f"step {global_step}: new best dice_ct {best_dice_ct:.4f} -> saved best"
                )

        if config.overfit is False:
            pbar.update(1)

    if config.overfit is False:
        pbar.close()

    torch.save(model.state_dict(), final_model_path)
    np.save(
        config.save_dir
        / f"loss_{config.mlflow_experiment}_stagelvl2_{total_steps}.npy",
        lossall,
    )

    result = {
        "final": final_model_path,
        "best": best_model_path,
        "best_optimizer": best_optimizer_path,
    }
    return result
