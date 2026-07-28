from pathlib import Path
from typing import Callable, Dict, Optional

import affine_reg
import instance_opt
import jacobian
import my_data
import numpy as np
import poly_affine_reg
import synthetic
import torch
import tqdm
import utils
from config import TrainingConfig
from Functions import (
    generate_grid,
    generate_grid_unit,
    transform_unit_flow_to_flow_cuda,
)
from miccai2020_model_stage import (
    NCC,
    Miccai2020_LDR_laplacian_unit_add_lvl1,
    Miccai2020_LDR_laplacian_unit_add_lvl2,
    Miccai2020_LDR_laplacian_unit_add_lvl3,
    SpatialTransform_unit,
    SpatialTransformNearest_unit,
    multi_resolution_NCC_fast,
    smoothloss,
)
from torch.utils import data as torch_data


def evaluate_lvl3(
    model: Miccai2020_LDR_laplacian_unit_add_lvl3,
    val_generator: torch_data.DataLoader,
    config: TrainingConfig,
    device: torch.device,
    loss_similarity_ct: multi_resolution_NCC_fast,
    loss_similarity_pet: multi_resolution_NCC_fast,
    loss_smooth: Callable,
    loss_Jdet: Callable,
    transform: SpatialTransform_unit,
    grid: torch.Tensor,
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
        "mtv_bias": 0.0,
        "tlg_bias": 0.0,
        "jacobian_tumor": 0.0,
        "rigidity": 0.0,
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
        for batch in val_generator:
            X = batch["x"].to(device).float()
            Y = batch["y"].to(device).float()
            X_lbl_ct = batch["x_label_ct"].to(device)
            X_lbl_pet = batch["x_label_pet"].to(device)
            Y_lbl_ct = batch["y_label_ct"].to(device)
            Y_lbl_pet = batch["y_label_pet"].to(device)

            if "_" in batch["case_id"][0]:
                case_id_x, case_id_y = batch["case_id"][0].split("_")
            else:
                case_id_x = batch["case_id"][0]
                case_id_y = batch["case_id"][0]

            tp_y: str = batch["tp_y"][0]
            tp_x: str = batch["tp_x"][0]

            flow_prereg = affine_reg.create_affine_flow(
                config=config,
                device=device,
                case_id_x=case_id_x,
                case_id_y=case_id_y,
                tp_x=tp_x,
                tp_y=tp_y,
                aug_flipped=batch["aug_flipped"],
                aug_crop_head=batch["aug_crop_head"],
                aug_crop_feet=batch["aug_crop_feet"],
                aug_crop_head_fixed=batch["aug_crop_head_fixed"],
                aug_crop_feet_fixed=batch["aug_crop_feet_fixed"],
            )

            if config.use_poly_affine is False:
                X_prereg = transform(X, flow_prereg, grid_full)
            else:
                poly_dvf = poly_affine_reg.get_polyaffine_dvf(
                    case_id_x=case_id_x,
                    case_id_y=case_id_y,
                    tp_x=tp_x,
                    tp_y=tp_y,
                    fixed_seg_path=config.data_dir
                    / "labelsTr"
                    / f"PSMARegPSMA_{case_id_y}_0000_{tp_y}.nii.gz",
                    moving_seg_path=config.data_dir
                    / "labelsTr"
                    / f"PSMARegPSMA_{case_id_x}_0000_{tp_x}.nii.gz",
                    get_affine_dvf_fn=lambda: affine_reg.get_affine_dvf(
                        case_id_x=case_id_x,
                        case_id_y=case_id_y,
                        tp_x=tp_x,
                        tp_y=tp_y,
                        fixed_ct_path=config.data_dir
                        / "imagesTr"
                        / f"PSMARegPSMA_{case_id_y}_0000_{tp_y}.nii.gz",
                        moving_ct_path=config.data_dir
                        / "imagesTr"
                        / f"PSMARegPSMA_{case_id_x}_0000_{tp_x}.nii.gz",
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
                    aug_crop_head_fixed=batch["aug_crop_head_fixed"],
                    aug_crop_feet_fixed=batch["aug_crop_feet_fixed"],
                    cfg=config,
                    device=device,
                )
                flow_prereg = poly_affine_reg.compose_flows(
                    flow_prereg, flow_poly, grid_full
                )

                X_prereg = transform(X, flow_prereg, grid_full)

            F_X_Y, X_Y, Y_4x, F_xy, _, _, _ = model(X_prereg, Y)
            if epoch % (val_interval * 50) == 0 or is_last:
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
                        name="x_ref_ct_lvl3",
                    )
                    my_data.save_volume(
                        volume=y_ref[:, 0:1, ...],
                        out_dir=config.save_dir / "initial",
                        epoch=epoch,
                        name="y_ref_ct_lvl3",
                    )
                    saved_initial = True

                if saved is False:
                    ct = X_Y[:, 0:1, :, :, :]
                    my_data.save_volume(
                        volume=ct,
                        out_dir=config.save_dir / "warped",
                        epoch=epoch,
                        name="warped_ct_lvl3",
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
            jac_det, jac = jacobian.jacobian_matrix(F_X_Y_norm)
            body_mask = batch["y_body_mask"].to(device)  # full res at lvl3
            ndv = jacobian.percent_ndv(F_X_Y_norm, mask=body_mask)
            loss_jacobian = loss_Jdet(F_X_Y_norm, grid, mask=body_mask)

            _, _, x, y, z = F_xy.shape
            F_xy[:, 0, :, :, :] = F_xy[:, 0, :, :, :] * (z - 1)
            F_xy[:, 1, :, :, :] = F_xy[:, 1, :, :, :] * (y - 1)
            F_xy[:, 2, :, :, :] = F_xy[:, 2, :, :, :] * (x - 1)
            loss_regulation = loss_smooth(F_xy)

            X_lbl_ct = transform_nearest(X_lbl_ct.float(), flow_prereg, grid_full)
            X_lbl_pet = transform_nearest(X_lbl_pet.float(), flow_prereg, grid_full)

            loss_dice_ct = utils.dice_loss_with_grad(
                X_lbl_ct, Y_lbl_ct, F_X_Y, model.grid_1, transform
            )
            loss_dice_pet = utils.dice_loss_with_grad(
                X_lbl_pet, Y_lbl_pet, F_X_Y, model.grid_1, transform
            )

            moving_pet_mask = (X_lbl_pet == 1).float()
            warped_pet_mask = utils.warp_binary_mask(
                moving_pet_mask, F_X_Y, model.grid_1, transform
            )
            warped_pet_image = X_Y[:, 1:2]
            moving_pet_image = X_prereg[:, 1:2]

            loss_mtv = utils.mtv_bias_loss(warped_pet_mask, moving_pet_mask)
            loss_tlg = utils.tlg_bias_loss(
                warped_pet_image, warped_pet_mask, moving_pet_image, moving_pet_mask
            )
            loss_jacobian_tumor = utils.masked_jac_det_loss(jac_det, moving_pet_mask)

            bone_labels_tensor = torch.tensor(
                synthetic.BONE_LABEL_VALUES, device=device, dtype=X_lbl_ct.dtype
            )
            bone_mask = torch.isin(X_lbl_ct, bone_labels_tensor).float()

            if batch["is_abdomen"]:
                # empty loss
                loss_rigidity = torch.tensor(0.0, device=device)
            else:
                loss_rigidity, _ = utils.enforce_rigidity_loss(
                    jac_det, jac, F_X_Y_norm, bone_mask
                )

            loss = (
                loss_multiNCC
                + config.w_jacobian * loss_jacobian
                + config.w_smooth * loss_regulation
                + config.w_tlg * loss_tlg
                + config.w_jacobian_tumor * loss_jacobian_tumor
                + config.w_bone_rigidity * loss_rigidity
            )
            if loss_dice_ct is not None:
                loss = loss + config.w_dice_ct_lvl3 * loss_dice_ct
            if loss_dice_pet is not None:
                loss = loss + config.w_dice_pet * loss_dice_pet

            val_losses["loss"] += loss.item()
            val_losses["ncc_ct"] += loss_ncc_ct.item()
            val_losses["ncc_pet"] += loss_ncc_pet.item()
            val_losses["smooth"] += loss_regulation.item()
            val_losses["jacobian"] += loss_jacobian.item()
            val_losses["mtv_bias"] += loss_mtv.item()
            val_losses["tlg_bias"] += loss_tlg.item()
            val_losses["jacobian_tumor"] += loss_jacobian_tumor.item()
            val_losses["rigidity"] += loss_rigidity.item()
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


def train_lvl3(
    config: TrainingConfig,
    path_model_level2: Path,
    train_generator: torch_data.DataLoader,
    valid_generator: torch_data.DataLoader,
    valid_tubingen_generator: Optional[torch_data.DataLoader] = None,
    valid_nlst_generator: Optional[torch_data.DataLoader] = None,
    valid_abdomen_generator: Optional[torch_data.DataLoader] = None,
    resume_model_path: Optional[Path] = None,
    resume_optimizer_path: Optional[Path] = None,
) -> Dict[str, Path]:
    print("Training lvl3...")

    steps_per_epoch = len(train_generator)
    total_steps = config.total_steps_lvl3
    val_step_interval = config.val_interval * steps_per_epoch
    unfreeze_step = config.unfreeze_epoch_in_lvl3 * steps_per_epoch

    run_name = utils.get_run_name()

    best_dice_ct = float("inf")
    config.model_save_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = (
        config.model_save_dir
        / f"{config.mlflow_experiment}_{run_name}_stagelvl3_best.pth"
    )
    final_model_path = (
        config.model_save_dir
        / f"{config.mlflow_experiment}_{run_name}_stagelvl3_{total_steps}.pth"
    )
    if (resume_model_path is None) != (resume_optimizer_path is None):
        raise ValueError(
            "For resuming, provide both resume_model_path and "
            "resume_optimizer_path (or neither)."
        )
    best_optimizer_path = (
        config.model_save_dir
        / f"{config.mlflow_experiment}_{run_name}_stagelvl3_best_optimizer.pth"
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
    model_lvl2 = Miccai2020_LDR_laplacian_unit_add_lvl2(
        in_channel=config.in_channel,
        n_classes=config.n_classes,
        start_channel=config.start_channel,
        is_train=True,
        imgshape=config.img_shape_2,
        range_flow=config.range_flow,
        model_lvl1=model_lvl1,
    ).to(device)

    print("Loading weight for model_lvl2...", path_model_level2)
    model_lvl2.load_state_dict(torch.load(path_model_level2))

    for param in model_lvl2.parameters():
        param.requires_grad = False

    model = Miccai2020_LDR_laplacian_unit_add_lvl3(
        in_channel=config.in_channel,
        n_classes=config.n_classes,
        start_channel=config.start_channel,
        is_train=True,
        imgshape=config.img_shape,
        range_flow=config.range_flow,
        model_lvl2=model_lvl2,
    ).to(device)

    loss_similarity_ct = multi_resolution_NCC_fast(win=config.lvl3_ncc_win, scale=3)
    loss_similarity_pet = multi_resolution_NCC_fast(win=config.lvl3_ncc_win, scale=3)
    loss_smooth = smoothloss
    loss_Jdet = jacobian.non_diff_volume_loss

    # single-resolution NCC used by the unrolled-IO objective, matching the loss
    # that test-time run_io optimizes (compute_io_loss)
    loss_ncc_io = NCC(config.lvl3_ncc_win)

    transform = SpatialTransform_unit().to(device)
    transform_nearest = SpatialTransformNearest_unit().to(device)

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True
    for param in transform_nearest.parameters():
        param.requires_grad = False

    grid = generate_grid(config.img_shape)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).to(device).float()

    grid_full = generate_grid_unit(config.img_shape)
    grid_full = (
        torch.from_numpy(np.reshape(grid_full, (1,) + grid_full.shape))
        .to(device)
        .float()
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr_lvl3)

    config.save_dir.mkdir(parents=True, exist_ok=True)

    lossall = np.zeros((4, total_steps))

    start_global_step = 0
    if resume_model_path is not None:
        print("Resuming lvl3 from...", resume_model_path)
        model.load_state_dict(torch.load(resume_model_path, map_location=device))
        opt_ckpt = torch.load(resume_optimizer_path, map_location=device)
        optimizer.load_state_dict(opt_ckpt["optimizer"])
        start_global_step = opt_ckpt["global_step"]
        best_dice_ct = opt_ckpt["best_dice_ct"]

    train_iter = utils.cycle(train_generator)

    warmup_steps = int(round(config.warmup_epochs * steps_per_epoch))
    if config.warmup_epochs >= config.unfreeze_epoch_in_lvl3:
        print(
            f"[WARN] lvl3 warmup ({config.warmup_epochs} ep) does not finish "
            f"before unfreeze ({config.unfreeze_epoch_in_lvl3} ep); the fresh "
            f"head is still warming when lvl2 is unfrozen."
        )

    if config.overfit is False:
        pbar = tqdm.tqdm(
            total=total_steps, initial=start_global_step, desc="lvl3 training"
        )

    saved_initial: bool = False

    epoch_metrics: Dict[str, float] = {}
    n_gated = 0

    # re-apply unfreeze if resuming past the threshold
    if start_global_step >= unfreeze_step:
        model.unfreeze_modellvl2()

    for global_step in range(start_global_step, total_steps):
        epoch = global_step // steps_per_epoch
        is_epoch_start = global_step % steps_per_epoch == 0
        is_epoch_end = global_step % steps_per_epoch == steps_per_epoch - 1
        is_last_step = global_step == total_steps - 1

        current_lr = utils.apply_warmup_lr(
            optimizer, config.lr_lvl3, global_step, warmup_steps
        )

        if is_epoch_start:
            epoch_metrics = {}
            n_gated = 0

        if global_step == unfreeze_step:
            model.unfreeze_modellvl2()

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

            if "_" in batch["case_id"][0]:
                case_id_x, case_id_y = batch["case_id"][0].split("_")
            else:
                case_id_x = batch["case_id"][0]
                case_id_y = batch["case_id"][0]

            tp_y: str = batch["tp_y"][0]
            tp_x: str = batch["tp_x"][0]

            flow_prereg = affine_reg.create_affine_flow(
                config=config,
                device=device,
                case_id_x=case_id_x,
                case_id_y=case_id_y,
                tp_x=tp_x,
                tp_y=tp_y,
                aug_flipped=batch["aug_flipped"],
                aug_crop_head=batch["aug_crop_head"],
                aug_crop_feet=batch["aug_crop_feet"],
                aug_crop_head_fixed=batch["aug_crop_head_fixed"],
                aug_crop_feet_fixed=batch["aug_crop_feet_fixed"],
            )

            if config.use_poly_affine is False:
                X_prereg = transform(X, flow_prereg, grid_full)
            else:
                poly_dvf = poly_affine_reg.get_polyaffine_dvf(
                    case_id_x=case_id_x,
                    case_id_y=case_id_y,
                    tp_x=tp_x,
                    tp_y=tp_y,
                    fixed_seg_path=config.data_dir
                    / "labelsTr"
                    / f"PSMARegPSMA_{case_id_y}_0000_{tp_y}.nii.gz",
                    moving_seg_path=config.data_dir
                    / "labelsTr"
                    / f"PSMARegPSMA_{case_id_x}_0000_{tp_x}.nii.gz",
                    get_affine_dvf_fn=lambda: affine_reg.get_affine_dvf(
                        case_id_x=case_id_x,
                        case_id_y=case_id_y,
                        tp_x=tp_x,
                        tp_y=tp_y,
                        fixed_ct_path=config.data_dir
                        / "imagesTr"
                        / f"PSMARegPSMA_{case_id_y}_0000_{tp_y}.nii.gz",
                        moving_ct_path=config.data_dir
                        / "imagesTr"
                        / f"PSMARegPSMA_{case_id_x}_0000_{tp_x}.nii.gz",
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
                    aug_crop_head_fixed=batch["aug_crop_head_fixed"],
                    aug_crop_feet_fixed=batch["aug_crop_feet_fixed"],
                    cfg=config,
                    device=device,
                )
                flow_prereg = poly_affine_reg.compose_flows(
                    flow_prereg, flow_poly, grid_full
                )

                X_prereg = transform(X, flow_prereg, grid_full)

        F_X_Y, X_Y, Y_4x, F_xy, _, _, _ = model(X_prereg, Y)

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
                level=3,
            )
            saved_initial = True

        if config.overfit is True and (
            (is_epoch_start and (epoch == 0 or epoch % 40 == 0)) or is_last_step
        ):
            ct = X_Y[:, 0:1, :, :, :]
            my_data.save_volume(
                volume=ct,
                out_dir=config.save_dir / "warped",
                epoch=epoch,
                name=f"warped_ct_lvl3_{run_name}_{batch['case_id'][0]}",
            )

        X_Y_ct = X_Y[:, 0:1, ...]
        X_Y_pet = X_Y[:, 1:2, ...]
        Y_4x_ct = Y_4x[:, 0:1, ...]
        Y_4x_pet = Y_4x[:, 1:2, ...]

        F_X_Y_norm = transform_unit_flow_to_flow_cuda(
            F_X_Y.permute(0, 2, 3, 4, 1).clone()
        )
        jac_det, jac = jacobian.jacobian_matrix(F_X_Y_norm)
        body_mask = batch["y_body_mask"].to(device)  # full res at lvl3
        ndv = jacobian.percent_ndv(F_X_Y_norm, mask=body_mask)
        loss_jacobian = loss_Jdet(F_X_Y_norm, grid, mask=body_mask)

        _, _, x, y, z = F_xy.shape
        F_xy[:, 0, :, :, :] = F_xy[:, 0, :, :, :] * (z - 1)
        F_xy[:, 1, :, :, :] = F_xy[:, 1, :, :, :] * (y - 1)
        F_xy[:, 2, :, :, :] = F_xy[:, 2, :, :, :] * (x - 1)
        loss_regulation = loss_smooth(F_xy)

        # synthetic labels are already in the (deformed) moving frame; the
        # real branch needs the affine applied first
        if not is_synthetic:
            X_lbl_ct = transform_nearest(X_lbl_ct.float(), flow_prereg, grid_full)
            X_lbl_pet = transform_nearest(X_lbl_pet.float(), flow_prereg, grid_full)

        if is_synthetic:
            use_dice_pet = True
            use_ncc_pet = True
        else:
            # pet_iou = utils.affine_pet_iou(
            #     batch["x_label_pet"].to(device),
            #     Y_lbl_pet,
            #     flow_prereg,
            #     grid_full,
            #     transform_nearest,
            # )
            # use_dice_pet = pet_iou >= config.dice_pet_iou_threshold
            # use_ncc_pet = pet_iou >= config.dice_pet_iou_threshold
            use_dice_pet = False
            use_ncc_pet = False
        if not use_dice_pet:
            n_gated += 1

        loss_ncc_ct = loss_similarity_ct(X_Y_ct, Y_4x_ct)
        with torch.no_grad():
            loss_ncc_pet = loss_similarity_pet(X_Y_pet, Y_4x_pet)
        if use_ncc_pet:
            loss_multiNCC = config.w_ct * loss_ncc_ct + config.w_pet * loss_ncc_pet
        else:
            loss_multiNCC = config.w_ct * loss_ncc_ct

        loss_dice_ct = utils.dice_loss_with_grad(
            X_lbl_ct, Y_lbl_ct, F_X_Y, model.grid_1, transform
        )
        with torch.no_grad():
            loss_dice_pet = utils.dice_loss_with_grad(
                X_lbl_pet, Y_lbl_pet, F_X_Y, model.grid_1, transform
            )

        moving_pet_mask = (X_lbl_pet == 1).float()
        loss_jacobian_tumor = utils.masked_jac_det_loss(jac_det, moving_pet_mask)

        bone_labels_tensor = torch.tensor(
            synthetic.BONE_LABEL_VALUES, device=device, dtype=X_lbl_ct.dtype
        )
        bone_mask = torch.isin(X_lbl_ct, bone_labels_tensor).float()

        if batch["is_abdomen"]:
            loss_rigidity = torch.tensor(0.0, device=device)
        else:
            loss_rigidity, (loss_rig_det, loss_rig_ortho, loss_rig_affine) = (
                utils.enforce_rigidity_loss(jac_det, jac, F_X_Y_norm, bone_mask)
            )

        loss = (
            loss_multiNCC
            + config.w_jacobian * loss_jacobian
            + config.w_smooth * loss_regulation
            + config.w_jacobian_tumor * loss_jacobian_tumor
            + config.w_bone_rigidity * loss_rigidity
        )
        if loss_dice_ct is not None:
            loss = loss + config.w_dice_ct_lvl3 * loss_dice_ct
        if loss_dice_pet is not None and use_dice_pet:
            loss = loss + config.w_dice_pet * loss_dice_pet

        if is_synthetic:
            loss_dvf = ((F_X_Y - gt_unit) ** 2).mean()
            loss = loss + config.w_dvf * loss_dvf
            loss_mtv = torch.zeros((), device=device)
            loss_tlg = torch.zeros((), device=device)
        else:
            warped_pet_mask = utils.warp_binary_mask(
                moving_pet_mask, F_X_Y, model.grid_1, transform
            )
            warped_pet_image = X_Y[:, 1:2]
            moving_pet_image = X_prereg[:, 1:2]
            loss_mtv = utils.mtv_bias_loss(warped_pet_mask, moving_pet_mask)
            loss_tlg = utils.tlg_bias_loss(
                warped_pet_image,
                warped_pet_mask,
                moving_pet_image,
                moving_pet_mask,
            )
            loss = loss + config.w_tlg * loss_tlg
            loss_dvf = torch.zeros((), device=device)

        # meta-learned / unrolled IO: run a few differentiable IO steps starting
        # from the net's output and add the loss on the *refined* field. This
        # trains F_X_Y to be a good seed for the deployed run_io optimizer rather
        # than a good final answer on its own. X_lbl_ct is already in the affine
        # (moving) frame here for both branches; X_prereg / Y are full-res.
        if config.use_unrolled_io and epoch >= config.unroll_start_epoch:
            y_ct_io = Y[:, 0:1]

            # Inner loop: descend the same objective run_io deploys (full moving
            # image so the PET channel is available), so the net is seeded for the
            # exact trajectory we run at test time. bone_labels_tensor is reused
            # from the feed-forward rigidity term above.
            def io_inner_loss_fn(disp_unit: torch.Tensor) -> torch.Tensor:
                return instance_opt.unrolled_io_loss(
                    disp_unit,
                    y_ct_io,
                    X_prereg,
                    X_lbl_ct,
                    Y_lbl_ct,
                    transform,
                    grid_full,
                    config,
                    loss_ncc_io,
                    ncc_weight=config.w_ct,
                    x_lbl_pet=X_lbl_pet,
                    bone_values=bone_labels_tensor,
                    include_pet=config.unroll_include_pet,
                    include_rigidity=config.unroll_include_rigidity,
                )

            refined_disp = instance_opt.unrolled_refine(
                F_X_Y,
                io_inner_loss_fn,
                config,
                device,
                n_steps=config.unroll_K,
                inner_lr=config.unroll_inner_lr,
                n_integration=config.unroll_n_integration,
                mode=config.unroll_mode,
            )

            # Outer meta-loss: grade the net on the same full objective the inner
            # loop descended, which is also what the challenge scores (dice,
            # folding, TLG, MTV). No reason to grade on a subset.
            loss_unrolled = io_inner_loss_fn(refined_disp)
            loss = loss + config.w_unrolled * loss_unrolled
        else:
            loss_unrolled = torch.zeros((), device=device)

        loss_scaled = loss / config.accumulation_steps
        is_step = (global_step + 1) % config.accumulation_steps == 0 or is_last_step

        utils.optimizer_step_with_guard(
            loss, loss_scaled, optimizer, model, is_step, global_step, level=3
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
                mtv=f"{loss_mtv.item():.4f}",
                tlg=f"{loss_tlg.item():.4f}",
                Jdet=f"{loss_jacobian.item():.6f}",
                smo=f"{loss_regulation.item():.4f}",
                dvf=f"{loss_dvf.item():.8f}",
            )
        train_metrics = {
            "train_lvl3/loss": loss.item(),
            "train_lvl3/ncc_ct": loss_ncc_ct.item(),
            "train_lvl3/ncc_pet": loss_ncc_pet.item(),
            "train_lvl3/smooth": loss_regulation.item(),
            "train_lvl3/mtv_bias": loss_mtv.item(),
            "train_lvl3/tlg_bias": loss_tlg.item(),
            "train_lvl3/jacobian_tumor": loss_jacobian_tumor.item(),
            "train_lvl3/jacobian": loss_jacobian.item(),
            "train_lvl3/rigidity": loss_rigidity.item(),
            "train_lvl3/ndv": ndv,
            "train_lvl3/dvf": loss_dvf.item(),
            "train_lvl3/lr": current_lr,
        }
        if loss_dice_ct is not None:
            train_metrics["train_lvl3/dice_ct"] = loss_dice_ct.item()
        if loss_dice_pet is not None:
            train_metrics["train_lvl3/dice_pet"] = loss_dice_pet.item()
        # weighted contributions (weight * term = actual share of the total loss)
        train_metrics["train_lvl3/w_ncc_ct"] = config.w_ct * loss_ncc_ct.item()
        train_metrics["train_lvl3/w_ncc_pet"] = (
            config.w_pet * loss_ncc_pet.item() if use_ncc_pet else 0.0
        )
        train_metrics["train_lvl3/w_jacob"] = config.w_jacobian * loss_jacobian.item()
        train_metrics["train_lvl3/w_smooth"] = config.w_smooth * loss_regulation.item()
        train_metrics["train_lvl3/w_jacob_tumor"] = (
            config.w_jacobian_tumor * loss_jacobian_tumor.item()
        )
        train_metrics["train_lvl3/w_rigidity"] = (
            config.w_bone_rigidity * loss_rigidity.item()
        )
        train_metrics["train_lvl3/w_tlg"] = config.w_tlg * loss_tlg.item()
        train_metrics["train_lvl3/w_dvf"] = config.w_dvf * loss_dvf.item()
        if loss_dice_ct is not None:
            train_metrics["train_lvl3/w_dice_ct"] = (
                config.w_dice_ct_lvl3 * loss_dice_ct.item()
            )
        if loss_dice_pet is not None and use_dice_pet:
            train_metrics["train_lvl3/w_dice_pet"] = (
                config.w_dice_pet * loss_dice_pet.item()
            )
        utils.log_metrics(train_metrics, step=global_step)

        for key, value in train_metrics.items():
            epoch_metrics[key] = epoch_metrics.get(key, 0.0) + value

        if is_epoch_end:
            utils.log_metrics(
                {
                    f"{key}_epoch": value / steps_per_epoch
                    for key, value in epoch_metrics.items()
                },
                step=global_step,
            )
            if config.overfit:
                print(
                    f"ep: {epoch}\t"
                    f"ncc={epoch_metrics['train_lvl3/ncc_ct']:.4f}\t"
                    f"dice={epoch_metrics['train_lvl3/dice_ct']:.4f}\t"
                    f"jacob={epoch_metrics['train_lvl3/jacob']:.6f}\t"
                    f"unrolled={epoch_metrics['train_lvl3/unrolled_io']:.4f}\t"
                )
        if config.overfit is False and (
            global_step % val_step_interval == 0 or is_last_step
        ):
            val_losses = evaluate_lvl3(
                model=model,
                val_generator=valid_generator,
                config=config,
                device=device,
                loss_similarity_ct=loss_similarity_ct,
                loss_similarity_pet=loss_similarity_pet,
                loss_smooth=loss_smooth,
                loss_Jdet=loss_Jdet,
                transform=transform,
                grid=grid,
                epoch=epoch,
                val_interval=config.val_interval,
                saved_initial=saved_initial,
                is_last=is_last_step,
            )
            saved_initial = True
            utils.log_metrics(
                {
                    f"valid_lvl3/val_{key}": value
                    for key, value in val_losses.items()
                    if not (isinstance(value, float) and np.isnan(value))
                },
                step=global_step,
            )

            if valid_tubingen_generator is not None:
                val_losses_tubingen = evaluate_lvl3(
                    model=model,
                    val_generator=valid_tubingen_generator,
                    config=config,
                    device=device,
                    loss_similarity_ct=loss_similarity_ct,
                    loss_similarity_pet=loss_similarity_pet,
                    loss_smooth=loss_smooth,
                    loss_Jdet=loss_Jdet,
                    transform=transform,
                    grid=grid,
                    epoch=epoch,
                    val_interval=config.val_interval,
                    saved_initial=saved_initial,
                    is_last=is_last_step,
                )
                utils.log_metrics(
                    {
                        f"valid_lvl3/val_{key}_tubingen": value
                        for key, value in val_losses_tubingen.items()
                        if not (isinstance(value, float) and np.isnan(value))
                    },
                    step=global_step,
                )

            if valid_nlst_generator is not None:
                val_losses_nlst = evaluate_lvl3(
                    model=model,
                    val_generator=valid_nlst_generator,
                    config=config,
                    device=device,
                    loss_similarity_ct=loss_similarity_ct,
                    loss_similarity_pet=loss_similarity_pet,
                    loss_smooth=loss_smooth,
                    loss_Jdet=loss_Jdet,
                    transform=transform,
                    grid=grid,
                    epoch=epoch,
                    val_interval=config.val_interval,
                    saved_initial=saved_initial,
                    is_last=is_last_step,
                )
                utils.log_metrics(
                    {
                        f"valid_lvl3/val_{key}_nlst": value
                        for key, value in val_losses_nlst.items()
                        if not (isinstance(value, float) and np.isnan(value))
                    },
                    step=global_step,
                )

            if valid_abdomen_generator is not None:
                val_losses_abdomen = evaluate_lvl3(
                    model=model,
                    val_generator=valid_abdomen_generator,
                    config=config,
                    device=device,
                    loss_similarity_ct=loss_similarity_ct,
                    loss_similarity_pet=loss_similarity_pet,
                    loss_smooth=loss_smooth,
                    loss_Jdet=loss_Jdet,
                    transform=transform,
                    grid=grid,
                    epoch=epoch,
                    val_interval=config.val_interval,
                    saved_initial=saved_initial,
                    is_last=is_last_step,
                )
                utils.log_metrics(
                    {
                        f"valid_lvl3/val_{key}_abdomen": value
                        for key, value in val_losses_abdomen.items()
                        if not (isinstance(value, float) and np.isnan(value))
                    },
                    step=global_step,
                )

            tqdm.tqdm.write(
                f"step {global_step} (ep {epoch}) -> val loss {val_losses['loss']:.4f} "
                f"- ncc_ct {val_losses['ncc_ct']:.4f} "
                f"- ncc_pet {val_losses['ncc_pet']:.4f} "
                f"- dice_ct {val_losses['dice_ct']:.4f} "
                f"- dice_pet {val_losses['dice_pet']:.4f} "
                f"- mtv {val_losses['mtv_bias']:.4f} "
                f"- tlg {val_losses['tlg_bias']:.4f}"
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

        if config.overfit is False:
            pbar.update(1)

    if config.overfit is False:
        pbar.close()

    torch.save(model.state_dict(), final_model_path)
    np.save(
        config.save_dir
        / f"loss_{config.mlflow_experiment}_stagelvl3_{total_steps}.npy",
        lossall,
    )

    result = {
        "final": final_model_path,
        "best": best_model_path,
        "best_optimizer": best_optimizer_path,
    }
    return result
