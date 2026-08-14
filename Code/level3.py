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
    build_similarity_loss,
    smoothloss,
)
from torch.utils import data as torch_data


def evaluate_lvl3(
    model: Miccai2020_LDR_laplacian_unit_add_lvl3,
    val_generator: torch_data.DataLoader,
    config: TrainingConfig,
    device: torch.device,
    loss_similarity_ct: torch.nn.Module,
    loss_similarity_pet: torch.nn.Module,
    loss_smooth: Callable,
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
        "non_diff_loss": 0.0,
        "mtv_bias": 0.0,
        "mtv_mean": 0.0,
        "tlg_bias": 0.0,
        "jacobian_tumor": 0.0,
        "rigidity": 0.0,
        "rig_det": 0.0,
        "rig_ortho": 0.0,
        "rig_affine": 0.0,
        "rig_worst": 0.0,
        "ndv": 0.0,
    }
    for enabled, prefix in (
        (config.use_seg_pet_head, "seg_pet"),
        (config.use_seg_bone_head, "seg_bone"),
    ):
        if enabled:
            val_losses.update(
                {
                    prefix: 0.0,
                    f"{prefix}_dice_loss": 0.0,
                    f"{prefix}_bce": 0.0,
                    f"{prefix}_dice_fixed": 0.0,
                    f"{prefix}_dice_moving": 0.0,
                }
            )
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

            Y_4x_ct = Y_4x[:, 0:1, ...]
            Y_4x_pet = Y_4x[:, 1:2, ...]

            # scored terms (NCC / dice / tumour / rigidity) are measured on the
            # TOTAL warp (prereg + network) of the ORIGINAL moving image and
            # labels — single interpolation, det(A) included — matching the
            # final eval. The residual-only version resampled twice and was
            # blind to any distortion the pre-registration introduced.
            flow_total = poly_affine_reg.compose_flows(
                flow_prereg, F_X_Y.permute(0, 2, 3, 4, 1), grid_full
            )
            flow_total_norm = transform_unit_flow_to_flow_cuda(flow_total.clone())
            X_Y_total = transform(X, flow_total, grid_full)
            X_Y_ct = X_Y_total[:, 0:1]
            X_Y_pet = X_Y_total[:, 1:2]

            loss_ncc_ct = loss_similarity_ct(X_Y_ct, Y_4x_ct)
            loss_ncc_pet = loss_similarity_pet(X_Y_pet, Y_4x_pet)
            loss_multiNCC = config.w_ct * loss_ncc_ct + config.w_pet * loss_ncc_pet

            F_X_Y_norm = transform_unit_flow_to_flow_cuda(
                F_X_Y.permute(0, 2, 3, 4, 1).clone()
            )
            jac_det, jac = jacobian.jacobian_matrix(F_X_Y_norm)
            body_mask = batch["y_body_mask"].to(device)  # full res at lvl3
            ndv = jacobian.percent_ndv(F_X_Y_norm, mask=body_mask)
            loss_non_diff = jacobian.non_diff_volume_loss(
                F_X_Y_norm, grid, mask=body_mask
            )

            _, _, x, y, z = F_xy.shape
            F_xy[:, 0, :, :, :] = F_xy[:, 0, :, :, :] * (z - 1)
            F_xy[:, 1, :, :, :] = F_xy[:, 1, :, :, :] * (y - 1)
            F_xy[:, 2, :, :, :] = F_xy[:, 2, :, :, :] * (x - 1)
            loss_regulation = loss_smooth(F_xy)

            X_lbl_ct_orig = X_lbl_ct
            X_lbl_pet_orig = X_lbl_pet
            X_lbl_ct = transform_nearest(X_lbl_ct.float(), flow_prereg, grid_full)
            X_lbl_pet = transform_nearest(X_lbl_pet.float(), flow_prereg, grid_full)

            flow_total_ch = flow_total.permute(0, 4, 1, 2, 3)
            loss_dice_ct = utils.dice_loss_with_grad_bbox(
                X_lbl_ct_orig.float(),
                Y_lbl_ct,
                flow_total_ch,
                model.grid_1,
                transform,
                use_checkpoint=True,
            )
            loss_dice_pet = utils.dice_loss_with_grad_bbox(
                X_lbl_pet_orig.float(),
                Y_lbl_pet,
                flow_total_ch,
                model.grid_1,
                transform,
                use_checkpoint=True,
            )

            moving_pet_mask = (X_lbl_pet == 1).float()
            moving_pet_mask_orig = (X_lbl_pet_orig == 1).float()
            warped_pet_mask = transform(moving_pet_mask_orig, flow_total, grid_full)

            loss_mtv = utils.mtv_bias_loss(warped_pet_mask, moving_pet_mask_orig)
            loss_tlg = utils.tlg_bias_loss(
                X_Y_pet,
                warped_pet_mask,
                X[:, 1:2],
                moving_pet_mask_orig,
            )
            jac_det_total, _ = jacobian.jacobian_matrix(flow_total_norm)
            # det(J_total) lives on the fixed grid, so mask it with the
            # fixed-frame lesion (the composed-warped mask)
            warped_pet_mask_hard = transform_nearest(
                moving_pet_mask_orig, flow_total, grid_full
            )
            loss_jacobian_tumor = utils.masked_jac_det_loss(
                jac_det_total, warped_pet_mask_hard
            )
            loss_mtv_mean = utils.mtv_mean_bias_loss(
                jac_det_total, warped_pet_mask.detach()
            )

            bone_labels_tensor = torch.tensor(
                synthetic.BONE_LABEL_VALUES, device=device, dtype=X_lbl_ct.dtype
            )
            bone_mask = torch.isin(X_lbl_ct, bone_labels_tensor).float()

            zero_rig = torch.tensor(0.0, device=device)
            loss_rig_det = loss_rig_ortho = loss_rig_affine = zero_rig
            loss_rig_worst = zero_rig
            if batch["is_abdomen"]:
                # empty loss
                loss_rigidity = zero_rig
            elif config.use_per_label_rigidity:
                # rigidity on the TOTAL field over the FIXED-frame bones: no
                # label resampling, and the net is pushed to undo non-rigid
                # bone motion the affine introduced (scale/shear)
                loss_rigidity, rig_info = utils.per_label_rigid_loss(
                    flow_total_norm,
                    Y_lbl_ct.float(),
                    bone_labels_tensor,
                    min_voxels=config.rigidity_min_voxels,
                )
                loss_rig_worst = rig_info["worst"]
            else:
                loss_rigidity, (loss_rig_det, loss_rig_ortho, loss_rig_affine) = (
                    utils.enforce_rigidity_loss(
                        jac_det,
                        jac,
                        F_X_Y_norm,
                        bone_mask,
                        w_det=config.w_rig_det,
                        w_ortho=config.w_rig_ortho,
                        w_affine=config.w_rig_affine,
                    )
                )

            loss = (
                loss_multiNCC
                + config.w_non_diff * loss_non_diff
                + config.w_smooth * loss_regulation
                + config.w_tlg * loss_tlg
                + config.w_mtv * loss_mtv**2
                + config.w_mtv_mean * loss_mtv_mean
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
            val_losses["non_diff_loss"] += loss_non_diff.item()
            val_losses["mtv_bias"] += loss_mtv.item()
            val_losses["mtv_mean"] += loss_mtv_mean.item()
            val_losses["tlg_bias"] += loss_tlg.item()
            val_losses["jacobian_tumor"] += loss_jacobian_tumor.item()
            val_losses["rigidity"] += loss_rigidity.item()
            val_losses["rig_det"] += loss_rig_det.item()
            val_losses["rig_ortho"] += loss_rig_ortho.item()
            val_losses["rig_affine"] += loss_rig_affine.item()
            val_losses["rig_worst"] += loss_rig_worst.item()
            val_losses["ndv"] += ndv

            if config.use_seg_pet_head:
                loss_seg, seg_metrics = utils.seg_head_terms(
                    model.seg_pet_logits,
                    model.lvl2_disp_up_inv,
                    (Y_lbl_pet == 1).float(),
                    moving_pet_mask,
                    transform,
                    grid_full,
                    body_mask,
                    prefix="seg_pet",
                )
                val_losses["seg_pet"] += loss_seg.item()
                for key, value in seg_metrics.items():
                    val_losses[key] += value

            if config.use_seg_bone_head:
                loss_seg_bone, seg_bone_metrics = utils.seg_head_terms(
                    model.seg_bone_logits,
                    model.lvl2_disp_up_inv,
                    torch.isin(Y_lbl_ct, bone_labels_tensor).float(),
                    bone_mask,
                    transform,
                    grid_full,
                    body_mask,
                    prefix="seg_bone",
                )
                val_losses["seg_bone"] += loss_seg_bone.item()
                for key, value in seg_bone_metrics.items():
                    val_losses[key] += value

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
    best_tumour = float("inf")
    best_combined = float("inf")
    config.model_save_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = (
        config.model_save_dir
        / f"{config.mlflow_experiment}_{run_name}_stagelvl3_best.pth"
    )
    # tumour val metrics turn upward long before dice_ct plateaus, so selecting
    # on dice alone ships a checkpoint deep into the tumour-overfitting region.
    # Keep a checkpoint per objective and let the test set decide.
    best_tumour_model_path = (
        config.model_save_dir
        / f"{config.mlflow_experiment}_{run_name}_stagelvl3_best_tumour.pth"
    )
    best_combined_model_path = (
        config.model_save_dir
        / f"{config.mlflow_experiment}_{run_name}_stagelvl3_best_combined.pth"
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
    # one optimizer checkpoint per selection criterion, written at the same
    # moment as its model, so any of the three can be resumed from
    best_tumour_optimizer_path = (
        config.model_save_dir
        / f"{config.mlflow_experiment}_{run_name}_stagelvl3_best_tumour_optimizer.pth"
    )
    best_combined_optimizer_path = (
        config.model_save_dir
        / f"{config.mlflow_experiment}_{run_name}_stagelvl3_best_combined_optimizer.pth"
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if config.use_lvl1:
        model_lvl1 = Miccai2020_LDR_laplacian_unit_add_lvl1(
            in_channel=config.in_channel,
            n_classes=config.n_classes,
            start_channel=config.start_channel,
            is_train=True,
            imgshape=config.img_shape_4,
            range_flow=config.range_flow,
            cost_volume_mode=config.cost_volume_mode,
            cost_volume_radius=config.cost_volume_radius,
            cost_volume_dilation=config.cost_volume_dilation,
            cost_volume_feat_channels=config.cost_volume_feat_channels,
            cost_volume_out_channels=config.cost_volume_out_channels,
            n_resblocks=config.n_resblocks,
            resblock_expansion=config.resblock_expansion,
        ).to(device)
    else:
        model_lvl1 = None

    model_lvl2 = Miccai2020_LDR_laplacian_unit_add_lvl2(
        in_channel=config.in_channel,
        n_classes=config.n_classes,
        start_channel=config.start_channel,
        is_train=True,
        imgshape=config.img_shape_2,
        range_flow=config.range_flow,
        model_lvl1=model_lvl1,
        n_resblocks=config.n_resblocks,
        resblock_expansion=config.resblock_expansion,
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
        n_resblocks=config.n_resblocks,
        resblock_expansion=config.resblock_expansion,
        use_seg_pet_head=config.use_seg_pet_head,
        seg_pet_head_channels=config.seg_pet_head_channels,
        use_bone_head=config.use_seg_bone_head,
        bone_head_channels=config.seg_bone_head_channels,
    ).to(device)

    loss_similarity_ct = build_similarity_loss(config, level=3)
    loss_similarity_pet = build_similarity_loss(config, level=3)
    loss_smooth = smoothloss

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
        # .get: optimizer checkpoints written before the extra selection
        # criteria existed only carry best_dice_ct
        best_tumour = opt_ckpt.get("best_tumour", float("inf"))
        best_combined = opt_ckpt.get("best_combined", float("inf"))

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

    grad_conflict_tracker = utils.GradConflictTracker(
        window=config.grad_conflict_window
    )
    # A scheduled measurement is *armed* at the interval, then *taken* on the
    # next batch where every objective term is live. Measuring on the scheduled
    # step directly would land on lesion-free batches ~half the time (tlg/jactum
    # have no gradient there), so those keys would be absent and every wandb
    # expression joining them against an always-present key would misalign.
    grad_conflict_pending = False

    # re-apply unfreeze if resuming past the threshold
    if start_global_step >= unfreeze_step:
        model.unfreeze_modellvl2()

    _flag = utils.stop_flag_path(config.save_dir, 3)
    _flag.parent.mkdir(parents=True, exist_ok=True)
    tqdm.tqdm.write(f"[lvl3] stop early with:  touch {_flag}")
    utils.log_text(f"touch {_flag}", "stop_lvl3_cmd.txt")

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

        Y_4x_ct = Y_4x[:, 0:1, ...]
        Y_4x_pet = Y_4x[:, 1:2, ...]

        F_X_Y_norm = transform_unit_flow_to_flow_cuda(
            F_X_Y.permute(0, 2, 3, 4, 1).clone()
        )
        jac_det, jac = jacobian.jacobian_matrix(F_X_Y_norm)
        body_mask = batch["y_body_mask"].to(device)  # full res at lvl3
        ndv = jacobian.percent_ndv(F_X_Y_norm, mask=body_mask)
        loss_non_diff = jacobian.non_diff_volume_loss(F_X_Y_norm, grid, mask=body_mask)

        _, _, x, y, z = F_xy.shape
        F_xy[:, 0, :, :, :] = F_xy[:, 0, :, :, :] * (z - 1)
        F_xy[:, 1, :, :, :] = F_xy[:, 1, :, :, :] * (y - 1)
        F_xy[:, 2, :, :, :] = F_xy[:, 2, :, :, :] * (x - 1)
        loss_regulation = loss_smooth(F_xy)

        # synthetic labels are already in the (deformed) moving frame; the
        # real branch needs the affine applied first. The pre-affine labels are
        # kept: the scored terms (NCC / dice / tumour / rigidity) are measured
        # on the TOTAL warp of the ORIGINAL moving image and labels — single
        # interpolation, det(A) included — matching the final eval.
        X_lbl_ct_orig = X_lbl_ct
        X_lbl_pet_orig = X_lbl_pet
        if not is_synthetic:
            X_lbl_ct = transform_nearest(X_lbl_ct.float(), flow_prereg, grid_full)
            X_lbl_pet = transform_nearest(X_lbl_pet.float(), flow_prereg, grid_full)
            flow_total = poly_affine_reg.compose_flows(
                flow_prereg, F_X_Y.permute(0, 2, 3, 4, 1), grid_full
            )
            flow_total_norm = transform_unit_flow_to_flow_cuda(flow_total.clone())
        else:
            # no pre-registration on this branch: the network field IS the total
            flow_total = F_X_Y.permute(0, 2, 3, 4, 1)
            flow_total_norm = F_X_Y_norm

        # single-interpolation warp of the original moving for NCC / TLG
        X_Y_total = transform(X, flow_total, grid_full)
        X_Y_ct = X_Y_total[:, 0:1]
        X_Y_pet = X_Y_total[:, 1:2]

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

        # losses run in fp32: the bf16 autocast region is confined to the conv
        # trunk inside the model forwards, and F_X_Y / F_xy / X_Y are upcast to
        # fp32 before they reach here. This guard makes that invariant explicit
        # (a no-op today, protective if a future edit wraps the loop body).
        assert not torch.is_autocast_enabled(), "losses must run outside autocast"
        loss_ncc_ct = loss_similarity_ct(X_Y_ct, Y_4x_ct)
        with torch.no_grad():
            loss_ncc_pet = loss_similarity_pet(X_Y_pet, Y_4x_pet)
        if use_ncc_pet:
            loss_multiNCC = config.w_ct * loss_ncc_ct + config.w_pet * loss_ncc_pet
        else:
            loss_multiNCC = config.w_ct * loss_ncc_ct

        x = 0

        flow_total_ch = flow_total.permute(0, 4, 1, 2, 3)
        loss_dice_ct = utils.dice_loss_with_grad_bbox(
            X_lbl_ct_orig.float(),
            Y_lbl_ct,
            flow_total_ch,
            model.grid_1,
            transform,
            use_checkpoint=True,
        )
        with torch.no_grad():
            loss_dice_pet = utils.dice_loss_with_grad_bbox(
                X_lbl_pet_orig.float(),
                Y_lbl_pet,
                flow_total_ch,
                model.grid_1,
                transform,
                use_checkpoint=True,
            )

        moving_pet_mask = (X_lbl_pet == 1).float()
        moving_pet_mask_orig = (X_lbl_pet_orig == 1).float()
        # tumour-jac on det(J_total): the residual-only det(J) was blind to
        # volume changes the pre-registration introduced (det(A) != 1)
        if is_synthetic:
            jac_det_total = jac_det
        else:
            jac_det_total, _ = jacobian.jacobian_matrix(flow_total_norm)
        # det(J_total) lives on the fixed grid, so mask it with the
        # fixed-frame lesion (the composed-warped mask)
        with torch.no_grad():
            warped_pet_mask_hard = transform_nearest(
                moving_pet_mask_orig, flow_total, grid_full
            )
        loss_jacobian_tumor = utils.masked_jac_det_loss(
            jac_det_total, warped_pet_mask_hard
        )

        bone_labels_tensor = torch.tensor(
            synthetic.BONE_LABEL_VALUES, device=device, dtype=X_lbl_ct.dtype
        )
        bone_mask = torch.isin(X_lbl_ct, bone_labels_tensor).float()

        zero_rig = torch.tensor(0.0, device=device)
        loss_rig_det = loss_rig_ortho = loss_rig_affine = zero_rig
        loss_rig_worst = zero_rig
        rig_worst_label = zero_rig
        n_rig_labels = zero_rig
        if batch["is_abdomen"]:
            loss_rigidity = zero_rig
        elif config.use_per_label_rigidity:
            # rigidity on the TOTAL field over the FIXED-frame bones: no label
            # resampling, and the net is pushed to undo non-rigid bone motion
            # the affine introduced (scale/shear)
            loss_rigidity, rig_info = utils.per_label_rigid_loss(
                flow_total_norm,
                Y_lbl_ct.float(),
                bone_labels_tensor,
                min_voxels=config.rigidity_min_voxels,
            )
            n_rig_labels = rig_info["n_labels"]
            loss_rig_worst = rig_info["worst"]
            rig_worst_label = rig_info["worst_label"]
        else:
            raise NotImplementedError(
                "per-label rigidity is required for lvl3; the old single-mask rigidity is not supported"
            )
            loss_rigidity, (loss_rig_det, loss_rig_ortho, loss_rig_affine) = (
                utils.enforce_rigidity_loss(
                    jac_det,
                    jac,
                    F_X_Y_norm,
                    bone_mask,
                    w_det=config.w_rig_det,
                    w_ortho=config.w_rig_ortho,
                    w_affine=config.w_rig_affine,
                )
            )

        # Auxiliary segmentation heads. Both are predicted in the fixed frame
        # (that is what the lvl3 trunk sees: its input is warped_x, not x), and
        # channel 1 of each is warped back into the moving frame inside
        # seg_head_terms so that neither target has to be resampled.
        if config.use_seg_pet_head:
            loss_seg_pet, seg_metrics = utils.seg_head_terms(
                model.seg_pet_logits,
                model.lvl2_disp_up_inv,
                (Y_lbl_pet == 1).float(),
                moving_pet_mask,
                transform,
                grid_full,
                body_mask,
                prefix="seg_pet",
            )
            w_seg_pet = config.w_seg_pet * (
                min(1.0, (epoch + 1) / config.seg_pet_warmup_epochs)
                if config.seg_pet_warmup_epochs > 0
                else 1.0
            )
        else:
            loss_seg_pet = torch.zeros((), device=device)
            seg_metrics = {}
            w_seg_pet = 0.0

        if config.use_seg_bone_head:
            # bone_mask is already the moving (prereg) frame bone mask built for
            # the rigidity term; the fixed one is its counterpart on Y_lbl_ct
            loss_seg_bone, seg_bone_metrics = utils.seg_head_terms(
                model.seg_bone_logits,
                model.lvl2_disp_up_inv,
                torch.isin(Y_lbl_ct, bone_labels_tensor).float(),
                bone_mask,
                transform,
                grid_full,
                body_mask,
                prefix="seg_bone",
            )
            w_seg_bone = config.w_seg_bone * (
                min(1.0, (epoch + 1) / config.seg_bone_warmup_epochs)
                if config.seg_bone_warmup_epochs > 0
                else 1.0
            )
        else:
            loss_seg_bone = torch.zeros((), device=device)
            seg_bone_metrics = {}
            w_seg_bone = 0.0

        loss = (
            loss_multiNCC
            + config.w_non_diff * loss_non_diff
            + config.w_smooth * loss_regulation
            + config.w_jacobian_tumor * loss_jacobian_tumor
            + config.w_bone_rigidity * loss_rigidity
            + w_seg_pet * loss_seg_pet
            + w_seg_bone * loss_seg_bone
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
            loss_mtv_mean = torch.zeros((), device=device)
        else:
            # MTV/TLG on the total warp against the ORIGINAL moving PET:
            # single interpolation, det(A) included, matches the final eval
            warped_pet_mask = transform(moving_pet_mask_orig, flow_total, grid_full)
            loss_mtv = utils.mtv_bias_loss(warped_pet_mask, moving_pet_mask_orig)
            loss_tlg = utils.tlg_bias_loss(
                X_Y_pet,
                warped_pet_mask,
                X[:, 1:2],
                moving_pet_mask_orig,
            )
            # same recipe as deploy-time IO: squared soft-count MTV is exactly
            # the scored metric but only has gradient at the lesion boundary;
            # mean-det over the (detached) warped lesion pins the same net
            # volume with dense gradients in the interior
            loss_mtv_mean = utils.mtv_mean_bias_loss(
                jac_det_total, warped_pet_mask.detach()
            )
            loss = (
                loss
                + config.w_tlg * loss_tlg
                + config.w_mtv * loss_mtv**2
                + config.w_mtv_mean * loss_mtv_mean
            )
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
            # from the feed-forward rigidity term above. autocast(enabled=False):
            # the refined field + IO losses stay in fp32 (the inner velocity leaf
            # would otherwise inherit an autocast dtype).
            with torch.autocast(device_type="cuda", enabled=False):

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

        # Gradient-conflict diagnostic. Must run before the backward in
        # optimizer_step_with_guard (it needs the graph alive) but it uses
        # autograd.grad, so it neither writes .grad nor disturbs accumulation.
        # The unrolled-IO meta-loss is deliberately excluded: it is a weighted
        # mixture of the other terms and would appear as a copy of the objective.
        if config.log_grad_conflict and (
            global_step % (val_step_interval * config.grad_conflict_every_n_val) == 0
        ):
            grad_conflict_pending = True

        # Take the armed measurement on the first batch that exercises every
        # term, so all keys share one x-axis. moving_pet_mask exists only on the
        # real branch; synthetic is off, but guard anyway.
        pet_live = (not is_synthetic) and bool(moving_pet_mask.any())
        if grad_conflict_pending and not batch["is_abdomen"] and pet_live:
            grad_conflict_pending = False
            # One flat dict, no accuracy/tumour grouping. The original split was
            # the hypothesis under test and run wise-swan-39232655 falsified it:
            # of the -0.25 group cosine, bone rigidity supplied -0.20 while tlg
            # and jactum supplied ~0.00, and rigidity's own cosines (+0.53 with
            # smooth, -0.53 with ncc, 0.00 with tlg) mark it a regulariser, not a
            # tumour term. Worse, an aggregate dilutes every cosine it enters: an
            # orthogonal member adds norm but no alignment, so cos(ncc, B) read
            # -0.30 where the pairwise cos(ncc, rigidity) driving it was -0.53.
            # The pairwise matrix is the signal; the only aggregate kept is the
            # whole objective, which needs no assumption about what groups with
            # what.
            losses = {
                "ncc": loss_multiNCC,
                "non_diff_loss": config.w_non_diff * loss_non_diff,
                "smooth": config.w_smooth * loss_regulation,
                "rigidity": config.w_bone_rigidity * loss_rigidity,
                "tlg": config.w_tlg * loss_tlg,
                "mtv": config.w_mtv * loss_mtv**2,
                "mtv_mean": config.w_mtv_mean * loss_mtv_mean,
                "jactum": config.w_jacobian_tumor * loss_jacobian_tumor,
            }
            if loss_dice_ct is not None:
                losses["dice"] = config.w_dice_ct_lvl3 * loss_dice_ct
            if loss_dice_pet is not None and use_dice_pet:
                losses["dice_pet"] = config.w_dice_pet * loss_dice_pet

            # Diagnostic probes: Dice restricted to bone labels and to
            # everything else. The rigidity loss is masked to bone, but its
            # gradient still reaches every conv weight and the field it produces
            # is smooth, so it can move soft tissue too -- which is why this is
            # measured rather than assumed. Probes, not objective terms: the two
            # class means do not sum back to the full dice term, so folding them
            # in would corrupt g_total and the shares.
            probes: Dict[str, torch.Tensor] = {}
            if loss_dice_ct is not None:
                # mirror the objective's dice term: original labels, total field
                x_lbl_ct_probe = X_lbl_ct_orig.float()
                is_bone_x = torch.isin(
                    x_lbl_ct_probe, bone_labels_tensor.to(x_lbl_ct_probe.dtype)
                )
                is_bone_y = torch.isin(Y_lbl_ct, bone_labels_tensor)
                for probe_name, keep_x, keep_y in (
                    ("dice_bone", is_bone_x, is_bone_y),
                    ("dice_soft", ~is_bone_x, ~is_bone_y),
                ):
                    probe_dice = utils.dice_loss_with_grad_bbox(
                        x_lbl_ct_probe * keep_x,
                        Y_lbl_ct * keep_y,
                        flow_total_ch,
                        model.grid_1,
                        transform,
                        use_checkpoint=True,
                    )
                    if probe_dice is not None:
                        probes[probe_name] = config.w_dice_ct_lvl3 * probe_dice

            # The same objective as one tensor, one backward. Logged alongside
            # the summed-per-term version so any bf16/checkpointing discrepancy
            # between the two shows up as agg_rel_err rather than being mistaken
            # for a change in the training trajectory.
            loss_single = sum(losses.values())

            raw = utils.gradient_conflict_report(
                losses,
                model,
                named_probes=probes,
                loss_total=loss_single,
            )
            if raw:
                windowed = grad_conflict_tracker.update(raw)
                utils.log_metrics(
                    {
                        f"grad_conflict_lvl3/{key}": value
                        for key, value in {**raw, **windowed}.items()
                    },
                    step=global_step,
                )

        loss_scaled = loss / config.accumulation_steps
        is_step = (global_step + 1) % config.accumulation_steps == 0 or is_last_step

        utils.optimizer_step_with_guard(
            loss, loss_scaled, optimizer, model, is_step, global_step, level=3
        )

        lossall[:, global_step] = np.array(
            [
                loss.item(),
                loss_multiNCC.item(),
                loss_non_diff.item(),
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
                non_diff=f"{loss_non_diff.item():.6f}",
                smo=f"{loss_regulation.item():.4f}",
                dvf=f"{loss_dvf.item():.8f}",
            )
        train_metrics = {
            "train_lvl3/loss": loss.item(),
            "train_lvl3/ncc_ct": loss_ncc_ct.item(),
            "train_lvl3/ncc_pet": loss_ncc_pet.item(),
            "train_lvl3/smooth": loss_regulation.item(),
            "train_lvl3/mtv_bias": loss_mtv.item(),
            "train_lvl3/mtv_mean": loss_mtv_mean.item(),
            "train_lvl3/tlg_bias": loss_tlg.item(),
            "train_lvl3/jacobian_tumor": loss_jacobian_tumor.item(),
            "train_lvl3/non_diff_loss": loss_non_diff.item(),
            "train_lvl3/rigidity": loss_rigidity.item(),
            # the three rigidity conditions on their own (unweighted) scale, so
            # their relative magnitudes are visible before w_rig_* is tuned
            "train_lvl3/rig_det": loss_rig_det.item(),
            "train_lvl3/rig_ortho": loss_rig_ortho.item(),
            "train_lvl3/rig_affine": loss_rig_affine.item(),
            "train_lvl3/w_rig_det": config.w_rig_det * loss_rig_det.item(),
            "train_lvl3/w_rig_ortho": config.w_rig_ortho * loss_rig_ortho.item(),
            "train_lvl3/w_rig_affine": config.w_rig_affine * loss_rig_affine.item(),
            # per-label rigid fit (zero unless config.use_per_label_rigidity)
            "train_lvl3/rig_worst": loss_rig_worst.item(),
            "train_lvl3/rig_worst_label": rig_worst_label.item(),
            "train_lvl3/rig_n_labels": n_rig_labels.item(),
            "train_lvl3/ndv": ndv,
            "train_lvl3/dvf": loss_dvf.item(),
            "train_lvl3/lr": current_lr,
        }
        if config.use_seg_pet_head:
            train_metrics["train_lvl3/seg_pet"] = loss_seg_pet.item()
            train_metrics["train_lvl3/w_seg_pet"] = w_seg_pet * loss_seg_pet.item()
            for key, value in seg_metrics.items():
                train_metrics[f"train_lvl3/{key}"] = value
        if config.use_seg_bone_head:
            train_metrics["train_lvl3/seg_bone"] = loss_seg_bone.item()
            train_metrics["train_lvl3/w_seg_bone"] = w_seg_bone * loss_seg_bone.item()
            for key, value in seg_bone_metrics.items():
                train_metrics[f"train_lvl3/{key}"] = value
        if loss_dice_ct is not None:
            train_metrics["train_lvl3/dice_ct"] = loss_dice_ct.item()
        if loss_dice_pet is not None:
            train_metrics["train_lvl3/dice_pet"] = loss_dice_pet.item()
        # weighted contributions (weight * term = actual share of the total loss)
        train_metrics["train_lvl3/w_ncc_ct"] = config.w_ct * loss_ncc_ct.item()
        train_metrics["train_lvl3/w_ncc_pet"] = (
            config.w_pet * loss_ncc_pet.item() if use_ncc_pet else 0.0
        )
        train_metrics["train_lvl3/w_non_diff"] = (
            config.w_non_diff * loss_non_diff.item()
        )
        train_metrics["train_lvl3/w_smooth"] = config.w_smooth * loss_regulation.item()
        train_metrics["train_lvl3/w_jacob_tumor"] = (
            config.w_jacobian_tumor * loss_jacobian_tumor.item()
        )
        train_metrics["train_lvl3/w_rigidity"] = (
            config.w_bone_rigidity * loss_rigidity.item()
        )
        train_metrics["train_lvl3/w_tlg"] = config.w_tlg * loss_tlg.item()
        train_metrics["train_lvl3/w_mtv"] = config.w_mtv * loss_mtv.item() ** 2
        train_metrics["train_lvl3/w_mtv_mean"] = (
            config.w_mtv_mean * loss_mtv_mean.item()
        )
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
                if config.use_seg_pet_head and not config.use_seg_bone_head:
                    print(
                        f"ep: {epoch}\t"
                        f"ncc={epoch_metrics['train_lvl3/ncc_ct']:.4f}\t"
                        f"dice_ct={epoch_metrics['train_lvl3/dice_ct']:.4f}\t"
                        f"non_diff={epoch_metrics['train_lvl3/non_diff_loss']:.6f}\t"
                        f"seg_dice_pet_fix={epoch_metrics['train_lvl3/seg_pet_dice_fixed']:.4f}\t"
                        f"seg_dice_pet_mov={epoch_metrics['train_lvl3/seg_pet_dice_moving']:.4f}\t"
                    )
                elif config.use_seg_bone_head and not config.use_seg_pet_head:
                    print(
                        f"ep: {epoch}\t"
                        f"ncc={epoch_metrics['train_lvl3/ncc_ct']:.4f}\t"
                        f"dice_ct={epoch_metrics['train_lvl3/dice_ct']:.4f}\t"
                        f"non_diff={epoch_metrics['train_lvl3/non_diff_loss']:.6f}\t"
                        f"seg_dice_bone_fix={epoch_metrics['train_lvl3/seg_bone_dice_fixed']:.4f}\t"
                        f"seg_dice_bone_mov={epoch_metrics['train_lvl3/seg_bone_dice_moving']:.4f}\t"
                    )
                elif config.use_seg_pet_head and config.use_seg_bone_head:
                    print(
                        f"ep: {epoch}\t"
                        f"ncc={epoch_metrics['train_lvl3/ncc_ct']:.4f}\t"
                        f"dice_ct={epoch_metrics['train_lvl3/dice_ct']:.4f}\t"
                        f"non_diff={epoch_metrics['train_lvl3/non_diff_loss']:.6f}\t"
                        f"seg_dice_pet_fix={epoch_metrics['train_lvl3/seg_pet_dice_fixed']:.4f}\t"
                        f"seg_dice_pet_mov={epoch_metrics['train_lvl3/seg_pet_dice_moving']:.4f}\t"
                        f"seg_dice_bone_fix={epoch_metrics['train_lvl3/seg_bone_dice_fixed']:.4f}\t"
                        f"seg_dice_bone_mov={epoch_metrics['train_lvl3/seg_bone_dice_moving']:.4f}\t"
                    )
                else:
                    print(
                        f"ep: {epoch}\t"
                        f"ncc={epoch_metrics['train_lvl3/ncc_ct']:.4f}\t"
                        f"dice_ct={epoch_metrics['train_lvl3/dice_ct']:.4f}\t"
                        f"non_diff={epoch_metrics['train_lvl3/non_diff_loss']:.6f}\t"
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

            # all three are "lower is better"; scales make the terms comparable
            tumour_score = (
                config.sel_w_mtv * val_losses["mtv_bias"] / config.sel_scale_mtv
                + config.sel_w_tlg * val_losses["tlg_bias"] / config.sel_scale_tlg
            )
            combined_score = (
                config.sel_w_dice * val_losses["dice_ct"] / config.sel_scale_dice_ct
                + tumour_score
            )
            utils.log_metrics(
                {
                    "valid_lvl3/sel_tumour_score": tumour_score,
                    "valid_lvl3/sel_combined_score": combined_score,
                },
                step=global_step,
            )

            def save_selected(model_path: Path, optimizer_path: Path) -> None:
                # every optimizer checkpoint carries all three bests, so a resume
                # from any of them will not re-save worse checkpoints over better
                torch.save(model.state_dict(), model_path)
                torch.save(
                    {
                        "optimizer": optimizer.state_dict(),
                        "global_step": global_step,
                        "best_dice_ct": best_dice_ct,
                        "best_tumour": best_tumour,
                        "best_combined": best_combined,
                    },
                    optimizer_path,
                )

            if val_losses["dice_ct"] < best_dice_ct:
                best_dice_ct = val_losses["dice_ct"]
                save_selected(best_model_path, best_optimizer_path)
                tqdm.tqdm.write(
                    f"step {global_step}: new best dice_ct {best_dice_ct:.4f} -> saved best"
                )

            if tumour_score < best_tumour:
                best_tumour = tumour_score
                save_selected(best_tumour_model_path, best_tumour_optimizer_path)
                tqdm.tqdm.write(
                    f"step {global_step}: new best tumour {best_tumour:.4f} "
                    f"(mtv {val_losses['mtv_bias']:.4f} tlg {val_losses['tlg_bias']:.4f})"
                    " -> saved best_tumour"
                )

            if combined_score < best_combined:
                best_combined = combined_score
                save_selected(best_combined_model_path, best_combined_optimizer_path)
                tqdm.tqdm.write(
                    f"step {global_step}: new best combined {best_combined:.4f}"
                    " -> saved best_combined"
                )

        if config.overfit is False:
            pbar.update(1)

        if is_epoch_end and utils.check_stop_flag(config.save_dir, 3):
            tqdm.tqdm.write(f"[lvl3] stop flag at step {global_step} -> ending level")
            utils.log_metrics({"train_lvl3/early_stopped": 1.0}, step=global_step)
            break

    if config.overfit is False:
        pbar.close()

    torch.save(model.state_dict(), final_model_path)

    result = {
        "final": final_model_path,
        "best": best_model_path,
        "best_tumour": best_tumour_model_path,
        "best_combined": best_combined_model_path,
        "best_optimizer": best_optimizer_path,
        "best_tumour_optimizer": best_tumour_optimizer_path,
        "best_combined_optimizer": best_combined_optimizer_path,
    }
    return result
