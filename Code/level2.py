from pathlib import Path
from typing import Callable, Dict

import mlflow
import my_data
import numpy as np
import synthetic
import torch
import torch.nn.functional as F
import tqdm
import utils
from affine_reg import create_affine_flow
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
    saved_initial: bool,
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

            flow_affine = create_affine_flow(
                config=config,
                device=device,
                case_id=batch["case_id"][0],
                tp_x=batch["tp_x"][0],
                tp_y=batch["tp_y"][0],
                aug_flipped=batch["aug_flipped"],
                aug_crop_head=batch["aug_crop_head"],
                aug_crop_feet=batch["aug_crop_feet"],
            )

            X_affine = transform(X, flow_affine, grid_full)

            F_X_Y, X_Y, Y_4x, F_xy, _, _ = model(X_affine, Y)
            if epoch % (config.val_interval * 50) == 0 or epoch == config.epochs_lvl2:
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
) -> Dict[str, Path]:
    print("Training lvl2...")

    best_dice_ct = float("inf")
    config.model_save_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = (
        config.model_save_dir
        / f"{config.mlflow_experiment}_{mlflow.active_run().info.run_name}_stagelvl2_best.pth"
    )
    final_model_path = (
        config.model_save_dir
        / f"{config.mlflow_experiment}_{mlflow.active_run().info.run_name}_stagelvl2_{config.epochs_lvl1}.pth"
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
    scaler = torch.amp.GradScaler()

    config.save_dir.mkdir(parents=True, exist_ok=True)

    lossall = np.zeros((4, config.epochs_lvl2 + 1))

    steps_per_epoch = len(train_generator)
    lossall = np.zeros((4, (config.epochs_lvl2 + 1) * steps_per_epoch))
    global_step = 0

    epoch = 0
    # pbar = tqdm.tqdm(total=config.epochs_lvl2 + 1, desc="lvl2 training")

    saved_initial: int = 0
    run_name = mlflow.active_run().info.run_name

    while epoch <= config.epochs_lvl2:
        epoch_metrics: Dict[str, float] = {}
        n_steps = 0

        for batch in train_generator:
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
                            synthetic.generate_synthetic_moving(
                                source=Y,
                                source_label_ct=Y_lbl_ct,
                                source_label_pet=Y_lbl_pet,
                                bone_label_values=synthetic.BONE_LABEL_VALUES,
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
                        synthetic.generate_synthetic_moving(
                            source=Y,
                            source_label_ct=Y_lbl_ct,
                            source_label_pet=Y_lbl_pet,
                            bone_label_values=synthetic.BONE_LABEL_VALUES,
                            device=device,
                        )
                    )
                X_affine = X_full
                X = X_affine
            else:
                X = batch["x"].to(device).float()
                X_lbl_ct = batch["x_label_ct"].to(device)
                X_lbl_pet = batch["x_label_pet"].to(device)
                gt_unit = None

                flow_affine = create_affine_flow(
                    config=config,
                    device=device,
                    case_id=batch["case_id"][0],
                    tp_x=batch["tp_x"][0],
                    tp_y=batch["tp_y"][0],
                    aug_flipped=batch["aug_flipped"],
                    aug_crop_head=batch["aug_crop_head"],
                    aug_crop_feet=batch["aug_crop_feet"],
                )
                X_affine = transform(X, flow_affine, grid_full)

            F_X_Y, X_Y, Y_4x, F_xy, _, _ = model(X_affine, Y)

            if saved_initial < len(train_generator):
                zero_disp = torch.zeros_like(F_X_Y)
                x_ref = model.transform(
                    X, zero_disp.permute(0, 2, 3, 4, 1), model.grid_1
                )
                x_affine = model.transform(
                    X_affine, zero_disp.permute(0, 2, 3, 4, 1), model.grid_1
                )
                y_ref = model.transform(
                    Y, zero_disp.permute(0, 2, 3, 4, 1), model.grid_1
                )
                my_data.save_volume(
                    volume=x_ref[:, 0:1, ...],
                    out_dir=config.save_dir / "initial",
                    epoch=epoch,
                    name=f"x_ref_ct_lvl2_{run_name}_{batch['case_id'][0]}",
                )
                my_data.save_volume(
                    volume=x_affine[:, 0:1, ...],
                    out_dir=config.save_dir / "initial",
                    epoch=epoch,
                    name=f"x_affine_ct_lvl2_{run_name}_{batch['case_id'][0]}",
                )
                my_data.save_volume(
                    volume=Y_4x[:, 0:1, ...],
                    out_dir=config.save_dir / "initial",
                    epoch=epoch,
                    name=f"y_ref_ct_lvl2_{run_name}_{batch['case_id'][0]}",
                )
                saved_initial += 1

            if epoch == 0 or epoch == config.epochs_lvl2:
                # in the epoch==0 warped block:
                print(f"{F_X_Y.abs().mean().item()=} {F_X_Y.abs().max().item()=}")
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

            loss_ncc_ct = loss_similarity_ct(X_Y_ct, Y_4x_ct)
            loss_ncc_pet = loss_similarity_pet(X_Y_pet, Y_4x_pet)
            loss_multiNCC = config.w_ct * loss_ncc_ct + config.w_pet * loss_ncc_pet

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

            if epoch == config.epochs_lvl2 and False:
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
            if loss_dice_pet is not None:
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
            is_step = (global_step + 1) % config.accumulation_steps == 0
            is_last_in_epoch = (
                n_steps + 1
            ) == steps_per_epoch  # trailing-partial flush

            if torch.isfinite(loss):
                loss_scaled.backward()

                if is_step or is_last_in_epoch:
                    total_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=1.0
                    )
                    mlflow.log_metrics(
                        {"lvl2/grad_norm": total_norm.item()}, step=global_step
                    )
                    if not torch.isfinite(total_norm) or total_norm > 100.0:
                        tqdm.tqdm.write(
                            f"[lvl2] step {global_step}: grad_norm={total_norm.item():.2f} "
                            f"loss={loss.item():.4f} (skipped)"
                        )
                        optimizer.zero_grad()
                    else:
                        optimizer.step()
                        optimizer.zero_grad()
            else:
                tqdm.tqdm.write(f"[lvl2] step {global_step}: non-finite loss (skipped)")
                optimizer.zero_grad()  # drop any partial accumulation from this bad micro-step

            lossall[:, global_step] = np.array(
                [
                    loss.item(),
                    loss_multiNCC.item(),
                    loss_jacobian.item(),
                    loss_regulation.item(),
                ]
            )
            # pbar.set_postfix(
            #     loss=f"{loss.item():.4f}",
            #     ncc=f"{loss_multiNCC.item():.4f}",
            #     dice_ct=f"{loss_dice_ct.item():.4f}"
            #     if loss_dice_ct is not None
            #     else "n/a",
            #     dice_pet=f"{loss_dice_pet.item():.4f}"
            #     if loss_dice_pet is not None
            #     else "n/a",
            #     Jdet=f"{loss_jacobian.item():.6f}",
            #     smo=f"{loss_regulation.item():.4f}",
            #     dvf=f"{loss_dvf.item():.8f}",
            # )
            train_metrics = {
                "train_lvl2/loss": loss.item(),
                "train_lvl2/ncc_ct": loss_ncc_ct.item(),
                "train_lvl2/ncc_pet": loss_ncc_pet.item(),
                "train_lvl2/smooth": loss_regulation.item(),
                "train_lvl2/jacob": loss_jacobian.item(),
                "train_lvl2/ndv": ndv,
                "train_lvl3/dvf": loss_dvf.item(),
            }
            if loss_dice_ct is not None:
                train_metrics["train_lvl2/dice_ct"] = loss_dice_ct.item()
            if loss_dice_pet is not None:
                train_metrics["train_lvl2/dice_pet"] = loss_dice_pet.item()
            mlflow.log_metrics(train_metrics, step=global_step)

            for key, value in train_metrics.items():
                epoch_metrics[key] = epoch_metrics.get(key, 0.0) + value
            n_steps += 1
            global_step += 1

        mlflow.log_metrics(
            {f"{key}_epoch": value / n_steps for key, value in epoch_metrics.items()},
            step=global_step,
        )

        print(
            f"ep: {epoch} "
            f"ncc={epoch_metrics['train_lvl2/ncc_ct'] / len(train_generator):.4f} "
            f"dice={epoch_metrics['train_lvl2/dice_ct'] / len(train_generator):.4f}"
        )

        if False and (epoch % config.val_interval == 0 or epoch == config.epochs_lvl2):
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
                saved_initial=saved_initial,
            )
            saved_initial = True
            mlflow.log_metrics(
                {f"valid_lvl2/val_{key}": value for key, value in val_losses.items()},
                step=global_step,
            )
            tqdm.tqdm.write(
                f"epoch {epoch} -> val loss {val_losses['loss']:.4f} "
                f"- ncc_ct {val_losses['ncc_ct']:.4f} "
                f"- ncc_pet {val_losses['ncc_pet']:.4f} "
                f"- dice_ct {val_losses['dice_ct']:.4f} "
                f"- dice_pet {val_losses['dice_pet']:.4f}"
            )

            if val_losses["dice_ct"] < best_dice_ct:
                best_dice_ct = val_losses["dice_ct"]
                torch.save(model.state_dict(), best_model_path)
                tqdm.tqdm.write(
                    f"epoch {epoch}: new best dice_ct {best_dice_ct:.4f} -> saved best"
                )
                print(
                    f"epoch {epoch}: new best dice_ct {best_dice_ct:.4f} -> saved best"
                )

        if epoch == config.unfreeze_epoch_in_lvl2:
            model.unfreeze_modellvl1()

        if epoch == config.epochs_lvl2:
            torch.save(model.state_dict(), final_model_path)
            np.save(
                config.save_dir
                / f"loss_{config.mlflow_experiment}_stagelvl2_{config.epochs_lvl2}.npy",
                lossall,
            )
        # utils.save_checkpoint(model, optimizer, epoch, "lvl2", config, lossall)

        epoch += 1
        # pbar.update(1)

        if epoch > config.epochs_lvl2:
            break
    # pbar.close()

    return {"final": final_model_path, "best": best_model_path}
