from pathlib import Path
from typing import Callable, Dict

import mlflow
import my_data
import numpy as np
import torch
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
    NCC,
    Miccai2020_LDR_laplacian_unit_add_lvl1,
    SpatialTransform_unit,
    SpatialTransformNearest_unit,
    jacobian_determinant,
    neg_Jdet_loss,
    smoothloss,
)
from torch.utils import data as torch_data


def evaluate_lvl1(
    model: Miccai2020_LDR_laplacian_unit_add_lvl1,
    val_generator: torch_data.DataLoader,
    config: TrainingConfig,
    device: torch.device,
    loss_similarity_ct: NCC,
    loss_similarity_pet: NCC,
    loss_smooth: Callable,
    loss_Jdet: Callable,
    transform: SpatialTransform_unit,
    grid_4: torch.Tensor,
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
        for batch in val_generator:
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

            F_X_Y, X_Y, Y_4x, F_xy, _ = model(X_affine, Y)

            if epoch % (config.val_interval * 5) == 0 or epoch == config.epochs_lvl1:
                if not saved_initial:
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
                        name="x_ref_ct_lvl1",
                    )
                    my_data.save_volume(
                        volume=x_affine[:, 0:1, ...],
                        out_dir=config.save_dir / "initial",
                        epoch=epoch,
                        name="x_affine_ct_lvl1",
                    )
                    my_data.save_volume(
                        volume=y_ref[:, 0:1, ...],
                        out_dir=config.save_dir / "initial",
                        epoch=epoch,
                        name="y_ref_ct_lvl1",
                    )
                    saved_initial = True

                if saved is False:
                    ct = X_Y[:, 0:1, :, :, :]
                    my_data.save_volume(
                        volume=ct,
                        out_dir=config.save_dir / "warped",
                        epoch=epoch,
                        name="warped_ct_lvl1",
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
            loss_jacobian = loss_Jdet(F_X_Y_norm, grid_4)

            _, _, x, y, z = F_xy.shape
            F_xy[:, 0, :, :, :] = F_xy[:, 0, :, :, :] * (z - 1)
            F_xy[:, 1, :, :, :] = F_xy[:, 1, :, :, :] * (y - 1)
            F_xy[:, 2, :, :, :] = F_xy[:, 2, :, :, :] * (x - 1)
            loss_regulation = loss_smooth(F_xy)

            X_lbl_ct = transform_nearest(X_lbl_ct.float(), flow_affine, grid_full)
            X_lbl_pet = transform_nearest(X_lbl_pet.float(), flow_affine, grid_full)

            X_lbl_ct_down = utils.downsample_label(
                X_lbl_ct.to(device), scale_factor=0.25
            )
            Y_lbl_ct_down = utils.downsample_label(
                Y_lbl_ct.to(device), scale_factor=0.25
            )
            X_lbl_pet_down = utils.downsample_label(
                X_lbl_pet.to(device), scale_factor=0.25
            )
            Y_lbl_pet_down = utils.downsample_label(
                Y_lbl_pet.to(device), scale_factor=0.25
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


def train_lvl1(
    config: TrainingConfig,
    train_generator: torch_data.DataLoader,
    val_generator: torch_data.DataLoader,
) -> Path:
    print("Training lvl1...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Miccai2020_LDR_laplacian_unit_add_lvl1(
        in_channel=config.in_channel,
        n_classes=config.n_classes,
        start_channel=config.start_channel,
        is_train=True,
        imgshape=config.img_shape_4,
        range_flow=config.range_flow,
    ).to(device)

    loss_similarity_ct = NCC(win=5)
    loss_similarity_pet = NCC(win=5)
    loss_smooth = smoothloss
    loss_Jdet = neg_Jdet_loss

    transform = SpatialTransform_unit().to(device)
    transform_nearest = SpatialTransformNearest_unit().to(device)

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True
    for param in transform_nearest.parameters():
        param.requires_grad = False

    grid_4 = generate_grid(config.img_shape_4)
    grid_4 = (
        torch.from_numpy(np.reshape(grid_4, (1,) + grid_4.shape)).to(device).float()
    )

    grid_full = generate_grid_unit(config.img_shape)
    grid_full = (
        torch.from_numpy(np.reshape(grid_full, (1,) + grid_full.shape))
        .to(device)
        .float()
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr_lvl1)

    config.save_dir.mkdir(parents=True, exist_ok=True)

    lossall = np.zeros((4, config.epochs_lvl1 + 1))
    final_model_path = (
        config.save_dir
        / f"{config.mlflow_experiment}_stagelvl1_{config.epochs_lvl1}.pth"
    )

    steps_per_epoch = len(train_generator)
    lossall = np.zeros((4, (config.epochs_lvl1 + 1) * steps_per_epoch))
    global_step = 0

    epoch = 0
    pbar = tqdm.tqdm(total=config.epochs_lvl1 + 1, desc="lvl1 training")

    saved_initial: bool = False

    while epoch <= config.epochs_lvl1:
        epoch_metrics: Dict[str, float] = {}
        n_steps = 0

        for batch in train_generator:
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

            F_X_Y, X_Y, Y_4x, F_xy, _ = model(X_affine, Y)

            # 3 level deep supervision NCC
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

            loss_jacobian = loss_Jdet(F_X_Y_norm, grid_4)

            # reg2 - use velocity
            _, _, x, y, z = F_xy.shape
            F_xy[:, 0, :, :, :] = F_xy[:, 0, :, :, :] * (z - 1)
            F_xy[:, 1, :, :, :] = F_xy[:, 1, :, :, :] * (y - 1)
            F_xy[:, 2, :, :, :] = F_xy[:, 2, :, :, :] * (x - 1)
            loss_regulation = loss_smooth(F_xy)

            X_lbl_ct = transform_nearest(X_lbl_ct.float(), flow_affine, grid_full)
            X_lbl_pet = transform_nearest(X_lbl_pet.float(), flow_affine, grid_full)

            X_lbl_ct_down = utils.downsample_label(
                X_lbl_ct.to(device), scale_factor=0.25
            )
            Y_lbl_ct_down = utils.downsample_label(
                Y_lbl_ct.to(device), scale_factor=0.25
            )
            X_lbl_pet_down = utils.downsample_label(
                X_lbl_pet.to(device), scale_factor=0.25
            )
            Y_lbl_pet_down = utils.downsample_label(
                Y_lbl_pet.to(device), scale_factor=0.25
            )

            if epoch == config.epochs_lvl1 and False:
                transform_nearest = SpatialTransformNearest_unit().to(device)

                warped_seg_ct = transform_nearest(
                    X_lbl_ct_down.float(),
                    F_X_Y.permute(0, 2, 3, 4, 1),
                    model.grid_1,
                )

                my_data.save_volume(
                    volume=warped_seg_ct.to(torch.int16),
                    out_dir=config.save_dir / "warped",
                    epoch=epoch,
                    name="x_warped",
                )
                my_data.save_volume(
                    volume=Y_lbl_ct_down.to(torch.int16),
                    out_dir=config.save_dir / "warped",
                    epoch=epoch,
                    name="y",
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

            optimizer.zero_grad()
            if torch.isfinite(loss):
                loss.backward()
                total_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                mlflow.log_metrics(
                    {"lvl1/grad_norm": total_norm.item()}, step=global_step
                )
                if not torch.isfinite(total_norm) or total_norm > 100.0:
                    tqdm.tqdm.write(
                        f"[lvl1] step {global_step}: grad_norm={total_norm.item():.2f} "
                        f"loss={loss.item():.4f} (skipped)"
                    )
                    optimizer.zero_grad()
                else:
                    optimizer.step()
            else:
                tqdm.tqdm.write(f"[lvl1] step {global_step}: non-finite loss (skipped)")

            lossall[:, global_step] = np.array(
                [
                    loss.item(),
                    loss_multiNCC.item(),
                    loss_jacobian.item(),
                    loss_regulation.item(),
                ]
            )
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
            )
            train_metrics = {
                "train_lvl1/loss": loss.item(),
                "train_lvl1/ncc_ct": loss_ncc_ct.item(),
                "train_lvl1/ncc_pet": loss_ncc_pet.item(),
                "train_lvl1/smooth": loss_regulation.item(),
                "train_lvl1/jacob": loss_jacobian.item(),
                "train_lvl1/ndv": ndv,
            }
            if loss_dice_ct is not None:
                train_metrics["train_lvl1/dice_ct"] = loss_dice_ct.item()
            if loss_dice_pet is not None:
                train_metrics["train_lvl1/dice_pet"] = loss_dice_pet.item()
            mlflow.log_metrics(train_metrics, step=global_step)

            for key, value in train_metrics.items():
                epoch_metrics[key] = epoch_metrics.get(key, 0.0) + value
            n_steps += 1
            global_step += 1

        mlflow.log_metrics(
            {f"{key}_epoch": value / n_steps for key, value in epoch_metrics.items()},
            step=global_step,
        )

        if epoch % config.val_interval == 0 or epoch == config.epochs_lvl1:
            val_losses = evaluate_lvl1(
                model=model,
                val_generator=val_generator,
                config=config,
                device=device,
                loss_similarity_ct=loss_similarity_ct,
                loss_similarity_pet=loss_similarity_pet,
                loss_smooth=loss_smooth,
                loss_Jdet=loss_Jdet,
                transform=transform,
                grid_4=grid_4,
                epoch=epoch,
                saved_initial=saved_initial,
            )
            saved_initial = True
            mlflow.log_metrics(
                {
                    f"valid_lvl1/val_{key}": value
                    for key, value in val_losses.items()
                    if not (isinstance(value, float) and np.isnan(value))
                },
                step=global_step,
            )
            tqdm.tqdm.write(
                f"epoch {epoch} -> val loss {val_losses['loss']:.4f} "
                f"- ncc_ct {val_losses['ncc_ct']:.4f} "
                f"- ncc_pet {val_losses['ncc_pet']:.4f} "
                f"- dice_ct {val_losses['dice_ct']:.4f} "
                f"- dice_pet {val_losses['dice_pet']:.4f}"
            )

        # save final model
        if epoch == config.epochs_lvl1:
            torch.save(model.state_dict(), final_model_path)

        utils.save_checkpoint(model, optimizer, epoch, "lvl1", config, lossall)

        epoch += 1
        pbar.update(1)

        if epoch > config.epochs_lvl1:
            break
    pbar.close()

    return final_model_path
