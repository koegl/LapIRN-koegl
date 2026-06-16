from pathlib import Path
from typing import Callable, Dict

import mlflow
import my_data
import numpy as np
import torch
import tqdm
import utils
from config import TrainingConfig
from Functions import generate_grid, transform_unit_flow_to_flow_cuda
from miccai2020_model_stage import (
    Miccai2020_LDR_laplacian_unit_add_lvl1,
    Miccai2020_LDR_laplacian_unit_add_lvl2,
    Miccai2020_LDR_laplacian_unit_add_lvl3,
    SpatialTransform_unit,
    jacobian_determinant,
    multi_resolution_NCC,
    neg_Jdet_loss,
    smoothloss,
)
from torch.utils import data as torch_data


def evaluate_lvl3(
    model: Miccai2020_LDR_laplacian_unit_add_lvl3,
    val_generator: torch_data.DataLoader,
    config: TrainingConfig,
    device: torch.device,
    loss_similarity_ct: multi_resolution_NCC,
    loss_similarity_pet: multi_resolution_NCC,
    loss_smooth: Callable,
    loss_Jdet: Callable,
    transform: SpatialTransform_unit,
    grid: torch.Tensor,
) -> Dict[str, float]:
    """Run one validation pass over val_generator and return averaged losses.

    Args:
        model: The lvl3 registration model, switched to eval mode internally
            and restored to train mode before returning.
        val_generator: DataLoader yielding (X, Y, X_lbl_ct, X_lbl_pet,
            Y_lbl_ct, Y_lbl_pet) batches.
        config: Training configuration holding loss weights.
        device: Device to run the forward pass on.
        loss_similarity_ct: Multi-resolution NCC loss for the CT channel.
        loss_similarity_pet: Multi-resolution NCC loss for the PET channel.
        loss_smooth: Smoothness regularization loss function.
        loss_Jdet: Negative Jacobian determinant loss function.
        transform: Spatial transform module used by the dice loss.
        grid: Precomputed sampling grid at full resolution.

    Returns:
        Dict mapping loss name to its average value over val_generator.
    """
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
        "masked_jac": 0.0,
        "ndv": 0.0,
    }
    n_batches = 0

    with torch.no_grad():
        for X, Y, X_lbl_ct, X_lbl_pet, Y_lbl_ct, Y_lbl_pet in val_generator:
            X = X.to(device).float()
            Y = Y.to(device).float()

            F_X_Y, X_Y, Y_4x, F_xy, _, _, _ = model(X, Y)

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
            loss_jacobian = loss_Jdet(F_X_Y_norm, grid)

            _, _, x, y, z = F_xy.shape
            F_xy[:, 0, :, :, :] = F_xy[:, 0, :, :, :] * (z - 1)
            F_xy[:, 1, :, :, :] = F_xy[:, 1, :, :, :] * (y - 1)
            F_xy[:, 2, :, :, :] = F_xy[:, 2, :, :, :] * (x - 1)
            loss_regulation = loss_smooth(F_xy)

            loss_dice_ct = utils.dice_loss_with_grad(
                X_lbl_ct.to(device), Y_lbl_ct.to(device), F_X_Y, model.grid_1, transform
            )
            loss_dice_pet = utils.dice_loss_with_grad(
                X_lbl_pet.to(device),
                Y_lbl_pet.to(device),
                F_X_Y,
                model.grid_1,
                transform,
            )

            moving_pet_mask = (X_lbl_pet.to(device) == 1).float()
            warped_pet_mask = utils.warp_binary_mask(
                moving_pet_mask, F_X_Y, model.grid_1, transform
            )
            warped_pet_image = X_Y[:, 1:2]
            moving_pet_image = X[:, 1:2]

            loss_mtv = utils.mtv_bias_loss(warped_pet_mask, moving_pet_mask)
            loss_tlg = utils.tlg_bias_loss(
                warped_pet_image, warped_pet_mask, moving_pet_image, moving_pet_mask
            )
            loss_masked_jac = utils.masked_jac_det_loss(jac_det, moving_pet_mask)

            loss = (
                loss_multiNCC
                + config.w_jacobian * loss_jacobian
                + config.w_smooth * loss_regulation
                + config.w_dice_ct * loss_dice_ct
                + config.w_dice_pet * loss_dice_pet
                + config.w_mtv * loss_mtv
                + config.w_tlg * loss_tlg
                + config.w_masked_jac * loss_masked_jac
            )

            val_losses["loss"] += loss.item()
            val_losses["ncc_ct"] += loss_ncc_ct.item()
            val_losses["ncc_pet"] += loss_ncc_pet.item()
            val_losses["smooth"] += loss_regulation.item()
            val_losses["dice_ct"] += loss_dice_ct.item()
            val_losses["dice_pet"] += loss_dice_pet.item()
            val_losses["jacobian"] += loss_jacobian.item()
            val_losses["mtv_bias"] += loss_mtv.item()
            val_losses["tlg_bias"] += loss_tlg.item()
            val_losses["masked_jac"] += loss_masked_jac.item()
            val_losses["ndv"] += ndv
            n_batches += 1

    model.train()
    return {key: value / n_batches for key, value in val_losses.items()}


def train_lvl3(
    config: TrainingConfig,
    path_model_level2: Path,
    train_generator: torch_data.DataLoader,
    val_generator: torch_data.DataLoader,
) -> Path:
    """Train the full-resolution lvl3 model on top of a frozen lvl1+lvl2 stack.

    Args:
        config: Training configuration (paths, hyperparameters, loss
            weights, epoch count, validation interval, freeze epoch).
        path_model_level2: Path to the trained lvl2 checkpoint to load and
            embed as the frozen lvl2 sub-model (which itself wraps lvl1).
        train_generator: DataLoader yielding training batches.
        val_generator: DataLoader yielding validation batches.

    Returns:
        Path to the final saved model checkpoint.
    """
    print("Training lvl3...")
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

    loss_similarity_ct = multi_resolution_NCC(win=7, scale=2)
    loss_similarity_pet = multi_resolution_NCC(win=7, scale=2)
    loss_smooth = smoothloss
    loss_Jdet = neg_Jdet_loss

    transform = SpatialTransform_unit().to(device)

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True

    grid = generate_grid(config.img_shape)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).to(device).float()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    config.save_dir.mkdir(parents=True, exist_ok=True)

    lossall = np.zeros((4, config.epochs_lvl3 + 1))
    final_model_path = (
        config.save_dir
        / f"{config.mlflow_experiment}_stagelvl3_{config.epochs_lvl3}.pth"
    )

    steps_per_epoch = len(train_generator)
    lossall = np.zeros((4, (config.epochs_lvl1 + 1) * steps_per_epoch))
    global_step = 0

    epoch = 0
    pbar = tqdm.tqdm(total=config.epochs_lvl3 + 1, desc="lvl3 training")
    while epoch <= config.epochs_lvl3:
        epoch_metrics: Dict[str, float] = {}
        n_steps = 0

        for X, Y, X_lbl_ct, X_lbl_pet, Y_lbl_ct, Y_lbl_pet in train_generator:
            X = X.to(device).float()
            Y = Y.to(device).float()

            F_X_Y, X_Y, Y_4x, F_xy, _, _, _ = model(X, Y)

            if epoch % 100 == 0 or epoch == config.epochs_lvl3:
                ct = X_Y[:, 0:1, :, :, :]
                pet = X_Y[:, 1:2, :, :, :]
                my_data.save_volume(
                    volume=ct,
                    out_dir=config.save_dir / "warped",
                    epoch=epoch,
                    name="warped_ct_lvl3",
                )
                my_data.save_volume(
                    volume=pet,
                    out_dir=config.save_dir / "warped",
                    epoch=epoch,
                    name="warped_pet_lvl3",
                )

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

            loss_jacobian = loss_Jdet(F_X_Y_norm, grid)

            # reg2 - use velocity
            _, _, x, y, z = F_xy.shape
            F_xy[:, 0, :, :, :] = F_xy[:, 0, :, :, :] * (z - 1)
            F_xy[:, 1, :, :, :] = F_xy[:, 1, :, :, :] * (y - 1)
            F_xy[:, 2, :, :, :] = F_xy[:, 2, :, :, :] * (x - 1)
            loss_regulation = loss_smooth(F_xy)

            # full-resolution labels (no downsampling at lvl3)
            loss_dice_ct = utils.dice_loss_with_grad(
                X_lbl_ct.to(device), Y_lbl_ct.to(device), F_X_Y, model.grid_1, transform
            )
            loss_dice_pet = utils.dice_loss_with_grad(
                X_lbl_pet.to(device),
                Y_lbl_pet.to(device),
                F_X_Y,
                model.grid_1,
                transform,
            )

            # --- MTV / TLG / masked Jacobian losses (lvl3 only) ---
            moving_pet_mask = (X_lbl_pet.to(device) == 1).float()
            warped_pet_mask = utils.warp_binary_mask(
                moving_pet_mask, F_X_Y, model.grid_1, transform
            )
            # X_Y channel 1 is already the warped moving PET image
            warped_pet_image = X_Y[:, 1:2]
            moving_pet_image = X[:, 1:2]

            loss_mtv = utils.mtv_bias_loss(warped_pet_mask, moving_pet_mask)
            loss_tlg = utils.tlg_bias_loss(
                warped_pet_image, warped_pet_mask, moving_pet_image, moving_pet_mask
            )
            # enforce det(J)=1 inside tumor
            loss_masked_jac = utils.masked_jac_det_loss(jac_det, moving_pet_mask)

            # update total loss
            loss = (
                loss_multiNCC
                + config.w_jacobian * loss_jacobian
                + config.w_smooth * loss_regulation
                + config.w_dice_ct * loss_dice_ct
                + config.w_dice_pet * loss_dice_pet
                + config.w_mtv * loss_mtv
                + config.w_tlg * loss_tlg
                + config.w_masked_jac * loss_masked_jac
            )

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()  # apply gradients

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
                dice_ct=f"{loss_dice_ct.item():.4f}",
                dice_pet=f"{loss_dice_pet.item():.4f}",
                mtv=f"{loss_mtv.item():.4f}",
                tlg=f"{loss_tlg.item():.4f}",
                Jdet=f"{loss_jacobian.item():.6f}",
                smo=f"{loss_regulation.item():.4f}",
            )
            train_metrics = {
                "train_lvl3/loss": loss.item(),
                "train_lvl3/ncc_ct": loss_ncc_ct.item(),
                "train_lvl3/ncc_pet": loss_ncc_pet.item(),
                "train_lvl3/smooth": loss_regulation.item(),
                "train_lvl3/dice_ct": loss_dice_ct.item(),
                "train_lvl3/dice_pet": loss_dice_pet.item(),
                "train_lvl3/mtv_bias": loss_mtv.item(),
                "train_lvl3/tlg_bias": loss_tlg.item(),
                "train_lvl3/masked_jac": loss_masked_jac.item(),
                "train_lvl3/jacob": loss_jacobian.item(),
                "train_lvl3/ndv": ndv,
            }
            mlflow.log_metrics(train_metrics, step=global_step)
            for key, value in train_metrics.items():
                epoch_metrics[key] = epoch_metrics.get(key, 0.0) + value
            n_steps += 1
            global_step += 1

        mlflow.log_metrics(
            {f"{key}_epoch": value / n_steps for key, value in epoch_metrics.items()},
            step=global_step,
        )

        if epoch % config.val_interval == 0 or epoch == config.epochs_lvl3:
            val_losses = evaluate_lvl3(
                model=model,
                val_generator=val_generator,
                config=config,
                device=device,
                loss_similarity_ct=loss_similarity_ct,
                loss_similarity_pet=loss_similarity_pet,
                loss_smooth=loss_smooth,
                loss_Jdet=loss_Jdet,
                transform=transform,
                grid=grid,
            )
            mlflow.log_metrics(
                {f"valid_lvl3/val_{key}": value for key, value in val_losses.items()},
                step=global_step,
            )
            tqdm.tqdm.write(
                f"epoch {epoch} -> val loss {val_losses['loss']:.4f} "
                f"- ncc_ct {val_losses['ncc_ct']:.4f} "
                f"- ncc_pet {val_losses['ncc_pet']:.4f} "
                f"- dice_ct {val_losses['dice_ct']:.4f} "
                f"- dice_pet {val_losses['dice_pet']:.4f} "
                f"- mtv {val_losses['mtv_bias']:.4f} "
                f"- tlg {val_losses['tlg_bias']:.4f}"
            )

        if epoch == config.unfreeze_epoch_in_lvl3:
            model.unfreeze_modellvl2()

        if epoch == config.epochs_lvl3:
            torch.save(model.state_dict(), final_model_path)
            np.save(
                config.save_dir
                / f"loss_{config.mlflow_experiment}_stagelvl3_{config.epochs_lvl3}.npy",
                lossall,
            )

        utils.save_checkpoint(model, optimizer, epoch, "lvl3", config, lossall)

        epoch += 1
        pbar.update(1)

        if epoch > config.epochs_lvl3:
            break
    pbar.close()

    return final_model_path
