import glob
import json
import os
import sys
from pathlib import Path

import level1
import mlflow
import my_data
import numpy as np
import torch
from config import TrainingConfig
from Functions import generate_grid, transform_unit_flow_to_flow_cuda
from miccai2020_model_stage import (
    Miccai2020_LDR_laplacian_unit_add_lvl1,
    Miccai2020_LDR_laplacian_unit_add_lvl2,
    Miccai2020_LDR_laplacian_unit_add_lvl3,
    SpatialTransform_unit,
    SpatialTransformNearest_unit,
    jacobian_determinant,
    masked_jac_det_loss,
    mtv_bias_loss,
    multi_resolution_NCC,
    neg_Jdet_loss,
    smoothloss,
    tlg_bias_loss,
    warp_binary_mask,
)
from torch.utils import data as torch_data


def train_lvl2(path_model_level1: str) -> str:
    print("Training lvl2...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_lvl1 = Miccai2020_LDR_laplacian_unit_add_lvl1(
        4, 3, START_CHANNEL, is_train=True, imgshape=imgshape_4, range_flow=range_flow
    ).to(device)

    model_path = sorted(
        glob.glob(
            "/home/iml/fryderyk.koegl/code/LapIRN-koegl/checkpoints/Model/" + "*.pth"
        )
    )[-1]
    model_lvl1.load_state_dict(torch.load(path_model_level1))
    print("Loading weight for model_lvl1...", path_model_level1)

    for param in model_lvl1.parameters():
        param.requires_grad = False

    model = Miccai2020_LDR_laplacian_unit_add_lvl2(
        4,
        3,
        START_CHANNEL,
        is_train=True,
        imgshape=imgshape_2,
        range_flow=range_flow,
        model_lvl1=model_lvl1,
    ).to(device)

    loss_similarity_ct = multi_resolution_NCC(win=7, scale=2)
    loss_similarity_pet = multi_resolution_NCC(win=7, scale=2)
    loss_smooth = smoothloss
    loss_Jdet = neg_Jdet_loss

    transform = SpatialTransform_unit().to(device)

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True

    grid_2 = generate_grid(imgshape_2)
    grid_2 = (
        torch.from_numpy(np.reshape(grid_2, (1,) + grid_2.shape)).to(device).float()
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    model_dir = "/home/iml/fryderyk.koegl/code/LapIRN-koegl/Model"

    if not os.path.isdir(model_dir):
        Path(model_dir).mkdir(parents=True, exist_ok=True)

    lossall = np.zeros((4, config.iteration_lvl2 + 1))

    step = 0
    load_model = False
    if load_model is True:
        model_path = "../Model/LDR_LPBA_NCC_lap_share_preact_1_05_3000.pth"
        print("Loading weight: ", model_path)
        step = 3000
        model.load_state_dict(torch.load(model_path))
        temp_lossall = np.load(
            "../Model/loss_LDR_LPBA_NCC_lap_share_preact_1_05_3000.npy"
        )
        lossall[:, 0:3000] = temp_lossall[:, 0:3000]

    while step <= config.iteration_lvl2:
        for X, Y, X_lbl_ct, X_lbl_pet, Y_lbl_ct, Y_lbl_pet in training_generator:
            X = X.to(device).float()
            Y = Y.to(device).float()

            F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, _ = model(X, Y)

            if step % 1000 == 0 or step == config.iteration_lvl2:
                ct = X_Y[:, 0:1, :, :, :]
                pet = X_Y[:, 1:2, :, :, :]
                save_volume(
                    ct,
                    Path(model_dir) / "warped",
                    step,
                    Path(DATA_PATH) / "fixed_ct.nii.gz",
                    "warped_ct_lvl2",
                )
                save_volume(
                    pet,
                    Path(model_dir) / "warped",
                    step,
                    Path(DATA_PATH) / "fixed_ct.nii.gz",
                    "warped_pet_lvl2",
                )

            loss_ncc_ct = loss_similarity_ct(X_Y[:, 0:1, ...], Y_4x[:, 0:1, ...])
            loss_ncc_pet = loss_similarity_pet(X_Y[:, 1:2, ...], Y_4x[:, 1:2, ...])

            loss_multiNCC = W_CT * loss_ncc_ct + W_PET * loss_ncc_pet

            F_X_Y_norm = transform_unit_flow_to_flow_cuda(
                F_X_Y.permute(0, 2, 3, 4, 1).clone()
            )
            jac_det = jacobian_determinant(F_X_Y_norm)
            ndv = compute_ndv(jac_det)

            loss_Jacobian = loss_Jdet(F_X_Y_norm, grid_2)

            _, _, x, y, z = F_xy.shape
            F_xy[:, 0, :, :, :] = F_xy[:, 0, :, :, :] * (z - 1)
            F_xy[:, 1, :, :, :] = F_xy[:, 1, :, :, :] * (y - 1)
            F_xy[:, 2, :, :, :] = F_xy[:, 2, :, :, :] * (x - 1)
            loss_regulation = loss_smooth(F_xy)

            X_lbl_ct_down = downsample_label(X_lbl_ct.to(device), scale_factor=0.5)
            Y_lbl_ct_down = downsample_label(Y_lbl_ct.to(device), scale_factor=0.5)
            X_lbl_pet_down = downsample_label(X_lbl_pet.to(device), scale_factor=0.5)
            Y_lbl_pet_down = downsample_label(Y_lbl_pet.to(device), scale_factor=0.5)

            loss_dice_ct = dice_loss_with_grad(
                X_lbl_ct_down, Y_lbl_ct_down, F_X_Y, model.grid_1, transform
            )
            loss_dice_pet = dice_loss_with_grad(
                X_lbl_pet_down, Y_lbl_pet_down, F_X_Y, model.grid_1, transform
            )

            # update total loss
            loss = (
                loss_multiNCC
                + ANTIFOLD * loss_Jacobian
                + SMOOTH * loss_regulation
                + W_DICE_CT * loss_dice_ct
                + W_DICE_PET * loss_dice_pet
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            lossall[:, step] = np.array(
                [
                    loss.item(),
                    loss_multiNCC.item(),
                    loss_Jacobian.item(),
                    loss_regulation.item(),
                ]
            )
            sys.stdout.write(
                "\r"
                + 'step "{0}" -> training loss "{1:.4f}" - sim_NCC "{2:4f}" - dice_ct "{3:4f}" - dice_pet "{4:4f}" - Jdet "{5:.10f}" -smo "{6:.4f}"'.format(
                    step,
                    loss.item(),
                    loss_multiNCC.item(),
                    loss_dice_ct.item(),
                    loss_dice_pet.item(),
                    loss_Jacobian.item(),
                    loss_regulation.item(),
                )
            )
            sys.stdout.flush()
            mlflow.log_metrics(
                {
                    "lvl2/loss": loss.item(),
                    "lvl2/ncc_ct": loss_ncc_ct.item(),
                    "lvl2/ncc_pet": loss_ncc_pet.item(),
                    "lvl2/smooth": loss_regulation.item(),
                    "lvl2/dice_ct": loss_dice_ct.item(),
                    "lvl2/dice_pet": loss_dice_pet.item(),
                    "lvl2/ndv": ndv,
                },
                step=step,
            )

            if step == config.iteration_lvl2:
                modelname = (
                    model_dir + "/" + MODEL_NAME + "stagelvl2_" + str(step) + ".pth"
                )
                torch.save(model.state_dict(), modelname)
                np.save(
                    model_dir
                    + "/loss"
                    + MODEL_NAME
                    + "stagelvl2_"
                    + str(step)
                    + ".npy",
                    lossall,
                )

            if step == FREEZE_STEP:
                model.unfreeze_modellvl1()

            step += 1

            if step > config.iteration_lvl2:
                break
        print("one epoch pass")
    # np.save(model_dir + "/loss" + model_name + "stagelvl2.npy", lossall)

    return modelname


def train_lvl3(path_model_level2: str) -> None:
    print("Training lvl3...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_lvl1 = Miccai2020_LDR_laplacian_unit_add_lvl1(
        4, 3, START_CHANNEL, is_train=True, imgshape=imgshape_4, range_flow=range_flow
    ).to(device)
    model_lvl2 = Miccai2020_LDR_laplacian_unit_add_lvl2(
        4,
        3,
        START_CHANNEL,
        is_train=True,
        imgshape=imgshape_2,
        range_flow=range_flow,
        model_lvl1=model_lvl1,
    ).to(device)

    model_path = sorted(
        glob.glob(
            "/home/iml/fryderyk.koegl/code/LapIRN-koegl/checkpoints/Model/" + "*.pth"
        )
    )[-1]
    model_lvl2.load_state_dict(torch.load(path_model_level2))
    print("Loading weight for model_lvl2...", path_model_level2)

    for param in model_lvl2.parameters():
        param.requires_grad = False

    model = Miccai2020_LDR_laplacian_unit_add_lvl3(
        4,
        3,
        START_CHANNEL,
        is_train=True,
        imgshape=imgshape,
        range_flow=range_flow,
        model_lvl2=model_lvl2,
    ).to(device)

    loss_similarity_ct = multi_resolution_NCC(win=7, scale=2)
    loss_similarity_pet = multi_resolution_NCC(win=7, scale=2)
    loss_smooth = smoothloss
    loss_Jdet = neg_Jdet_loss

    transform = SpatialTransform_unit().to(device)
    transform_nearest = SpatialTransformNearest_unit().to(device)

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True

    grid = generate_grid(imgshape)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).to(device).float()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    model_dir = "/home/iml/fryderyk.koegl/code/LapIRN-koegl/Model"

    if not os.path.isdir(model_dir):
        Path(model_dir).mkdir(parents=True, exist_ok=True)

    lossall = np.zeros((4, config.iteration_lvl3 + 1))

    step = 0
    load_model = False
    if load_model is True:
        model_path = "../Model/LDR_OASIS_NCC_unit_add_reg_3_anti_1_stagelvl3_10000.pth"
        print("Loading weight: ", model_path)
        step = 10000
        model.load_state_dict(torch.load(model_path))
        temp_lossall = np.load(
            "../Model/lossLDR_OASIS_NCC_unit_add_reg_3_anti_1_stagelvl3_10000.npy"
        )
        lossall[:, 0:10000] = temp_lossall[:, 0:10000]

    while step <= config.iteration_lvl3:
        for X, Y, X_lbl_ct, X_lbl_pet, Y_lbl_ct, Y_lbl_pet in training_generator:
            X = X.to(device).float()
            Y = Y.to(device).float()

            # output_disp_e0, warpped_inputx_lvl1_out, y, compose_field_e0_lvl2_compose, lvl1_v, compose_lvl2_v, e0
            F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, F_xy_lvl2, _ = model(X, Y)

            if step % 1000 == 0 or step == config.iteration_lvl3:
                ct = X_Y[:, 0:1, :, :, :]
                pet = X_Y[:, 1:2, :, :, :]
                save_volume(
                    ct,
                    Path(model_dir) / "warped",
                    step,
                    Path(DATA_PATH) / "fixed_ct.nii.gz",
                    "warped_ct_lvl3",
                )
                save_volume(
                    pet,
                    Path(model_dir) / "warped",
                    step,
                    Path(DATA_PATH) / "fixed_ct.nii.gz",
                    "warped_pet_lvl3",
                )

            # 3 level deep supervision NCC
            loss_ncc_ct = loss_similarity_ct(X_Y[:, 0:1, ...], Y_4x[:, 0:1, ...])
            loss_ncc_pet = loss_similarity_pet(X_Y[:, 1:2, ...], Y_4x[:, 1:2, ...])

            loss_multiNCC = W_CT * loss_ncc_ct + W_PET * loss_ncc_pet

            F_X_Y_norm = transform_unit_flow_to_flow_cuda(
                F_X_Y.permute(0, 2, 3, 4, 1).clone()
            )

            loss_Jacobian = loss_Jdet(F_X_Y_norm, grid)

            # reg2 - use velocity
            _, _, x, y, z = F_xy.shape
            F_xy[:, 0, :, :, :] = F_xy[:, 0, :, :, :] * (z - 1)
            F_xy[:, 1, :, :, :] = F_xy[:, 1, :, :, :] * (y - 1)
            F_xy[:, 2, :, :, :] = F_xy[:, 2, :, :, :] * (x - 1)
            loss_regulation = loss_smooth(F_xy)

            loss_dice_ct = dice_loss_with_grad(
                X_lbl_ct.to(device), Y_lbl_ct.to(device), F_X_Y, model.grid_1, transform
            )
            loss_dice_pet = dice_loss_with_grad(
                X_lbl_pet.to(device),
                Y_lbl_pet.to(device),
                F_X_Y,
                model.grid_1,
                transform,
            )

            # --- MTV / TLG / masked Jacobian losses (lvl3 only) ---

            # warped PET mask (differentiable, bilinear)
            moving_pet_mask = (X_lbl_pet.to(device) == 1).float()
            warped_pet_mask = warp_binary_mask(
                moving_pet_mask, F_X_Y, model.grid_1, transform
            )

            # warped PET image — already in X_Y channel 1, but we need the moving PET image warped
            # X_Y[:, 1:2] is already the warped moving PET image from the network forward pass
            warped_pet_image = X_Y[:, 1:2]
            moving_pet_image = X[:, 1:2]

            loss_mtv = mtv_bias_loss(warped_pet_mask, moving_pet_mask)
            loss_tlg = tlg_bias_loss(
                warped_pet_image, warped_pet_mask, moving_pet_image, moving_pet_mask
            )

            # masked Jacobian det loss — enforce det(J)=1 inside tumor
            jac_det = jacobian_determinant(F_X_Y_norm)  # (B, 1, D, H, W)
            loss_masked_jac = masked_jac_det_loss(jac_det, moving_pet_mask)
            ndv = compute_ndv(jac_det)

            # update total loss
            loss = (
                loss_multiNCC
                + ANTIFOLD * loss_Jacobian
                + SMOOTH * loss_regulation
                + W_DICE_CT * loss_dice_ct
                + W_DICE_PET * loss_dice_pet
                + W_MTV * loss_mtv
                + W_TLG * loss_tlg
                + W_MASKED_JAC * loss_masked_jac
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            lossall[:, step] = np.array(
                [
                    loss.item(),
                    loss_multiNCC.item(),
                    loss_Jacobian.item(),
                    loss_regulation.item(),
                ]
            )
            sys.stdout.write(
                "\r"
                + 'step "{0}" -> training loss "{1:.4f}" - sim_NCC "{2:4f}" - dice_ct "{3:4f}" - dice_pet "{4:4f}" - Jdet "{5:.10f}" -smo "{6:.4f}"'.format(
                    step,
                    loss.item(),
                    loss_multiNCC.item(),
                    loss_dice_ct.item(),
                    loss_dice_pet.item(),
                    loss_Jacobian.item(),
                    loss_regulation.item(),
                )
            )
            sys.stdout.flush()
            mlflow.log_metrics(
                {
                    "lvl3/loss": loss.item(),
                    "lvl3/ncc_ct": loss_ncc_ct.item(),
                    "lvl3/ncc_pet": loss_ncc_pet.item(),
                    "lvl3/smooth": loss_regulation.item(),
                    "lvl3/dice_ct": loss_dice_ct.item(),
                    "lvl3/dice_pet": loss_dice_pet.item(),
                    "lvl3/mtv_bias": loss_mtv.item(),
                    "lvl3/tlg_bias": loss_tlg.item(),
                    "lvl3/masked_jac": loss_masked_jac.item(),
                    "lvl3/ndv": ndv,
                },
                step=step,
            )

            if step == config.iteration_lvl3:
                modelname = (
                    model_dir + "/" + MODEL_NAME + "stagelvl3_" + str(step) + ".pth"
                )
                torch.save(model.state_dict(), modelname)
                np.save(
                    model_dir
                    + "/loss"
                    + MODEL_NAME
                    + "stagelvl3_"
                    + str(step)
                    + ".npy",
                    lossall,
                )

            if step == FREEZE_STEP:
                model.unfreeze_modellvl2()

            step += 1

            if step > config.iteration_lvl3:
                break
        print("one epoch pass")
    # np.save(model_dir + "/loss" + model_name + "stagelvl3.npy", lossall)


def main() -> None:

    config = TrainingConfig()

    mlflow.set_tracking_uri("sqlite:////home/iml/fryderyk.koegl/code/mlruns.db")
    mlflow.set_experiment("PSMAReg_LapIRN")
    with mlflow.start_run():
        train_ids, val_ids = my_data.get_train_val_split(
            data_dir=config.data_dir,
            split_path=config.split_path,
            val_fraction=config.val_fraction,
        )
        config_to_log = config.to_mlflow_params()
        config_to_log["train_indices"] = train_ids
        config_to_log["val_indices"] = val_ids
        mlflow.log_params(config.to_mlflow_params())
        mlflow.log_text(
            json.dumps(config.to_mlflow_params(), indent=2),
            artifact_file="config.json",
        )

        train_dataset = my_data.PSMARegDataset(
            data_dir=config.data_dir,
            case_ids=train_ids,
            use_cache=config.use_cache,
            # overfit="0049",
        )
        val_dataset = my_data.PSMARegDataset(
            data_dir=config.data_dir,
            case_ids=val_ids,
            use_cache=config.use_cache,
            # overfit="0049",
        )

        train_generator = torch_data.DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            num_workers=config.num_workers,
        )
        valid_generator = torch_data.DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )

        path_model_level1 = level1.train_lvl1(config, train_generator, valid_generator)
        # path_model_level2 = train_lvl2(path_model_level1)
        # train_lvl3(path_model_level2)


if __name__ == "__main__":
    main()
