"""
PSMAReg LapIRN training script — overfit on a single baseline/follow-up pair.

Adapts Train_LapIRN_diff.py for 4-channel multimodal input (CT+PET moving,
CT+PET fixed) using naive concatenation. Preserves the original 3-level
sequential training curriculum from LapIRN.

Input convention (matches miccai2020_model_stage.py):
    x = moving image  (B, 2, H, W, D)  — channels: [CT_moving, PET_moving]
    y = fixed image   (B, 2, H, W, D)  — channels: [CT_fixed,  PET_fixed]
    The model internally does torch.cat((x, y), 1) → (B, 4, H, W, D)
    so in_channel=2 for each of x and y (total 4 after cat inside model).

Loss:
    multi_resolution_NCC computed separately on CT and PET channels,
    weighted and summed. Smoothness and Jacobian losses on DVF unchanged.
"""

import json
from pathlib import Path
from typing import Tuple

import mlflow
import nibabel as nib
import numpy as np
import torch
import torch.utils.data as Data
from config import TrainingConfig
from Functions import generate_grid, transform_unit_flow_to_flow_cuda
from miccai2020_model_stage import (
    Miccai2020_LDR_laplacian_unit_add_lvl1,
    Miccai2020_LDR_laplacian_unit_add_lvl2,
    Miccai2020_LDR_laplacian_unit_add_lvl3,
    multi_resolution_NCC,
    neg_Jdet_loss,
    smoothloss,
)
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class PSMARegSinglePairDataset(Data.Dataset):
    """
    Dataset returning a single fixed/moving CT+PET pair repeatedly.
    Used for overfitting validation of the training pipeline.

    Args:
        fixed_ct_path: Path to fixed CT NIfTI file.
        fixed_pet_path: Path to fixed PET NIfTI file.
        moving_ct_path: Path to moving CT NIfTI file.
        moving_pet_path: Path to moving PET NIfTI file.
    """

    def __init__(
        self,
        fixed_ct_path: Path,
        fixed_pet_path: Path,
        moving_ct_path: Path,
        moving_pet_path: Path,
    ) -> None:
        def load_and_norm(path: Path) -> torch.Tensor:
            vol = nib.load(path).get_fdata().astype(np.float32)
            min_v = vol.min()
            max_v = vol.max()
            vol = (vol - min_v) / (max_v - min_v + 1e-8)
            return torch.from_numpy(vol).unsqueeze(0)  # (1, H, W, D)

        self.fixed_ct = load_and_norm(fixed_ct_path)
        self.fixed_pet = load_and_norm(fixed_pet_path)
        self.moving_ct = load_and_norm(moving_ct_path)
        self.moving_pet = load_and_norm(moving_pet_path)

        # x = moving, y = fixed — each (2, H, W, D)
        # self.x = torch.cat([self.moving_ct, self.moving_pet], dim=0)
        # self.y = torch.cat([self.fixed_ct, self.fixed_pet], dim=0)

        self.x = self.moving_ct
        self.y = self.fixed_ct

    def __len__(self) -> int:
        return 1

    def __getitem__(self, _: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x, self.y


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def save_warped_volumes(
    warped_x: torch.Tensor,
    out_dir: Path,
    step: int,
    fixed_ct_path: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fixed_nib = nib.load(fixed_ct_path)
    affine = fixed_nib.affine

    warped_ct = warped_x[0, 0].detach().cpu().numpy()
    warped_pet = warped_x[0, 1].detach().cpu().numpy()

    nib.save(
        nib.Nifti1Image(warped_ct, affine),
        str(out_dir / f"warped_ct_step_{step:05d}.nii.gz"),
    )
    nib.save(
        nib.Nifti1Image(warped_pet, affine),
        str(out_dir / f"warped_pet_step_{step:05d}.nii.gz"),
    )


# ---------------------------------------------------------------------------
# Training levels
# ---------------------------------------------------------------------------


def train_lvl1(
    training_generator: Data.DataLoader,
    imgshape_4: Tuple[int, int, int],
    imgshape: Tuple[int, int, int],
    in_channel: int,
    n_classes: int,
    start_channel: int,
    range_flow: float,
    lr: float,
    antifold: float,
    smooth: float,
    iteration_lvl1: int,
    n_checkpoint: int,
    model_dir: Path,
    w_ct: float,
    w_pet: float,
    lvl1_ncc_win: int,
    lvl1_ncc_scale: int,
) -> None:
    """Train LapIRN level 1 (coarsest scale)."""
    print("Training lvl1...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Miccai2020_LDR_laplacian_unit_add_lvl1(
        in_channel=in_channel,
        n_classes=n_classes,
        start_channel=start_channel,
        is_train=True,
        imgshape=imgshape_4,
        range_flow=range_flow,
    ).to(device)

    loss_ncc_ct = multi_resolution_NCC(win=lvl1_ncc_win, scale=lvl1_ncc_scale)
    # loss_ncc_pet = multi_resolution_NCC(win=lvl1_ncc_win, scale=lvl1_ncc_scale)
    loss_Jdet = neg_Jdet_loss

    grid_4 = generate_grid(imgshape_4)
    grid_4 = (
        torch.from_numpy(np.reshape(grid_4, (1,) + grid_4.shape)).to(device).float()
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    pbar = tqdm(total=iteration_lvl1, desc="lvl1")

    while step <= iteration_lvl1:
        for x, y in training_generator:
            x = x.to(device).float()  # (B, 2, H, W, D) moving
            y = y.to(device).float()  # (B, 2, H, W, D) fixed

            # returns: output_disp_e0, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, e0
            F_X_Y, warped_x, down_y, output_disp_e0_v, _ = model(x, y)

            # warped_x is (B, 2, H, W, D) — split channels
            warped_ct = warped_x[:, 0:1, ...]
            # warped_pet = warped_x[:, 1:2, ...]

            # down_y is fixed downsampled — channel 1 of cat(x,y) downsampled 4x
            # cat(x,y) = [mov_ct, mov_pet, fix_ct, fix_pet], so index 2=fix_ct, 3=fix_pet
            # but down_y is extracted as cat_input_lvl1[:, 1:2] inside model (single ch)
            # We recompute fixed downsampled for loss manually:
            down_avg = torch.nn.AvgPool3d(
                kernel_size=3, stride=2, padding=1, count_include_pad=False
            )
            y_4x = down_avg(down_avg(y))
            fixed_ct_4x = y_4x[:, 0:1, ...]
            fixed_pet_4x = y_4x[:, 1:2, ...]

            ncc_ct = loss_ncc_ct(warped_ct, fixed_ct_4x)
            # ncc_pet = loss_ncc_pet(warped_pet, fixed_pet_4x)
            loss_multiNCC = w_ct * ncc_ct  # + w_pet * ncc_pet

            F_X_Y_norm = transform_unit_flow_to_flow_cuda(
                F_X_Y.permute(0, 2, 3, 4, 1).clone()
            )
            loss_Jacobian = loss_Jdet(F_X_Y_norm, grid_4)

            _, _, x_s, y_s, z_s = output_disp_e0_v.shape
            v = output_disp_e0_v.clone()
            v[:, 0] = v[:, 0] * (z_s - 1)
            v[:, 1] = v[:, 1] * (y_s - 1)
            v[:, 2] = v[:, 2] * (x_s - 1)
            loss_regulation = smoothloss(v)

            loss = loss_multiNCC + antifold * loss_Jacobian + smooth * loss_regulation

            optimizer.zero_grad()

            if torch.isnan(loss):
                tqdm.write(
                    f"NaN loss at step {step}: "
                    f"ncc_ct={ncc_ct.item():.6f} "
                    # f"ncc_pet={ncc_pet.item():.6f} "
                    f"smooth={loss_regulation.item():.6f} "
                    f"jac={loss_Jacobian.item():.6f} "
                    f"warped_ct_min={warped_ct.min().item():.4f} "
                    f"warped_ct_max={warped_ct.max().item():.4f} "
                    # f"warped_pet_min={warped_pet.min().item():.4f} "
                    # f"warped_pet_max={warped_pet.max().item():.4f}"
                )
                break
            loss.backward()
            optimizer.step()

            mlflow.log_metrics(
                {
                    "lvl1/loss": loss.item(),
                    "lvl1/ncc_ct": ncc_ct.item(),
                    # "lvl1/ncc_pet": ncc_pet.item(),
                    "lvl1/ncc": loss_multiNCC.item(),
                    "lvl1/jacobian": loss_Jacobian.item(),
                    "lvl1/smooth": loss_regulation.item(),
                },
                step=step,
            )

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                ncc_ct=f"{ncc_ct.item():.4f}",
                # ncc_pet=f"{ncc_pet.item():.4f}",
            )
            pbar.update(1)

            if step % n_checkpoint == 0:
                torch.save(
                    model.state_dict(),
                    model_dir / f"stagelvl1_{step:05d}.pth",
                )

            step += 1
            if step > iteration_lvl1:
                break

    pbar.close()
    torch.save(model.state_dict(), model_dir / f"stagelvl1_{step:05d}.pth")


def train_lvl2(
    training_generator: Data.DataLoader,
    imgshape_4: Tuple[int, int, int],
    imgshape_2: Tuple[int, int, int],
    imgshape: Tuple[int, int, int],
    in_channel: int,
    n_classes: int,
    start_channel: int,
    range_flow: float,
    lr: float,
    antifold: float,
    smooth: float,
    freeze_step: int,
    iteration_lvl2: int,
    n_checkpoint: int,
    model_dir: Path,
    w_ct: float,
    w_pet: float,
    lvl2_ncc_win: int,
    lvl2_ncc_scale: int,
) -> None:
    """Train LapIRN level 2 (medium scale), with frozen lvl1 initially."""
    print("Training lvl2...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_lvl1 = Miccai2020_LDR_laplacian_unit_add_lvl1(
        in_channel=in_channel,
        n_classes=n_classes,
        start_channel=start_channel,
        is_train=True,
        imgshape=imgshape_4,
        range_flow=range_flow,
    ).to(device)

    model_path = sorted((model_dir).glob("stagelvl1_?????.pth"))[-1]
    model_lvl1.load_state_dict(torch.load(model_path))
    tqdm.write(f"Loaded lvl1 weights: {model_path}")

    for param in model_lvl1.parameters():
        param.requires_grad = False

    model = Miccai2020_LDR_laplacian_unit_add_lvl2(
        in_channel=in_channel,
        n_classes=n_classes,
        start_channel=start_channel,
        is_train=True,
        imgshape=imgshape_2,
        range_flow=range_flow,
        model_lvl1=model_lvl1,
    ).to(device)

    loss_ncc_ct = multi_resolution_NCC(win=lvl2_ncc_win, scale=lvl2_ncc_scale)
    # loss_ncc_pet = multi_resolution_NCC(win=lvl2_ncc_win, scale=lvl2_ncc_scale)
    loss_Jdet = neg_Jdet_loss

    grid_2 = generate_grid(imgshape_2)
    grid_2 = (
        torch.from_numpy(np.reshape(grid_2, (1,) + grid_2.shape)).to(device).float()
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model_dir.mkdir(parents=True, exist_ok=True)

    down_avg = torch.nn.AvgPool3d(
        kernel_size=3, stride=2, padding=1, count_include_pad=False
    )

    step = 0
    pbar = tqdm(total=iteration_lvl2, desc="lvl2")

    while step <= iteration_lvl2:
        for x, y in training_generator:
            x = x.to(device).float()
            y = y.to(device).float()

            # returns: output_disp_e0, warpped_inputx_lvl1_out, y_down, compose_field_e0_lvl1v, lvl1_v, e0
            F_X_Y, warped_x, y_down, compose_field_e0_lvl1v, _, _ = model(x, y)

            warped_ct = warped_x[:, 0:1, ...]
            warped_pet = warped_x[:, 1:2, ...]

            y_2x = down_avg(y)
            fixed_ct_2x = y_2x[:, 0:1, ...]
            fixed_pet_2x = y_2x[:, 1:2, ...]

            ncc_ct = loss_ncc_ct(warped_ct, fixed_ct_2x)
            # ncc_pet = loss_ncc_pet(warped_pet, fixed_pet_2x)
            loss_multiNCC = w_ct * ncc_ct  # + w_pet * ncc_pet

            F_X_Y_norm = transform_unit_flow_to_flow_cuda(
                F_X_Y.permute(0, 2, 3, 4, 1).clone()
            )
            loss_Jacobian = loss_Jdet(F_X_Y_norm, grid_2)

            _, _, x_s, y_s, z_s = compose_field_e0_lvl1v.shape
            v = compose_field_e0_lvl1v.clone()
            v[:, 0] = v[:, 0] * (z_s - 1)
            v[:, 1] = v[:, 1] * (y_s - 1)
            v[:, 2] = v[:, 2] * (x_s - 1)
            loss_regulation = smoothloss(v)

            loss = loss_multiNCC + antifold * loss_Jacobian + smooth * loss_regulation

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mlflow.log_metrics(
                {
                    "lvl2/loss": loss.item(),
                    "lvl2/ncc_ct": ncc_ct.item(),
                    "lvl2/ncc_pet": ncc_pet.item(),
                    "lvl2/ncc": loss_multiNCC.item(),
                    "lvl2/jacobian": loss_Jacobian.item(),
                    "lvl2/smooth": loss_regulation.item(),
                },
                step=step,
            )

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                ncc_ct=f"{ncc_ct.item():.4f}",
                ncc_pet=f"{ncc_pet.item():.4f}",
            )
            pbar.update(1)

            if step % n_checkpoint == 0:
                torch.save(
                    model.state_dict(),
                    model_dir / f"stagelvl2_{step:05d}.pth",
                )

            if step == freeze_step:
                model.unfreeze_modellvl1()
                tqdm.write(f"Unfroze lvl1 at step {step}")

            step += 1
            if step > iteration_lvl2:
                break

    pbar.close()
    torch.save(model.state_dict(), model_dir / f"stagelvl2_{step:05d}.pth")


def train_lvl3(
    training_generator: Data.DataLoader,
    imgshape_4: Tuple[int, int, int],
    imgshape_2: Tuple[int, int, int],
    imgshape: Tuple[int, int, int],
    in_channel: int,
    n_classes: int,
    start_channel: int,
    range_flow: float,
    lr: float,
    antifold: float,
    smooth: float,
    freeze_step: int,
    iteration_lvl3: int,
    n_checkpoint: int,
    model_dir: Path,
    ckpt_dir: Path,
    fixed_ct_path: Path,
    w_ct: float,
    w_pet: float,
    lvl3_ncc_win: int,
    lvl3_ncc_scale: int,
) -> None:
    """Train LapIRN level 3 (full resolution), with frozen lvl2 initially."""
    print("Training lvl3...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_lvl1 = Miccai2020_LDR_laplacian_unit_add_lvl1(
        in_channel=in_channel,
        n_classes=n_classes,
        start_channel=start_channel,
        is_train=True,
        imgshape=imgshape_4,
        range_flow=range_flow,
    ).to(device)

    model_lvl2 = Miccai2020_LDR_laplacian_unit_add_lvl2(
        in_channel=in_channel,
        n_classes=n_classes,
        start_channel=start_channel,
        is_train=True,
        imgshape=imgshape_2,
        range_flow=range_flow,
        model_lvl1=model_lvl1,
    ).to(device)

    model_path = sorted((model_dir).glob("stagelvl2_?????.pth"))[-1]
    model_lvl2.load_state_dict(torch.load(model_path))
    tqdm.write(f"Loaded lvl2 weights: {model_path}")

    for param in model_lvl2.parameters():
        param.requires_grad = False

    model = Miccai2020_LDR_laplacian_unit_add_lvl3(
        in_channel=in_channel,
        n_classes=n_classes,
        start_channel=start_channel,
        is_train=True,
        imgshape=imgshape,
        range_flow=range_flow,
        model_lvl2=model_lvl2,
    ).to(device)

    loss_ncc_ct = multi_resolution_NCC(win=lvl3_ncc_win, scale=lvl3_ncc_scale)
    # loss_ncc_pet = multi_resolution_NCC(win=lvl3_ncc_win, scale=lvl3_ncc_scale)
    loss_Jdet = neg_Jdet_loss

    grid = generate_grid(imgshape)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).to(device).float()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    step = 0
    pbar = tqdm(total=iteration_lvl3, desc="lvl3")

    while step <= iteration_lvl3:
        for x, y in training_generator:
            x = x.to(device).float()
            y = y.to(device).float()

            # returns: output_disp_e0, warpped_inputx_lvl1_out, y,
            #          compose_field_e0_lvl2_compose, lvl1_v, compose_lvl2_v, e0
            F_X_Y, warped_x, y_full, compose_field, _, _, _ = model(x, y)

            warped_ct = warped_x[:, 0:1, ...]
            warped_pet = warped_x[:, 1:2, ...]
            fixed_ct = y[:, 0:1, ...]
            fixed_pet = y[:, 1:2, ...]

            ncc_ct = loss_ncc_ct(warped_ct, fixed_ct)
            # ncc_pet = loss_ncc_pet(warped_pet, fixed_pet)
            loss_multiNCC = w_ct * ncc_ct  # + w_pet * ncc_pet

            F_X_Y_norm = transform_unit_flow_to_flow_cuda(
                F_X_Y.permute(0, 2, 3, 4, 1).clone()
            )
            loss_Jacobian = loss_Jdet(F_X_Y_norm, grid)

            _, _, x_s, y_s, z_s = compose_field.shape
            v = compose_field.clone()
            v[:, 0] = v[:, 0] * (z_s - 1)
            v[:, 1] = v[:, 1] * (y_s - 1)
            v[:, 2] = v[:, 2] * (x_s - 1)
            loss_regulation = smoothloss(v)

            loss = loss_multiNCC + antifold * loss_Jacobian + smooth * loss_regulation

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mlflow.log_metrics(
                {
                    "lvl3/loss": loss.item(),
                    "lvl3/ncc_ct": ncc_ct.item(),
                    "lvl3/ncc_pet": ncc_pet.item(),
                    "lvl3/ncc": loss_multiNCC.item(),
                    "lvl3/jacobian": loss_Jacobian.item(),
                    "lvl3/smooth": loss_regulation.item(),
                },
                step=step,
            )

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                ncc_ct=f"{ncc_ct.item():.4f}",
                ncc_pet=f"{ncc_pet.item():.4f}",
            )
            pbar.update(1)

            if step % n_checkpoint == 0:
                torch.save(
                    model.state_dict(),
                    model_dir / f"stagelvl3_{step:05d}.pth",
                )
                save_warped_volumes(
                    warped_x, ckpt_dir, step, fixed_ct_path=fixed_ct_path
                )

            if step == freeze_step:
                model.unfreeze_modellvl2()
                tqdm.write(f"Unfroze lvl2 at step {step}")

            step += 1
            if step > iteration_lvl3:
                break

    pbar.close()
    torch.save(model.state_dict(), model_dir / f"stagelvl3_{step:05d}.pth")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    config = TrainingConfig()

    # --- Dataloader ----------------------------------------------------------
    dataset = PSMARegSinglePairDataset(
        fixed_ct_path=config.fixed_ct_path,
        fixed_pet_path=config.fixed_pet_path,
        moving_ct_path=config.moving_ct_path,
        moving_pet_path=config.moving_pet_path,
    )
    training_generator = Data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
    )

    # --- MLflow --------------------------------------------------------------
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment)

    with mlflow.start_run():
        mlflow.log_params(config.to_mlflow_params())
        mlflow.log_text(
            json.dumps(config.to_mlflow_params(), indent=2),
            artifact_file="config.json",
        )

        if config.train_lvl1:
            train_lvl1(
                training_generator=training_generator,
                imgshape_4=config.img_shape_4,
                imgshape=config.img_shape,
                in_channel=config.in_channel,
                n_classes=config.n_classes,
                start_channel=config.start_channel,
                range_flow=config.range_flow,
                lr=config.lr,
                antifold=config.w_jacobian,
                smooth=config.w_smooth,
                iteration_lvl1=config.epochs_lvl1 * config.iteration_multiplier,
                n_checkpoint=config.n_checkpoint * config.iteration_multiplier,
                model_dir=config.model_dir,
                w_ct=config.w_ct,
                w_pet=config.w_pet,
                lvl1_ncc_win=config.lvl1_ncc_win,
                lvl1_ncc_scale=config.lvl1_ncc_scale,
            )

        if config.train_lvl2:
            train_lvl2(
                training_generator=training_generator,
                imgshape_4=config.img_shape_4,
                imgshape_2=config.img_shape_2,
                imgshape=config.img_shape,
                in_channel=config.in_channel,
                n_classes=config.n_classes,
                start_channel=config.start_channel,
                range_flow=config.range_flow,
                lr=config.lr,
                antifold=config.w_jacobian,
                smooth=config.w_smooth,
                freeze_step=config.unfreeze_epoch_in_lvl2: * config.iteration_multiplier,
                iteration_lvl2=config.epochs_lvl2 * config.iteration_multiplier,
                n_checkpoint=config.n_checkpoint * config.iteration_multiplier,
                model_dir=config.model_dir,
                w_ct=config.w_ct,
                w_pet=config.w_pet,
                lvl2_ncc_win=config.lvl2_ncc_win,
                lvl2_ncc_scale=config.lvl2_ncc_scale,
            )

        if config.train_lvl3:
            train_lvl3(
                training_generator=training_generator,
                imgshape_4=config.img_shape_4,
                imgshape_2=config.img_shape_2,
                imgshape=config.img_shape,
                in_channel=config.in_channel,
                n_classes=config.n_classes,
                start_channel=config.start_channel,
                range_flow=config.range_flow,
                lr=config.lr,
                antifold=config.w_jacobian,
                smooth=config.w_smooth,
                freeze_step=config.unfreeze_epoch_in_lvl2: * config.iteration_multiplier,
                iteration_lvl3=config.epochs_lvl3 * config.iteration_multiplier,
                n_checkpoint=config.n_checkpoint * config.iteration_multiplier,
                model_dir=config.model_dir,
                ckpt_dir=config.ckpt_dir,
                fixed_ct_path=config.fixed_ct_path,
                w_ct=config.w_ct,
                w_pet=config.w_pet,
                lvl3_ncc_win=config.lvl3_ncc_win,
                lvl3_ncc_scale=config.lvl3_ncc_scale,
            )


if __name__ == "__main__":
    main()
