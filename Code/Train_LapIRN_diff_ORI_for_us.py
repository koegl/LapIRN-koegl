"""
Minimal adaptation of Train_LapIRN_diff.py for CT-only overfit on a single
baseline/follow-up pair from PSMAReg.

Changes from original:
- Replaced Dataset_epoch + glob with direct loading of two CT NIfTI files
- Updated imgshape to (192, 192, 288)
- Replaced argparse with hardcoded variables
- Added MLflow logging
- Replaced sys.stdout.write with tqdm
- Added gradient clipping

Everything else (loss structure, model instantiation, freeze/unfreeze logic,
level curriculum) is identical to the original.
"""

from pathlib import Path

import mlflow
import nibabel as nib
import numpy as np
import torch
import torch.utils.data as Data
from Functions import generate_grid, transform_unit_flow_to_flow_cuda
from miccai2020_model_stage import (
    NCC,
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


class SinglePairDataset(Data.Dataset):
    """Loads a single fixed/moving CT pair and returns it repeatedly."""

    def __init__(self, fixed_path: Path, moving_path: Path) -> None:
        def load(path: Path) -> torch.Tensor:
            vol = nib.load(path).get_fdata().astype(np.float32)
            min_v, max_v = vol.min(), vol.max()
            vol = (vol - min_v) / (max_v - min_v + 1e-8)
            # shape: (1, H, W, D)
            return torch.from_numpy(vol).unsqueeze(0)

        # x = moving, y = fixed (LapIRN convention: warps x toward y)
        self.x = load(moving_path)
        self.y = load(fixed_path)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, _: int):
        return self.x, self.y


# ---------------------------------------------------------------------------
# Training functions — identical structure to original
# ---------------------------------------------------------------------------


def train_lvl1(
    training_generator,
    imgshape_4,
    start_channel,
    range_flow,
    lr,
    antifold,
    smooth,
    iteration_lvl1,
    n_checkpoint,
    model_dir,
):
    print("Training lvl1...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Miccai2020_LDR_laplacian_unit_add_lvl1(
        2, 3, start_channel, is_train=True, imgshape=imgshape_4, range_flow=range_flow
    ).to(device)

    loss_similarity = NCC(win=7)
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
        for X, Y in training_generator:
            X = X.to(device).float()
            Y = Y.to(device).float()

            F_X_Y, X_Y, Y_4x, F_xy, _ = model(X, Y)

            loss_multiNCC = loss_similarity(X_Y, Y_4x)

            F_X_Y_norm = transform_unit_flow_to_flow_cuda(
                F_X_Y.permute(0, 2, 3, 4, 1).clone()
            )
            loss_Jacobian = loss_Jdet(F_X_Y_norm, grid_4)

            _, _, x, y, z = F_xy.shape
            F_xy[:, 0, :, :, :] = F_xy[:, 0, :, :, :] * (z - 1)
            F_xy[:, 1, :, :, :] = F_xy[:, 1, :, :, :] * (y - 1)
            F_xy[:, 2, :, :, :] = F_xy[:, 2, :, :, :] * (x - 1)
            loss_regulation = smoothloss(F_xy)

            loss = loss_multiNCC + antifold * loss_Jacobian + smooth * loss_regulation

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            mlflow.log_metrics(
                {
                    "lvl1/loss": loss.item(),
                    "lvl1/ncc": loss_multiNCC.item(),
                    "lvl1/jacobian": loss_Jacobian.item(),
                    "lvl1/smooth": loss_regulation.item(),
                },
                step=step,
            )

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                ncc=f"{loss_multiNCC.item():.4f}",
                smooth=f"{loss_regulation.item():.4f}",
            )
            pbar.update(1)

            if step % n_checkpoint == 0:
                torch.save(model.state_dict(), model_dir / f"stagelvl1_{step:05d}.pth")

            step += 1
            if step > iteration_lvl1:
                break

    pbar.close()
    torch.save(model.state_dict(), model_dir / f"stagelvl1_{step:05d}.pth")


def train_lvl2(
    training_generator,
    imgshape_4,
    imgshape_2,
    start_channel,
    range_flow,
    lr,
    antifold,
    smooth,
    freeze_step,
    iteration_lvl2,
    n_checkpoint,
    model_dir,
):
    print("Training lvl2...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_lvl1 = Miccai2020_LDR_laplacian_unit_add_lvl1(
        2, 3, start_channel, is_train=True, imgshape=imgshape_4, range_flow=range_flow
    ).to(device)
    model_path = sorted(model_dir.glob("stagelvl1_?????.pth"))[-1]
    model_lvl1.load_state_dict(torch.load(model_path))
    tqdm.write(f"Loaded lvl1: {model_path}")
    for param in model_lvl1.parameters():
        param.requires_grad = False

    model = Miccai2020_LDR_laplacian_unit_add_lvl2(
        2,
        3,
        start_channel,
        is_train=True,
        imgshape=imgshape_2,
        range_flow=range_flow,
        model_lvl1=model_lvl1,
    ).to(device)

    loss_similarity = multi_resolution_NCC(win=5, scale=2)
    loss_Jdet = neg_Jdet_loss

    grid_2 = generate_grid(imgshape_2)
    grid_2 = (
        torch.from_numpy(np.reshape(grid_2, (1,) + grid_2.shape)).to(device).float()
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    step = 0
    pbar = tqdm(total=iteration_lvl2, desc="lvl2")

    while step <= iteration_lvl2:
        for X, Y in training_generator:
            X = X.to(device).float()
            Y = Y.to(device).float()

            F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, _ = model(X, Y)

            loss_multiNCC = loss_similarity(X_Y, Y_4x)

            F_X_Y_norm = transform_unit_flow_to_flow_cuda(
                F_X_Y.permute(0, 2, 3, 4, 1).clone()
            )
            loss_Jacobian = loss_Jdet(F_X_Y_norm, grid_2)

            _, _, x, y, z = F_xy.shape
            F_xy[:, 0, :, :, :] = F_xy[:, 0, :, :, :] * (z - 1)
            F_xy[:, 1, :, :, :] = F_xy[:, 1, :, :, :] * (y - 1)
            F_xy[:, 2, :, :, :] = F_xy[:, 2, :, :, :] * (x - 1)
            loss_regulation = smoothloss(F_xy)

            loss = loss_multiNCC + antifold * loss_Jacobian + smooth * loss_regulation

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            mlflow.log_metrics(
                {
                    "lvl2/loss": loss.item(),
                    "lvl2/ncc": loss_multiNCC.item(),
                    "lvl2/jacobian": loss_Jacobian.item(),
                    "lvl2/smooth": loss_regulation.item(),
                },
                step=step,
            )

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                ncc=f"{loss_multiNCC.item():.4f}",
            )
            pbar.update(1)

            if step % n_checkpoint == 0:
                torch.save(model.state_dict(), model_dir / f"stagelvl2_{step:05d}.pth")

            if step == freeze_step:
                model.unfreeze_modellvl1()
                tqdm.write(f"Unfroze lvl1 at step {step}")

            step += 1
            if step > iteration_lvl2:
                break

    pbar.close()
    torch.save(model.state_dict(), model_dir / f"stagelvl2_{step:05d}.pth")


def train_lvl3(
    training_generator,
    imgshape_4,
    imgshape_2,
    imgshape,
    start_channel,
    range_flow,
    lr,
    antifold,
    smooth,
    freeze_step,
    iteration_lvl3,
    n_checkpoint,
    model_dir,
    fixed_ct_path,
    ckpt_dir,
):
    print("Training lvl3...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_lvl1 = Miccai2020_LDR_laplacian_unit_add_lvl1(
        2, 3, start_channel, is_train=True, imgshape=imgshape_4, range_flow=range_flow
    ).to(device)
    model_lvl2 = Miccai2020_LDR_laplacian_unit_add_lvl2(
        2,
        3,
        start_channel,
        is_train=True,
        imgshape=imgshape_2,
        range_flow=range_flow,
        model_lvl1=model_lvl1,
    ).to(device)
    model_path = sorted(model_dir.glob("stagelvl2_?????.pth"))[-1]
    model_lvl2.load_state_dict(torch.load(model_path))
    tqdm.write(f"Loaded lvl2: {model_path}")
    for param in model_lvl2.parameters():
        param.requires_grad = False

    model = Miccai2020_LDR_laplacian_unit_add_lvl3(
        2,
        3,
        start_channel,
        is_train=True,
        imgshape=imgshape,
        range_flow=range_flow,
        model_lvl2=model_lvl2,
    ).to(device)

    loss_similarity = multi_resolution_NCC(win=7, scale=3)
    loss_Jdet = neg_Jdet_loss

    grid = generate_grid(imgshape)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).to(device).float()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # fixed image affine for saving warped volumes in correct world space
    fixed_affine = nib.load(fixed_ct_path).affine
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    pbar = tqdm(total=iteration_lvl3, desc="lvl3")

    while step <= iteration_lvl3:
        for X, Y in training_generator:
            X = X.to(device).float()
            Y = Y.to(device).float()

            F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, F_xy_lvl2, _ = model(X, Y)

            loss_multiNCC = loss_similarity(X_Y, Y_4x)

            F_X_Y_norm = transform_unit_flow_to_flow_cuda(
                F_X_Y.permute(0, 2, 3, 4, 1).clone()
            )
            loss_Jacobian = loss_Jdet(F_X_Y_norm, grid)

            _, _, x, y, z = F_xy.shape
            F_xy[:, 0, :, :, :] = F_xy[:, 0, :, :, :] * (z - 1)
            F_xy[:, 1, :, :, :] = F_xy[:, 1, :, :, :] * (y - 1)
            F_xy[:, 2, :, :, :] = F_xy[:, 2, :, :, :] * (x - 1)
            loss_regulation = smoothloss(F_xy)

            loss = loss_multiNCC + antifold * loss_Jacobian + smooth * loss_regulation

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            mlflow.log_metrics(
                {
                    "lvl3/loss": loss.item(),
                    "lvl3/ncc": loss_multiNCC.item(),
                    "lvl3/jacobian": loss_Jacobian.item(),
                    "lvl3/smooth": loss_regulation.item(),
                },
                step=step,
            )

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                ncc=f"{loss_multiNCC.item():.4f}",
            )
            pbar.update(1)

            if step % n_checkpoint == 0:
                torch.save(model.state_dict(), model_dir / f"stagelvl3_{step:05d}.pth")
                warped_ct = X_Y[0, 0].detach().cpu().numpy()
                nib.save(
                    nib.Nifti1Image(warped_ct, fixed_affine),
                    str(ckpt_dir / f"warped_ct_step_{step:05d}.nii.gz"),
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

if __name__ == "__main__":
    # --- Paths ---------------------------------------------------------------
    FIXED_CT_PATH = Path(
        "/home/iml/fryderyk.koegl/data/PSMAReg_dataset/imagesTr/PSMARegPSMA_0006_0000_00.nii.gz"
    )
    MOVING_CT_PATH = Path(
        "/home/iml/fryderyk.koegl/data/PSMAReg_dataset/imagesTr/PSMARegPSMA_0006_0000_01.nii.gz"
    )
    MODEL_DIR = Path("./checkpoints/overfit_ct/stages")
    CKPT_DIR = Path("./checkpoints/overfit_ct/warped")

    # --- Shapes --------------------------------------------------------------
    imgshape = (192, 192, 288)
    imgshape_2 = (imgshape[0] // 2, imgshape[1] // 2, imgshape[2] // 2)
    imgshape_4 = (imgshape[0] // 4, imgshape[1] // 4, imgshape[2] // 4)

    # --- Hyperparameters -----------------------------------------------------
    range_flow = 0.4
    lr = 1e-4
    start_channel = 7
    antifold = 0.0
    smooth = 3.5
    freeze_step = 1
    n_checkpoint = 1000
    iteration_lvl1 = 10000
    iteration_lvl2 = 10000
    iteration_lvl3 = 20000

    # --- Dataloader ----------------------------------------------------------
    dataset = SinglePairDataset(
        fixed_path=FIXED_CT_PATH,
        moving_path=MOVING_CT_PATH,
    )
    training_generator = Data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0
    )

    # --- MLflow --------------------------------------------------------------
    mlflow.set_tracking_uri("sqlite:////home/iml/fryderyk.koegl/code/psmareg/mlruns.db")
    mlflow.set_experiment("PSMAReg_LapIRN_overfit_CT_only")

    with mlflow.start_run():
        mlflow.log_params(
            {
                "imgshape": str(imgshape),
                "imgshape_2": str(imgshape_2),
                "imgshape_4": str(imgshape_4),
                "range_flow": range_flow,
                "lr": lr,
                "start_channel": start_channel,
                "antifold": antifold,
                "smooth": smooth,
                "freeze_step": freeze_step,
                "iteration_lvl1": iteration_lvl1,
                "iteration_lvl2": iteration_lvl2,
                "iteration_lvl3": iteration_lvl3,
            }
        )

        train_lvl1(
            training_generator,
            imgshape_4,
            start_channel,
            range_flow,
            lr,
            antifold,
            smooth,
            iteration_lvl1,
            n_checkpoint,
            MODEL_DIR,
        )
        train_lvl2(
            training_generator,
            imgshape_4,
            imgshape_2,
            start_channel,
            range_flow,
            lr,
            antifold,
            smooth,
            freeze_step,
            iteration_lvl2,
            n_checkpoint,
            MODEL_DIR,
        )
        train_lvl3(
            training_generator,
            imgshape_4,
            imgshape_2,
            imgshape,
            start_channel,
            range_flow,
            lr,
            antifold,
            smooth,
            freeze_step,
            iteration_lvl3,
            n_checkpoint,
            MODEL_DIR,
            FIXED_CT_PATH,
            CKPT_DIR,
        )
