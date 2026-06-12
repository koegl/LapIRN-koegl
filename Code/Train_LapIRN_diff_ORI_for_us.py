import glob
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import mlflow
import nibabel as nib
import numpy as np
import torch
from Functions import generate_grid, transform_unit_flow_to_flow_cuda
from miccai2020_model_stage import (
    NCC,
    Miccai2020_LDR_laplacian_unit_add_lvl1,
    Miccai2020_LDR_laplacian_unit_add_lvl2,
    Miccai2020_LDR_laplacian_unit_add_lvl3,
    SpatialTransform_unit,
    SpatialTransformNearest_unit,
    multi_resolution_NCC,
    neg_Jdet_loss,
    smoothloss,
)

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = ArgumentParser()
parser.add_argument("--lr", type=float, dest="lr", default=1e-4, help="learning rate")
parser.add_argument(
    "--iteration_lvl1",
    type=int,
    dest="iteration_lvl1",
    default=3001,
    help="number of lvl1 iterations",
)
parser.add_argument(
    "--iteration_lvl2",
    type=int,
    dest="iteration_lvl2",
    default=30001,
    help="number of lvl2 iterations",
)
parser.add_argument(
    "--iteration_lvl3",
    type=int,
    dest="iteration_lvl3",
    default=60001,
    help="number of lvl3 iterations",
)
parser.add_argument(
    "--antifold",
    type=float,
    dest="antifold",
    default=0.0,
    help="Anti-fold loss: suggested range 0 to 1000",
)
parser.add_argument(
    "--smooth",
    type=float,
    dest="smooth",
    default=3.5,
    help="Gradient smooth loss: suggested range 0.1 to 10",
)
parser.add_argument(
    "--checkpoint",
    type=int,
    dest="checkpoint",
    default=5000,
    help="frequency of saving models",
)
parser.add_argument(
    "--start_channel",
    type=int,
    dest="start_channel",
    default=7,
    help="number of start channels",
)
parser.add_argument(
    "--datapath",
    type=str,
    dest="datapath",
    default="/PATH/TO/YOUR/DATA",
    help="data path for training images",
)
parser.add_argument(
    "--freeze_step",
    type=int,
    dest="freeze_step",
    default=2000,
    help="Number step for freezing the previous level",
)
opt = parser.parse_args()

lr = opt.lr
start_channel = opt.start_channel
antifold = opt.antifold
n_checkpoint = opt.checkpoint
smooth = opt.smooth
datapath = opt.datapath
freeze_step = opt.freeze_step

iteration_lvl1 = opt.iteration_lvl1
iteration_lvl2 = opt.iteration_lvl2
iteration_lvl3 = opt.iteration_lvl3

model_name = "LDR_OASIS_NCC_unit_add_reg_35_"


def save_volume(
    volume: torch.Tensor, out_dir: Path, step, reference_path: Path, name: str
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fixed_nib = nib.load(reference_path.as_posix())
    affine = fixed_nib.affine

    nib.save(
        nib.Nifti1Image(volume.detach().squeeze().cpu().numpy(), affine),
        str(out_dir / f"{name}_{step:05d}.nii.gz"),
    )


def train_lvl1():
    print("Training lvl1...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Miccai2020_LDR_laplacian_unit_add_lvl1(
        2, 3, start_channel, is_train=True, imgshape=imgshape_4, range_flow=range_flow
    ).to(device)

    loss_similarity = NCC(win=3)
    loss_smooth = smoothloss
    loss_Jdet = neg_Jdet_loss

    transform = SpatialTransform_unit().to(device)

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True

    X_vol = nib.load(datapath + "/moving_ct.nii.gz").get_fdata().astype("float32")
    Y_vol = nib.load(datapath + "/fixed_ct.nii.gz").get_fdata().astype("float32")
    X_vol = (X_vol - X_vol.min()) / (X_vol.max() - X_vol.min() + 1e-8)
    Y_vol = (Y_vol - Y_vol.min()) / (Y_vol.max() - Y_vol.min() + 1e-8)
    X_t = torch.from_numpy(X_vol).unsqueeze(0).unsqueeze(0)  # (1,1,H,W,D)
    Y_t = torch.from_numpy(Y_vol).unsqueeze(0).unsqueeze(0)  # (1,1,H,W,D)
    training_generator = [(X_t, Y_t)]

    grid_4 = generate_grid(imgshape_4)
    grid_4 = (
        torch.from_numpy(np.reshape(grid_4, (1,) + grid_4.shape)).to(device).float()
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model_dir = "/home/iml/fryderyk.koegl/code/LapIRN-koegl/checkpoints/Model/Stage"

    if not os.path.isdir(model_dir):
        from pathlib import Path

        Path(model_dir).mkdir(parents=True, exist_ok=True)
        # os.mkdir(model_dir)

    lossall = np.zeros((4, iteration_lvl1 + 1))

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

    while step <= iteration_lvl1:
        for X, Y in training_generator:
            X = X.to(device).float()
            Y = Y.to(device).float()

            if step == 0:
                save_volume(
                    X,
                    Path(model_dir) / "moving_image",
                    step,
                    Path(datapath) / "moving_ct.nii.gz",
                    "moving_ct",
                )
                save_volume(
                    Y,
                    Path(model_dir) / "fixed_image",
                    step,
                    Path(datapath) / "fixed_ct.nii.gz",
                    "fixed_ct",
                )

            # output_disp_e0, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, e0
            F_X_Y, X_Y, Y_4x, F_xy, _ = model(X, Y)

            if step % 100 == 0:
                save_volume(
                    X_Y,
                    Path(model_dir) / "warped_image",
                    step,
                    Path(datapath) / "fixed_ct.nii.gz",
                    "warped_ct",
                )

            # 3 level deep supervision NCC
            loss_multiNCC = loss_similarity(X_Y, Y_4x)

            F_X_Y_norm = transform_unit_flow_to_flow_cuda(
                F_X_Y.permute(0, 2, 3, 4, 1).clone()
            )

            loss_Jacobian = loss_Jdet(F_X_Y_norm, grid_4)

            # reg2 - use velocity
            _, _, x, y, z = F_xy.shape
            F_xy[:, 0, :, :, :] = F_xy[:, 0, :, :, :] * (z - 1)
            F_xy[:, 1, :, :, :] = F_xy[:, 1, :, :, :] * (y - 1)
            F_xy[:, 2, :, :, :] = F_xy[:, 2, :, :, :] * (x - 1)
            loss_regulation = loss_smooth(F_xy)

            loss = loss_multiNCC + antifold * loss_Jacobian + smooth * loss_regulation
            # loss = loss_multiNCC + smooth * loss_regulation

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()  # apply gradients

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
                + 'step "{0}" -> training loss "{1:.4f}" - sim_NCC "{2:4f}" - Jdet "{3:.10f}" -smo "{4:.4f}"'.format(
                    step,
                    loss.item(),
                    loss_multiNCC.item(),
                    loss_Jacobian.item(),
                    loss_regulation.item(),
                )
            )
            sys.stdout.flush()
            mlflow.log_metrics(
                {
                    "lvl1/loss": loss.item(),
                    "lvl1/ncc": loss_multiNCC.item(),
                    "lvl1/smooth": loss_regulation.item(),
                },
                step=step,
            )

            # with lr 1e-3 + with bias
            if step % n_checkpoint == 0:
                modelname = (
                    model_dir + "/" + model_name + "stagelvl1_" + str(step) + ".pth"
                )
                torch.save(model.state_dict(), modelname)
                np.save(
                    model_dir
                    + "/loss"
                    + model_name
                    + "stagelvl1_"
                    + str(step)
                    + ".npy",
                    lossall,
                )

            step += 1

            if step > iteration_lvl1:
                break
        print("one epoch pass")
    np.save(model_dir + "/loss" + model_name + "stagelvl1.npy", lossall)


def train_lvl2():
    print("Training lvl2...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_lvl1 = Miccai2020_LDR_laplacian_unit_add_lvl1(
        2, 3, start_channel, is_train=True, imgshape=imgshape_4, range_flow=range_flow
    ).to(device)

    # model_path = "../Model/Stage/LDR_LPBA_NCC_1_1_stagelvl1_1500.pth"
    model_path = sorted(
        glob.glob("../Model/Stage/" + model_name + "stagelvl1_?????.pth")
    )[-1]
    model_lvl1.load_state_dict(torch.load(model_path))
    print("Loading weight for model_lvl1...", model_path)

    # Freeze model_lvl1 weight
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
    loss_smooth = smoothloss
    loss_Jdet = neg_Jdet_loss

    transform = SpatialTransform_unit().to(device)

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True

    X_vol = nib.load(datapath + "/moving_ct.nii.gz").get_fdata().astype("float32")
    Y_vol = nib.load(datapath + "/fixed_ct.nii.gz").get_fdata().astype("float32")
    X_vol = (X_vol - X_vol.min()) / (X_vol.max() - X_vol.min() + 1e-8)
    Y_vol = (Y_vol - Y_vol.min()) / (Y_vol.max() - Y_vol.min() + 1e-8)
    X_t = torch.from_numpy(X_vol).unsqueeze(0).unsqueeze(0)  # (1,1,H,W,D)
    Y_t = torch.from_numpy(Y_vol).unsqueeze(0).unsqueeze(0)  # (1,1,H,W,D)
    training_generator = [(X_t, Y_t)]

    grid_2 = generate_grid(imgshape_2)
    grid_2 = (
        torch.from_numpy(np.reshape(grid_2, (1,) + grid_2.shape)).to(device).float()
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model_dir = "../Model/Stage"

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    lossall = np.zeros((4, iteration_lvl2 + 1))

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

    while step <= iteration_lvl2:
        for X, Y in training_generator:
            X = X.to(device).float()
            Y = Y.to(device).float()

            # output_disp_e0, warpped_inputx_lvl1_out, y_down, compose_field_e0_lvl1v, lvl1_v, e0
            F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, _ = model(X, Y)

            # 3 level deep supervision NCC
            loss_multiNCC = loss_similarity(X_Y, Y_4x)

            F_X_Y_norm = transform_unit_flow_to_flow_cuda(
                F_X_Y.permute(0, 2, 3, 4, 1).clone()
            )

            loss_Jacobian = loss_Jdet(F_X_Y_norm, grid_2)

            # reg2 - use velocity
            _, _, x, y, z = F_xy.shape
            F_xy[:, 0, :, :, :] = F_xy[:, 0, :, :, :] * (z - 1)
            F_xy[:, 1, :, :, :] = F_xy[:, 1, :, :, :] * (y - 1)
            F_xy[:, 2, :, :, :] = F_xy[:, 2, :, :, :] * (x - 1)
            loss_regulation = loss_smooth(F_xy)

            loss = loss_multiNCC + antifold * loss_Jacobian + smooth * loss_regulation

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()  # apply gradients

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
                + 'step "{0}" -> training loss "{1:.4f}" - sim_NCC "{2:4f}" - Jdet "{3:.10f}" -smo "{4:.4f}"'.format(
                    step,
                    loss.item(),
                    loss_multiNCC.item(),
                    loss_Jacobian.item(),
                    loss_regulation.item(),
                )
            )
            sys.stdout.flush()
            mlflow.log_metrics(
                {
                    "lvl2/loss": loss.item(),
                    "lvl2/ncc": loss_multiNCC.item(),
                    "lvl2/smooth": loss_regulation.item(),
                },
                step=step,
            )

            # with lr 1e-3 + with bias
            if step % n_checkpoint == 0:
                modelname = (
                    model_dir + "/" + model_name + "stagelvl2_" + str(step) + ".pth"
                )
                torch.save(model.state_dict(), modelname)
                np.save(
                    model_dir
                    + "/loss"
                    + model_name
                    + "stagelvl2_"
                    + str(step)
                    + ".npy",
                    lossall,
                )

            if step == freeze_step:
                model.unfreeze_modellvl1()

            step += 1

            if step > iteration_lvl2:
                break
        print("one epoch pass")
    np.save(model_dir + "/loss" + model_name + "stagelvl2.npy", lossall)


def train_lvl3():
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

    model_path = sorted(
        glob.glob("../Model/Stage/" + model_name + "stagelvl2_?????.pth")
    )[-1]
    model_lvl2.load_state_dict(torch.load(model_path))
    print("Loading weight for model_lvl2...", model_path)

    # Freeze model_lvl1 weight
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

    loss_smooth = smoothloss
    loss_Jdet = neg_Jdet_loss

    transform = SpatialTransform_unit().to(device)
    transform_nearest = SpatialTransformNearest_unit().to(device)

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True

    X_vol = nib.load(datapath + "/moving_ct.nii.gz").get_fdata().astype("float32")
    Y_vol = nib.load(datapath + "/fixed_ct.nii.gz").get_fdata().astype("float32")
    X_vol = (X_vol - X_vol.min()) / (X_vol.max() - X_vol.min() + 1e-8)
    Y_vol = (Y_vol - Y_vol.min()) / (Y_vol.max() - Y_vol.min() + 1e-8)
    X_t = torch.from_numpy(X_vol).unsqueeze(0).unsqueeze(0)  # (1,1,H,W,D)
    Y_t = torch.from_numpy(Y_vol).unsqueeze(0).unsqueeze(0)  # (1,1,H,W,D)
    training_generator = [(X_t, Y_t)]

    grid = generate_grid(imgshape)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).to(device).float()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model_dir = "../Model"

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    lossall = np.zeros((4, iteration_lvl3 + 1))

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

    while step <= iteration_lvl3:
        for X, Y in training_generator:
            X = X.to(device).float()
            Y = Y.to(device).float()

            # output_disp_e0, warpped_inputx_lvl1_out, y, compose_field_e0_lvl2_compose, lvl1_v, compose_lvl2_v, e0
            F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, F_xy_lvl2, _ = model(X, Y)

            # 3 level deep supervision NCC
            loss_multiNCC = loss_similarity(X_Y, Y_4x)

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

            loss = loss_multiNCC + antifold * loss_Jacobian + smooth * loss_regulation

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()  # apply gradients

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
                + 'step "{0}" -> training loss "{1:.4f}" - sim_NCC "{2:4f}" - Jdet "{3:.10f}" -smo "{4:.4f}"'.format(
                    step,
                    loss.item(),
                    loss_multiNCC.item(),
                    loss_Jacobian.item(),
                    loss_regulation.item(),
                )
            )
            sys.stdout.flush()
            mlflow.log_metrics(
                {
                    "lvl3/loss": loss.item(),
                    "lvl3/ncc": loss_multiNCC.item(),
                    "lvl3/smooth": loss_regulation.item(),
                },
                step=step,
            )

            # with lr 1e-3 + with bias
            if step % n_checkpoint == 0:
                modelname = (
                    model_dir + "/" + model_name + "stagelvl3_" + str(step) + ".pth"
                )
                torch.save(model.state_dict(), modelname)
                np.save(
                    model_dir
                    + "/loss"
                    + model_name
                    + "stagelvl3_"
                    + str(step)
                    + ".npy",
                    lossall,
                )

                # Validation

            if step == freeze_step:
                model.unfreeze_modellvl2()

            step += 1

            if step > iteration_lvl3:
                break
        print("one epoch pass")
    np.save(model_dir + "/loss" + model_name + "stagelvl3.npy", lossall)


imgshape = (192, 192, 288)
imgshape_4 = (192 // 4, 192 // 4, 288 // 4)  # (48, 48, 72)
imgshape_2 = (192 // 2, 192 // 2, 288 // 2)  # (96, 96, 144)

range_flow = 0.4

# datapath must point to a folder containing moving_ct.nii.gz and fixed_ct.nii.gz
# Override default argparse value:
datapath = "/home/iml/fryderyk.koegl/data/PSMAReg_dataset/imagesTr/PSMARegPSMA_0006"

mlflow.set_tracking_uri("sqlite:////home/iml/fryderyk.koegl/code/mlruns.db")
mlflow.set_experiment("PSMAReg_LapIRN_overfit_CT_only")
with mlflow.start_run():
    mlflow.log_params(
        {
            "imgshape": str(imgshape),
            "lr": lr,
            "smooth": smooth,
            "antifold": antifold,
            "start_channel": start_channel,
            "range_flow": range_flow,
        }
    )
    train_lvl1()
    # train_lvl2()
    # train_lvl3()
