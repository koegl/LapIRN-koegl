import time
from typing import Dict, Tuple

import config
import Functions
import synthetic
import torch
import tqdm
import utils
from miccai2020_model_stage import (
    jacobian_determinant,
    neg_Jdet_loss,
    smoothloss,
)


def warp_label(
    label: torch.Tensor,
    disp_unit: torch.Tensor,
    grid: torch.Tensor,
    transform_nearest: torch.nn.Module,
) -> torch.Tensor:
    warped = transform_nearest(label, disp_unit.permute(0, 2, 3, 4, 1), grid)
    return warped


def compute_io_loss(
    disp_unit: torch.Tensor,
    X_affine: torch.Tensor,
    x_lbl_ct: torch.Tensor,
    x_lbl_pet: torch.Tensor,
    y_lbl_ct: torch.Tensor,
    transform: torch.nn.Module,
    grid: torch.Tensor,
    cfg: config.TrainingConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    disp_flow = disp_unit.permute(0, 2, 3, 4, 1)

    disp_voxel = Functions.transform_unit_flow_to_flow_cuda(disp_flow.clone())
    jac_det = jacobian_determinant(disp_voxel)
    loss_jac = neg_Jdet_loss(disp_voxel, grid)
    loss_smooth = smoothloss(disp_unit)

    loss_dice_ct = utils.dice_loss_with_grad(
        x_lbl_ct, y_lbl_ct, disp_unit, grid, transform
    )

    moving_pet_mask = (x_lbl_pet == 1).float()
    warped_pet_mask = utils.warp_binary_mask(
        moving_pet_mask, disp_unit, grid, transform
    )
    warped_pet_image = transform(
        X_affine[:, 1:2], disp_unit.permute(0, 2, 3, 4, 1), grid
    )
    moving_pet_image = X_affine[:, 1:2]

    loss_mtv = utils.mtv_bias_loss(warped_pet_mask, moving_pet_mask)
    loss_tlg = utils.tlg_bias_loss(
        warped_pet_image, warped_pet_mask, moving_pet_image, moving_pet_mask
    )
    loss_masked_jac = utils.masked_jac_det_loss(jac_det, moving_pet_mask)

    loss = (
        cfg.w_jacobian * loss_jac
        + cfg.w_smooth * loss_smooth
        + cfg.w_mtv * loss_mtv
        + cfg.w_tlg * loss_tlg
        + cfg.w_masked_jac * loss_masked_jac
    )
    if loss_dice_ct is not None:
        loss = loss + cfg.w_dice_ct * loss_dice_ct

    logs = {
        "dice_ct": loss_dice_ct.item() if loss_dice_ct is not None else float("nan"),
        "smooth": loss_smooth.item(),
        "jac": loss_jac.item(),
        "masked_jac": loss_masked_jac.item(),
        "mtv": loss_mtv.item(),
        "tlg": loss_tlg.item(),
    }

    # print each los and next to it each loss multiplied by its weights
    # print(
    #     f"dice_ct: {logs['dice_ct']:.4f} ({cfg.w_dice_ct * logs['dice_ct']:.4f})\n"
    #     f"smooth: {logs['smooth']:.4f} ({cfg.w_smooth * logs['smooth']:.4f})\n"
    #     f"jac: {logs['jac']:.4f} ({cfg.w_jacobian * logs['jac']:.4f})\n"
    #     f"masked_jac: {logs['masked_jac']:.4f} ({cfg.w_masked_jac * logs['masked_jac']:.4f})\n"
    #     f"mtv: {logs['mtv']:.4f} ({cfg.w_mtv * logs['mtv']:.4f})\n"
    #     f"tlg: {logs['tlg']:.4f} ({cfg.w_tlg * logs['tlg']:.4f})"
    # )
    return loss, logs


def run_io(
    f_x_y: torch.Tensor,
    X_affine: torch.Tensor,
    x_lbl_ct: torch.Tensor,
    x_lbl_pet: torch.Tensor,
    y_lbl_ct: torch.Tensor,
    transform: torch.nn.Module,
    grid: torch.Tensor,
    cfg: config.TrainingConfig,
    device: torch.device,
    n_steps: int = 100,
    lr: float = 1e-2,
    n_integration: int = 7,
) -> torch.Tensor:
    identity_vox = synthetic.build_identity_grid(cfg.img_shape, device)
    velocity = torch.zeros_like(f_x_y, requires_grad=True)
    optimizer = torch.optim.Adam([velocity], lr=lr)

    base = f_x_y.detach()
    pbar = tqdm.tqdm(range(n_steps), desc="IO optimization")
    for i in pbar:
        start_time = time.time()
        optimizer.zero_grad()
        disp_res = synthetic.integrate_svf(velocity, identity_vox, n_integration)
        disp_res_unit = synthetic.unit_flow_from_voxel_disp(
            disp_res, cfg.img_shape
        ).flip(1)
        disp_unit = base + disp_res_unit
        loss, _ = compute_io_loss(
            disp_unit,
            X_affine,
            x_lbl_ct,
            x_lbl_pet,
            y_lbl_ct,
            transform,
            grid,
            cfg,
        )
        loss.backward()
        optimizer.step()
        pbar.set_postfix(
            loss=f"{loss.item():.4f}", time=f"{time.time() - start_time:.2f}s"
        )

    disp_res = synthetic.integrate_svf(velocity.detach(), identity_vox, n_integration)
    disp_res_unit = synthetic.unit_flow_from_voxel_disp(disp_res, cfg.img_shape).flip(1)
    refined = (base + disp_res_unit).detach()
    return refined
