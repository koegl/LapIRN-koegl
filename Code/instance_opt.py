import time
from typing import Dict, Optional, Tuple

import config
import Functions
import synthetic
import torch
import tqdm
import utils
from miccai2020_model_stage import NCC, jacobian_determinant, neg_Jdet_loss, smoothloss


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
    y: torch.Tensor,
    X_affine: torch.Tensor,
    x_lbl_ct: torch.Tensor,
    x_lbl_pet: torch.Tensor,
    y_lbl_ct: torch.Tensor,
    transform: torch.nn.Module,
    grid: torch.Tensor,
    cfg: config.TrainingConfig,
    bone_values: torch.Tensor,
    loss_ncc: NCC,
    ncc_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    disp_flow = disp_unit.permute(0, 2, 3, 4, 1)

    disp_voxel = Functions.transform_unit_flow_to_flow_cuda(disp_flow.clone())
    jac_det = jacobian_determinant(disp_voxel)
    loss_jac = neg_Jdet_loss(disp_voxel, grid)
    loss_smooth = smoothloss(disp_unit)

    # warp the moving image once, reuse the CT and PET channels
    x_y = transform(X_affine, disp_flow, grid)
    x_y_ct = x_y[:, 0:1]
    x_y_pet = x_y[:, 1:2]
    y_ct = y[:, 0:1]

    loss_ncc_ct = loss_ncc(x_y_ct, y_ct)

    loss_dice_ct = utils.dice_loss_with_grad(
        x_lbl_ct, y_lbl_ct, disp_unit, grid, transform
    )

    moving_pet_mask = (x_lbl_pet == 1).float()
    warped_pet_mask = utils.warp_binary_mask(
        moving_pet_mask, disp_unit, grid, transform
    )
    warped_pet_image = x_y_pet
    moving_pet_image = X_affine[:, 1:2]

    loss_mtv = utils.mtv_bias_loss(warped_pet_mask, moving_pet_mask)
    loss_tlg = utils.tlg_bias_loss(
        warped_pet_image, warped_pet_mask, moving_pet_image, moving_pet_mask
    )
    loss_masked_jac = utils.masked_jac_det_loss(jac_det, moving_pet_mask)

    moving_bone_mask = torch.isin(x_lbl_ct, bone_values).float()
    loss_masked_jac_bone = utils.masked_jac_det_loss(jac_det, moving_bone_mask)

    loss = (
        ncc_weight * loss_ncc_ct
        + cfg.w_jacobian * loss_jac
        + cfg.w_smooth * loss_smooth
        + cfg.w_tlg * loss_tlg
        + cfg.w_masked_jac * loss_masked_jac
        + cfg.w_masked_jac * loss_masked_jac_bone
    )
    if loss_dice_ct is not None:
        loss = loss + cfg.w_dice_ct * loss_dice_ct

    logs = {
        "ncc_ct": loss_ncc_ct.item(),
        "dice_ct": loss_dice_ct.item() if loss_dice_ct is not None else float("nan"),
        "smooth": loss_smooth.item(),
        "jac": loss_jac.item(),
        "masked_jac": loss_masked_jac.item(),
        "masked_jac_bone": loss_masked_jac_bone.item(),
        "mtv": loss_mtv.item(),
        "tlg": loss_tlg.item(),
    }
    return loss, logs


def run_io(
    y: torch.Tensor,
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
    lr: float = 1e-1,
    n_integration: int = 7,
    ncc_weight: Optional[float] = None,
) -> torch.Tensor:
    # NCC is not scored by the challenge; keep it as a modest dense proxy that
    # fills gradient where label-dice is flat. Try a small value (e.g. 1-3);
    # defaults to cfg.w_ct if not set.
    if ncc_weight is None:
        ncc_weight = cfg.w_ct

    identity_vox = synthetic.build_identity_grid(cfg.img_shape, device)
    velocity = torch.zeros_like(f_x_y, requires_grad=True)
    optimizer = torch.optim.Adam([velocity], lr=lr)

    bone_values = torch.tensor(
        synthetic.BONE_LABEL_VALUES, dtype=torch.float32, device=device
    )
    loss_ncc = NCC(cfg.lvl3_ncc_win)

    base = f_x_y.detach()
    best_loss = float("inf")
    best_disp = base.clone()
    best_disp_i = 0
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
            y,
            X_affine,
            x_lbl_ct,
            x_lbl_pet,
            y_lbl_ct,
            transform,
            grid,
            cfg,
            bone_values,
            loss_ncc=loss_ncc,
            ncc_weight=ncc_weight,
        )
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_disp = disp_unit.detach().clone()
            best_disp_i = i

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            best=f"{best_loss:.4f}",
            best_i=best_disp_i,
            time=f"{time.time() - start_time:.2f}s",
        )

    print(f"Best loss: {best_loss:.4f} at step {best_disp_i}")

    refined = best_disp
    return refined
