import time
from typing import Callable, Dict, Optional, Tuple

import config
import Functions
import jacobian
import synthetic
import torch
import tqdm
import utils
from miccai2020_model_stage import (
    NCC,
    SpatialTransform_unit,
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


def dice_loss_with_grad(
    moving_label: torch.Tensor,
    fixed_label: torch.Tensor,
    disp: torch.Tensor,
    grid: torch.Tensor,
    transform: SpatialTransform_unit,
    class_weights: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor | None:
    """Per-class soft dice loss. If class_weights is given (one weight per
    class value, indexed by the integer label), the per-class dice terms are
    weighted before averaging; weights are renormalized over the classes
    actually used so the loss scale stays comparable across subjects."""
    classes = fixed_label.unique()
    classes = classes[classes != 0]
    if classes.numel() == 0:
        return None
    flow = disp.permute(0, 2, 3, 4, 1)
    dice_scores = []
    weights = []
    for c in classes:
        moving_c = (moving_label == c).float()
        fixed_c = (fixed_label == c).float()
        warped_c = transform(moving_c, flow, grid)
        intersection = (warped_c * fixed_c).sum()
        cardinality = warped_c.sum() + fixed_c.sum()
        dice_c = (2.0 * intersection + eps) / (cardinality + eps)
        dice_scores.append(dice_c)
        if class_weights is not None:
            weights.append(class_weights[int(c.item())])
    dice_stack = torch.stack(dice_scores)
    if class_weights is not None:
        w = torch.stack(weights)
        w = w / (w.mean() + eps)
        loss = 1.0 - (w * dice_stack).sum() / w.sum()
    else:
        loss = 1.0 - dice_stack.mean()
    return loss


import numpy as np


def multilabel_dice(
    pred: torch.Tensor,
    target: torch.Tensor,
    label_ids: range = range(1, 118),
) -> float:
    """Mean Dice over a fixed label set, matching the official scorer.
    pred/target: (H, W, D) int."""
    dices = []
    for lbl in label_ids:
        p = pred == lbl
        t = target == lbl
        volume_sum = p.sum() + t.sum()
        if volume_sum == 0:
            dice = 0.0
        else:
            dice = (2.0 * (p & t).sum() / volume_sum).item()
        dices.append(dice)
    mean = float(np.mean(dices)) if dices else float("nan")
    return mean


def compute_io_loss(
    disp_unit: torch.Tensor,
    y: torch.Tensor,
    X_affine: torch.Tensor,
    x_lbl_ct: torch.Tensor,
    x_lbl_pet: torch.Tensor,
    y_lbl_ct: torch.Tensor,
    transform: torch.nn.Module,
    transform_nearest: torch.nn.Module,
    grid: torch.Tensor,
    cfg: config.TrainingConfig,
    bone_values: torch.Tensor,
    loss_ncc: NCC,
    ncc_weight: float,
    class_weights: torch.Tensor | None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    disp_flow = disp_unit.permute(0, 2, 3, 4, 1)

    disp_voxel = Functions.transform_unit_flow_to_flow_cuda(disp_flow.clone())
    jac_det, jac = jacobian.jacobian_matrix(disp_voxel)
    loss_jac = jacobian.non_diff_volume_loss(disp_voxel)
    loss_smooth = smoothloss(disp_unit)

    # warp the moving image once, reuse the CT and PET channels
    x_y = transform(X_affine, disp_flow, grid)
    x_y_ct = x_y[:, 0:1]
    x_y_pet = x_y[:, 1:2]
    y_ct = y[:, 0:1]

    loss_ncc_ct = loss_ncc(x_y_ct, y_ct)

    loss_dice_ct = dice_loss_with_grad(
        x_lbl_ct, y_lbl_ct, disp_unit, grid, transform, class_weights=class_weights
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
    loss_rigidity, _ = utils.enforce_rigidity_loss(
        jac_det, jac, disp_voxel, moving_bone_mask
    )

    with torch.no_grad():
        warped_lbl_ct = warp_label(x_lbl_ct, disp_unit, grid, transform_nearest)
        pred = warped_lbl_ct[0, 0].round().long()
        target = y_lbl_ct[0, 0].round().long()
        hard_dices = []
        for lbl in range(1, 118):
            p = pred == lbl
            t = target == lbl
            volume_sum = p.sum() + t.sum()
            dice = 0.0 if volume_sum == 0 else (2.0 * (p & t).sum() / volume_sum).item()
            hard_dices.append(dice)
        hard_dice = float(np.mean(hard_dices))

    loss = (
        ncc_weight * loss_ncc_ct
        + cfg.w_jacobian * loss_jac
        # + cfg.w_smooth * loss_smooth
        + cfg.w_tlg * loss_tlg
        + cfg.w_jacobian_tumor * loss_masked_jac
        + cfg.w_bone_rigidity * loss_rigidity
    )
    if loss_dice_ct is not None:
        loss = loss + cfg.w_dice_ct_lvl3 * loss_dice_ct

    logs = {
        "ncc_ct": loss_ncc_ct.item(),
        "dice_ct": loss_dice_ct.item() if loss_dice_ct is not None else float("nan"),
        "hard_dice_ct": hard_dice,
        "smooth": loss_smooth.item(),
        "jac": loss_jac.item(),
        "masked_jac": loss_masked_jac.item(),
        "bone_rigidity": loss_rigidity.item(),
        "mtv": loss_mtv.item(),
        "tlg": loss_tlg.item(),
    }
    return loss, logs


def build_class_weights(
    moving_label: torch.Tensor,
    fixed_label: torch.Tensor,
    n_labels: int,
    device: torch.device,
) -> torch.Tensor:
    """w_c = 1 - starting_hard_dice_c, indexed by label value (size n_labels)."""
    weights = torch.ones(n_labels, device=device)
    pred = moving_label[0, 0].round().long()
    target = fixed_label[0, 0].round().long()
    for lbl in range(1, n_labels):
        p = pred == lbl
        t = target == lbl
        volume_sum = p.sum() + t.sum()
        if volume_sum == 0:
            continue
        dice = (2.0 * (p & t).sum() / volume_sum).item()
        weights[lbl] = 1.0 - dice

    return weights


# ---------------------------------------------------------------------------
# Shared IO refinement operator
#
# Both test-time IO (run_io) and training-time unrolled IO parametrize the
# refinement as a half-resolution stationary velocity field, integrate it with
# scaling-and-squaring, upsample to full resolution and add it to the network's
# output `base`. Keeping the parametrization in one place guarantees that what
# we train against is exactly what we deploy.
# ---------------------------------------------------------------------------


def svf_to_disp(
    base: torch.Tensor,
    velocity: torch.Tensor,
    identity_vox_half: torch.Tensor,
    cfg: config.TrainingConfig,
    n_integration: int,
) -> torch.Tensor:
    """base (unit flow) + refinement decoded from a half-res velocity field.

    Returns a full-resolution unit-flow displacement. Fully differentiable in
    both `velocity` (inner loop) and `base` (network output)."""
    disp_res_half = synthetic.integrate_svf(velocity, identity_vox_half, n_integration)
    disp_res_full = torch.nn.functional.interpolate(
        disp_res_half, scale_factor=2, mode="trilinear", align_corners=False
    )
    disp_res_unit = synthetic.unit_flow_from_voxel_disp(
        disp_res_full, cfg.img_shape
    ).flip(1)
    return base + disp_res_unit


def unrolled_io_loss(
    disp_unit: torch.Tensor,
    y_ct: torch.Tensor,
    x_ct: torch.Tensor,
    x_lbl_ct: torch.Tensor,
    y_lbl_ct: torch.Tensor,
    transform: torch.nn.Module,
    grid: torch.Tensor,
    cfg: config.TrainingConfig,
    loss_ncc: NCC,
    ncc_weight: float,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """CT-only IO objective used for training-time unrolling.

    A differentiable subset of compute_io_loss (label dice + CT NCC + Jacobian
    volume penalty). PET/MTV/TLG/rigidity terms are dropped because they are
    gated off during lvl3 training anyway, and the measured amortization gap is
    a label-dice gap, so this is the objective the net should be seeded for."""
    disp_flow = disp_unit.permute(0, 2, 3, 4, 1)
    disp_voxel = Functions.transform_unit_flow_to_flow_cuda(disp_flow.clone())
    loss_jac = jacobian.non_diff_volume_loss(disp_voxel)

    x_y_ct = transform(x_ct, disp_flow, grid)
    loss_ncc_ct = loss_ncc(x_y_ct, y_ct)

    loss_dice_ct = dice_loss_with_grad(
        x_lbl_ct, y_lbl_ct, disp_unit, grid, transform, class_weights=None
    )

    loss = ncc_weight * loss_ncc_ct + cfg.w_jacobian * loss_jac
    if loss_dice_ct is not None:
        loss = loss + cfg.w_dice_ct_lvl3 * loss_dice_ct
    return loss


def unrolled_refine(
    base: torch.Tensor,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    cfg: config.TrainingConfig,
    device: torch.device,
    n_steps: int,
    inner_lr: float,
    n_integration: int = 7,
    mode: str = "fomaml",
) -> torch.Tensor:
    """Run `n_steps` differentiable gradient-descent IO steps starting from the
    network output `base`, and return the refined full-res unit flow.

    The returned tensor stays connected to `base` (and hence the network
    weights) so the caller can backprop a loss on the refined field into the
    net - the meta-learning signal "how should my output change so that a few IO
    steps get further".

    mode:
      "full"   - keep the whole inner trajectory in the graph and differentiate
                 through every step (create_graph=True). K x memory.
      "fomaml" - first-order: take the inner steps without retaining their graph,
                 then rebuild the final field so gradient reaches `base` only
                 through the explicit `base + refinement` term. ~1x memory.
    """
    if mode not in ("full", "fomaml"):
        raise ValueError(f"unknown unroll mode: {mode!r}")

    half_shape = tuple(s // 2 for s in cfg.img_shape)
    identity_vox_half = synthetic.build_identity_grid(half_shape, device)
    velocity = torch.zeros((1, 3) + half_shape, device=device, requires_grad=True)

    create_graph = mode == "full"
    # In fomaml the inner loop must not see gradients flowing back into `base`,
    # otherwise we'd still pay the full unrolled memory cost.
    inner_base = base if create_graph else base.detach()

    for _ in range(n_steps):
        disp_unit = svf_to_disp(
            inner_base, velocity, identity_vox_half, cfg, n_integration
        )
        loss = loss_fn(disp_unit)
        (grad,) = torch.autograd.grad(loss, velocity, create_graph=create_graph)
        if create_graph:
            velocity = velocity - inner_lr * grad
        else:
            # detach so each inner step frees its graph (cheap, first-order)
            velocity = (velocity - inner_lr * grad).detach().requires_grad_(True)

    # Final field. In "full" this closes the K-step graph; in "fomaml" the
    # velocity is frozen and gradient reaches the net only via `base`.
    final_velocity = velocity if create_graph else velocity.detach()
    return svf_to_disp(base, final_velocity, identity_vox_half, cfg, n_integration)


def run_io(
    y: torch.Tensor,
    f_x_y: torch.Tensor,
    X_affine: torch.Tensor,
    x_lbl_ct: torch.Tensor,
    x_lbl_pet: torch.Tensor,
    y_lbl_ct: torch.Tensor,
    transform: torch.nn.Module,
    transform_nearest: torch.nn.Module,
    grid: torch.Tensor,
    cfg: config.TrainingConfig,
    device: torch.device,
    use_class_weights: bool = False,
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

    half_shape = tuple(s // 2 for s in cfg.img_shape)
    identity_vox_half = synthetic.build_identity_grid(half_shape, device)
    velocity = torch.zeros((1, 3) + half_shape, device=device)
    velocity.requires_grad_(True)
    optimizer = torch.optim.Adam([velocity], lr=lr)

    bone_values = torch.tensor(
        synthetic.BONE_LABEL_VALUES, dtype=torch.float32, device=device
    )
    loss_ncc = NCC(cfg.lvl3_ncc_win)

    base = f_x_y.detach()
    if use_class_weights:
        x_lbl_ct_start = warp_label(
            x_lbl_ct, base, grid, transform_nearest
        )  # or transform w/ nearest
        class_weights = build_class_weights(
            x_lbl_ct_start, y_lbl_ct, n_labels=118, device=device
        )
    else:
        class_weights = None

    base = f_x_y.detach()
    best_loss = float("inf")
    best_disp = base.clone()
    best_disp_i = 0
    pbar = tqdm.tqdm(range(n_steps), desc="IO optimization")
    for i in pbar:
        start_time = time.time()
        optimizer.zero_grad()
        disp_unit = svf_to_disp(base, velocity, identity_vox_half, cfg, n_integration)
        loss, logs = compute_io_loss(
            disp_unit,
            y,
            X_affine,
            x_lbl_ct,
            x_lbl_pet,
            y_lbl_ct,
            transform,
            transform_nearest,
            grid,
            cfg,
            bone_values,
            loss_ncc=loss_ncc,
            ncc_weight=ncc_weight,
            class_weights=class_weights,
        )
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_disp = disp_unit.detach().clone()
            best_disp_i = i

        pbar.set_postfix(
            dice_weighted=f"{1 - logs['dice_ct']:.4f}",
            dice_hard=f"{logs['hard_dice_ct']:.4f}",
            loss=f"{loss.item():.4f}",
            best=f"{best_loss:.4f}",
            best_i=best_disp_i,
            time=f"{time.time() - start_time:.2f}s",
        )

    print(f"Best loss: {best_loss:.4f} at step {best_disp_i}")

    refined = best_disp
    return refined
