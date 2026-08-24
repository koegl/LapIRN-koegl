import csv
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import config
import Functions
import jacobian
import numpy as np
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
    chunk_size: int = 16,
) -> torch.Tensor | None:
    """Per-class soft dice loss. If class_weights is given (one weight per
    class value, indexed by the integer label), the per-class dice terms are
    weighted before averaging; weights are renormalized over the classes
    actually used so the loss scale stays comparable across subjects.

    Thin wrapper over utils.dice_loss_with_grad so IO and training share one
    implementation (chunked warps with a hoisted sampling grid, and skipping of
    classes missing from either label)."""
    return utils.dice_loss_with_grad(
        moving_label,
        fixed_label,
        disp,
        grid,
        transform,
        eps=eps,
        class_weights=class_weights,
        chunk_size=chunk_size,
    )


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


# weights of the two MTV terms in io_objective. W_MTV is also used by run_io
# to swap the soft squared-MTV contribution for the hard (nearest-counted) one
# when selecting the best step, so keep them defined in one place.


def io_objective(
    disp_unit: torch.Tensor,
    x_moving: torch.Tensor,
    y_ct: torch.Tensor,
    x_lbl_ct: torch.Tensor | None,
    y_lbl_ct: torch.Tensor | None,
    transform: torch.nn.Module,
    grid: torch.Tensor,
    cfg: config.TrainingConfig,
    loss_ncc: NCC,
    ncc_weight: float,
    *,
    class_weights: torch.Tensor | None = None,
    x_lbl_pet: torch.Tensor | None = None,
    pet_cc_masks: torch.Tensor | None = None,
    bone_values: torch.Tensor | None = None,
    transform_nearest: torch.nn.Module | None = None,
    include_pet: bool = True,
    include_rigidity: bool = True,
    include_dice: bool = True,
    compute_hard_dice: bool = False,
    log_hard_dice: bool = True,
    n_hard_dice_labels: int = 118,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """The single IO objective shared by deploy-time IO (compute_io_loss /
    run_io) and training-time unrolling (unrolled_io_loss).

    Keeping both callers routed through one function is what guarantees that the
    inner loop the net is *seeded* for stays in lockstep with the objective we
    actually descend at test time - the two used to drift silently. Toggle the
    include_* groups to deliberately diverge (e.g. a CT-only unroll).

    x_moving: moving image, channel 0 = CT, channel 1 = PET (the PET channel is
        only read when include_p

    Differentiability note: with unroll_mode="full" the PET and rigidity terms
    are differentiated twice (double-backward); they are only guaranteed
    first-order smooth, so parity there is best paired with mode="fomaml"."""
    device = disp_unit.device
    disp_flow = disp_unit.permute(0, 2, 3, 4, 1)
    disp_voxel = Functions.transform_unit_flow_to_flow_cuda(disp_flow.clone())
    loss_jac = jacobian.non_diff_volume_loss(disp_voxel)
    loss_smooth = smoothloss(disp_unit)

    # warp the moving image once, reuse the CT and PET channels
    x_y = transform(x_moving, disp_flow, grid)
    x_y_ct = x_y[:, 0:1]
    loss_ncc_ct = loss_ncc(x_y_ct, y_ct)

    # include_dice=False drops the only term that reads the CT segmentations,
    # which together with include_pet=False / include_rigidity=False leaves a
    # purely image-driven objective (NCC + smoothness + the Jacobian barrier) --
    # the label-free setting we may face at test time.
    loss_dice_ct = None
    if include_dice:
        loss_dice_ct = dice_loss_with_grad(
            x_lbl_ct, y_lbl_ct, disp_unit, grid, transform, class_weights=class_weights
        )

    loss = (
        ncc_weight * loss_ncc_ct
        + cfg.w_io_non_diff * loss_jac
        + cfg.w_io_smooth * loss_smooth
    )

    if loss_dice_ct is not None:
        loss = loss + cfg.w_io_dice * loss_dice_ct

    # jac_det / jac are only needed by the PET-tumor and rigidity terms
    jac_det = jac = None
    if include_pet or include_rigidity:
        jac_det, jac = jacobian.jacobian_matrix(disp_voxel)

    zero = torch.zeros((), device=device)
    loss_mtv = loss_tlg = loss_masked_jac = loss_mtv_avg = zero
    loss_mtv_cc = loss_tlg_cc = loss_mtv_avg_cc = zero
    if include_pet:
        assert x_lbl_pet is not None, "include_pet requires x_lbl_pet"
        moving_pet_mask = (x_lbl_pet == 1).float()
        warped_pet_mask = utils.warp_binary_mask(
            moving_pet_mask, disp_unit, grid, transform
        )
        warped_pet_image = x_y[:, 1:2]
        moving_pet_image = x_moving[:, 1:2]
        loss_mtv = utils.mtv_bias_loss(warped_pet_mask, moving_pet_mask)
        loss_tlg = utils.tlg_bias_loss(
            warped_pet_image, warped_pet_mask, moving_pet_image, moving_pet_mask
        )
        # det(J) lives on the FIXED grid, so the jac-based volume terms are
        # masked with the fixed-frame lesion (the warped mask), not the moving
        # one — with the total field those regions sit a whole affine apart.
        # Detached: gradient must flow through det(J), not through shifting the
        # mask off the high-det region.
        fixed_pet_mask = warped_pet_mask.detach()
        loss_masked_jac = utils.masked_jac_det_loss(jac_det, fixed_pet_mask)

        loss_mtv_avg = utils.mtv_mean_bias_loss(jac_det, fixed_pet_mask)

        loss += (
            cfg.w_io_mtv * loss_mtv**2
            + cfg.w_io_mtv_avg * loss_mtv_avg
            + cfg.w_io_jacobian_tumor * loss_masked_jac
            + cfg.w_io_tlg * loss_tlg
        )

        # per-lesion terms. The global MTV / TLG above are sums over the whole
        # mask, so a lesion that expands cancels one that contracts; these pin
        # each component on its own. One extra warp of a (1, C, ...) stack --
        # the components are disjoint, so warping the stack is the same as
        # warping each mask separately, in a single grid_sample.
        use_cc = pet_cc_masks is not None and (
            cfg.w_io_mtv_cc > 0.0 or cfg.w_io_tlg_cc > 0.0 or cfg.w_io_mtv_avg_cc > 0.0
        )
        if use_cc:
            warped_cc = utils.warp_binary_mask(pet_cc_masks, disp_unit, grid, transform)
            loss_mtv_cc = utils.mtv_bias_loss_per_component(warped_cc, pet_cc_masks)
            loss_tlg_cc = utils.tlg_bias_loss_per_component(
                warped_pet_image, warped_cc, moving_pet_image, pet_cc_masks
            )
            # same convention as the global mtv_avg: det(J) lives on the fixed
            # grid, so the mask is the warped stack, detached (gradient must
            # flow through det(J), not through moving the mask)
            loss_mtv_avg_cc = utils.mtv_mean_bias_loss_per_component(
                jac_det, warped_cc.detach()
            )
            loss += (
                cfg.w_io_mtv_cc * loss_mtv_cc
                + cfg.w_io_tlg_cc * loss_tlg_cc
                + cfg.w_io_mtv_avg_cc * loss_mtv_avg_cc
            )

    loss_rigidity = zero
    if include_rigidity:
        assert bone_values is not None, "include_rigidity requires bone_values"
        loss_rigidity, _ = utils.per_label_rigid_loss(
            disp_voxel,
            x_lbl_ct,
            bone_values,
            min_voxels=cfg.rigidity_min_voxels,
        )
        loss = loss + cfg.w_io_bone_rigidity * loss_rigidity

    hard_dice = float("nan")
    hard_mtv = float("nan")
    hard_tlg = float("nan")
    if compute_hard_dice:
        assert transform_nearest is not None, (
            "compute_hard_dice needs transform_nearest"
        )
        with torch.no_grad():
            if include_pet:
                # MTV exactly as the scorer counts it: nearest-warped binary
                # mask, whole voxels (the soft bilinear `mtv` sits between the
                # scorer's integer rounding levels)
                warped_pet_hard = warp_label(
                    moving_pet_mask, disp_unit, grid, transform_nearest
                )
                n_moving = moving_pet_mask.sum()
                warped_pet_hard = (warped_pet_hard > 0.5).float()
                n_warped = warped_pet_hard.sum()
                hard_mtv = (
                    torch.abs(n_warped - n_moving) / n_moving.clamp_min(1)
                ).item()
                # TLG exactly as the scorer computes it: the lesion mask warped
                # with NEAREST, the PET image warped with bilinear. The soft
                # `tlg` above uses the bilinear mask for both, which is not a
                # quantity the scorer ever evaluates.
                tlg_moving_hard = (moving_pet_image * moving_pet_mask).sum()
                tlg_warped_hard = (warped_pet_image * warped_pet_hard).sum()
                hard_tlg = (
                    torch.abs(tlg_warped_hard - tlg_moving_hard)
                    / tlg_moving_hard.clamp_min(1e-5)
                ).item()
            # hard-dice logging reads the CT segmentations, so it needs
            # include_dice. hard_mtv / hard_tlg above must NOT: run_io selects
            # its best step with them. log_hard_dice switches this off even when
            # the dice LOSS is on -- it is 117 full-volume comparisons per step,
            # bought purely for a progress-bar number, which the container's
            # runtime budget cannot spare.
            if include_dice and log_hard_dice:
                warped_lbl_ct = warp_label(x_lbl_ct, disp_unit, grid, transform_nearest)
                pred = warped_lbl_ct[0, 0].round().long()
                target = y_lbl_ct[0, 0].round().long()
                hard_dices = []
                for lbl in range(1, n_hard_dice_labels):
                    p = pred == lbl
                    t = target == lbl
                    volume_sum = p.sum() + t.sum()
                    dice = (
                        0.0
                        if volume_sum == 0
                        else (2.0 * (p & t).sum() / volume_sum).item()
                    )
                    hard_dices.append(dice)
                hard_dice = float(np.mean(hard_dices))

    logs = {
        "ncc_ct": loss_ncc_ct.item(),
        "dice_ct": loss_dice_ct.item() if loss_dice_ct is not None else float("nan"),
        "hard_dice_ct": hard_dice,
        # "smooth": loss_smooth.item(),
        "jac": loss_jac.item(),
        "masked_jac": loss_masked_jac.item(),
        "bone_rigidity": loss_rigidity.item(),
        "mtv": loss_mtv.item(),
        "hard_mtv": hard_mtv,
        "mtv_avg": loss_mtv_avg.item(),
        "tlg": loss_tlg.item(),
        "hard_tlg": hard_tlg,
        "mtv_cc": loss_mtv_cc.item(),
        "tlg_cc": loss_tlg_cc.item(),
        "mtv_avg_cc": loss_mtv_avg_cc.item(),
    }
    return loss, logs


def compute_io_loss(
    disp_unit: torch.Tensor,
    y: torch.Tensor,
    x_moving: torch.Tensor,
    x_lbl_ct: torch.Tensor | None,
    x_lbl_pet: torch.Tensor,
    y_lbl_ct: torch.Tensor | None,
    transform: torch.nn.Module,
    transform_nearest: torch.nn.Module,
    grid: torch.Tensor,
    cfg: config.TrainingConfig,
    bone_values: torch.Tensor | None,
    loss_ncc: NCC,
    ncc_weight: float,
    include_pet: bool,
    include_rigidity: bool,
    class_weights: torch.Tensor | None,
    pet_cc_masks: torch.Tensor | None = None,
    include_dice: bool = True,
    log_hard_dice: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Deploy-time IO objective (used by run_io). Thin wrapper over io_objective
    with every term on and hard-dice logging enabled.

    x_moving / x_lbl_ct / x_lbl_pet are in whatever frame disp_unit maps from:
    the inference pipeline passes the ORIGINAL moving image and labels together
    with the TOTAL (affine-composed) field, so the volume terms include the
    affine's det(A) and match the final eval."""
    return io_objective(
        disp_unit,
        x_moving=x_moving,
        y_ct=y[:, 0:1],
        x_lbl_ct=x_lbl_ct,
        y_lbl_ct=y_lbl_ct,
        transform=transform,
        grid=grid,
        cfg=cfg,
        loss_ncc=loss_ncc,
        ncc_weight=ncc_weight,
        class_weights=class_weights,
        x_lbl_pet=x_lbl_pet,
        pet_cc_masks=pet_cc_masks,
        bone_values=bone_values,
        transform_nearest=transform_nearest,
        include_pet=include_pet,
        include_rigidity=include_rigidity,
        include_dice=include_dice,
        compute_hard_dice=True,
        log_hard_dice=log_hard_dice,
    )


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
# refinement as a full-resolution stationary velocity field, integrate it with
# scaling-and-squaring and add it to the network's output `base`. Keeping the
# parametrization in one place guarantees that what we train against is exactly
# what we deploy.
# ---------------------------------------------------------------------------


def svf_to_disp(
    base: torch.Tensor,
    velocity: torch.Tensor,
    identity_vox: torch.Tensor,
    cfg: config.TrainingConfig,
    n_integration: int,
    shape: Optional[Tuple[int, int, int]] = None,
) -> torch.Tensor:
    """base (unit flow) + refinement decoded from a velocity field.

    Returns a unit-flow displacement at `shape` (defaults to cfg.img_shape).
    `shape` must match the velocity / identity_vox / base spatial size so the
    voxel->unit conversion uses the right dims; pass the half shape to optimise
    a half-resolution field. Fully differentiable in both `velocity` (inner
    loop) and `base` (network output)."""
    if shape is None:
        shape = cfg.img_shape
    disp_res_full = synthetic.integrate_svf(velocity, identity_vox, n_integration)
    disp_res_unit = synthetic.unit_flow_from_voxel_disp(disp_res_full, shape).flip(1)
    return base + disp_res_unit


def unrolled_io_loss(
    disp_unit: torch.Tensor,
    y_ct: torch.Tensor,
    x_moving: torch.Tensor,
    x_lbl_ct: torch.Tensor,
    y_lbl_ct: torch.Tensor,
    transform: torch.nn.Module,
    grid: torch.Tensor,
    cfg: config.TrainingConfig,
    loss_ncc: NCC,
    ncc_weight: float,
    *,
    class_weights: torch.Tensor | None = None,
    x_lbl_pet: torch.Tensor | None = None,
    bone_values: torch.Tensor | None = None,
    include_pet: bool = True,
    include_rigidity: bool = True,
) -> torch.Tensor:
    """IO objective used for training-time unrolling. Thin wrapper over
    io_objective so the inner loop the net is seeded for matches the deployed
    run_io objective (compute_io_loss) term-for-term by default.

    Set include_pet / include_rigidity False to fall back to the CT-only subset
    (label dice + CT NCC + Jacobian volume penalty). x_moving must carry the PET
    channel (index 1) when include_pet. hard-dice logging is skipped: the
    nearest-warp label loop is pure overhead inside the unroll."""
    loss, _ = io_objective(
        disp_unit,
        x_moving=x_moving,
        y_ct=y_ct,
        x_lbl_ct=x_lbl_ct,
        y_lbl_ct=y_lbl_ct,
        transform=transform,
        grid=grid,
        cfg=cfg,
        loss_ncc=loss_ncc,
        ncc_weight=ncc_weight,
        class_weights=class_weights,
        x_lbl_pet=x_lbl_pet,
        bone_values=bone_values,
        include_pet=include_pet,
        include_rigidity=include_rigidity,
        compute_hard_dice=False,
    )
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

    full_shape = tuple(cfg.img_shape)
    identity_vox_full = synthetic.build_identity_grid(full_shape, device)
    velocity = torch.zeros((1, 3) + full_shape, device=device, requires_grad=True)

    create_graph = mode == "full"
    # In fomaml the inner loop must not see gradients flowing back into `base`,
    # otherwise we'd still pay the full unrolled memory cost.
    inner_base = base if create_graph else base.detach()

    for _ in range(n_steps):
        disp_unit = svf_to_disp(
            inner_base, velocity, identity_vox_full, cfg, n_integration
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
    return svf_to_disp(base, final_velocity, identity_vox_full, cfg, n_integration)


def run_io(
    y: torch.Tensor,
    f_x_y: torch.Tensor,
    x_moving: torch.Tensor,
    x_lbl_ct: torch.Tensor | None,
    x_lbl_pet: torch.Tensor,
    y_lbl_ct: torch.Tensor | None,
    transform: torch.nn.Module,
    transform_nearest: torch.nn.Module,
    grid: torch.Tensor,
    cfg: config.TrainingConfig,
    device: torch.device,
    include_pet: bool,
    include_rigidity: bool,
    include_dice: bool = True,
    log_hard_dice: bool = True,
    use_class_weights: bool = False,
    n_integration: int = 7,
    ncc_weight: Optional[float] = None,
    opt_shape: Optional[Tuple[int, int, int]] = None,
    history_csv: Optional[Path] = None,
) -> torch.Tensor:
    # NCC is not scored by the challenge; keep it as a modest dense proxy that
    # fills gradient where label-dice is flat. Try a small value (e.g. 1-3);
    # defaults to cfg.w_ct if not set.
    if ncc_weight is None:
        ncc_weight = cfg.w_ct

    # `opt_shape` (chase_leaderboard) optimises a half-resolution field: the
    # velocity / identity grid / `base` all live at `opt_shape`, and each step
    # the decoded field is upsampled to full res so the loss (dice / NCC / TLG /
    # rigidity, against full-res images and labels) sees exactly what the
    # challenge scorer evaluates after it upsamples the submitted half-res field.
    # The returned field stays at `opt_shape`. opt_shape=None keeps full-res IO.
    full_shape = tuple(cfg.img_shape)
    var_shape = tuple(opt_shape) if opt_shape is not None else full_shape

    def to_full(disp_unit: torch.Tensor) -> torch.Tensor:
        if opt_shape is None:
            return disp_unit
        return torch.nn.functional.interpolate(
            disp_unit, size=full_shape, mode="trilinear", align_corners=False
        )

    identity_vox = synthetic.build_identity_grid(var_shape, device)
    velocity = torch.zeros((1, 3) + var_shape, device=device)
    velocity.requires_grad_(True)
    optimizer = torch.optim.Adam([velocity], lr=cfg.io_lr)

    bone_values = torch.tensor(
        synthetic.BONE_LABEL_VALUES, dtype=torch.float32, device=device
    )
    loss_ncc = NCC(cfg.lvl3_ncc_win)

    # lesion connected components, computed once: the moving mask does not move
    # during IO, only the field does. None when the per-lesion weights are all
    # zero, when the case has no lesion, or when nothing survives the size
    # filter -- the global MTV / TLG terms still cover those.
    pet_cc_masks: Optional[torch.Tensor] = None
    if include_pet and (
        cfg.w_io_mtv_cc > 0.0 or cfg.w_io_tlg_cc > 0.0 or cfg.w_io_mtv_avg_cc > 0.0
    ):
        pet_cc_masks = utils.lesion_component_masks(
            (x_lbl_pet == 1).float(),
            max_components=cfg.io_cc_max_components,
            min_voxels=cfg.io_cc_min_voxels,
        )
        n_cc = 0 if pet_cc_masks is None else pet_cc_masks.shape[1]
        print(f"IO: {n_cc} lesion component(s) under per-lesion tumour terms")

    base = f_x_y.detach()
    # class weights are derived from the starting label overlap, so they are
    # meaningless (and unavailable) without the CT labels
    if use_class_weights and include_dice:
        x_lbl_ct_start = warp_label(
            x_lbl_ct, to_full(base), grid, transform_nearest
        )  # or transform w/ nearest
        class_weights = build_class_weights(
            x_lbl_ct_start, y_lbl_ct, n_labels=118, device=device
        )
    else:
        class_weights = None

    best_loss = float("inf")
    best_disp = base.clone()
    best_disp_i = 0
    history: list[Dict[str, float]] = []
    pbar = tqdm.tqdm(range(cfg.io_it), desc="IO optimization")
    for i in pbar:
        start_time = time.time()
        optimizer.zero_grad()
        disp_unit = svf_to_disp(
            base, velocity, identity_vox, cfg, n_integration, shape=var_shape
        )
        loss, logs = compute_io_loss(
            to_full(disp_unit),
            y,
            x_moving,
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
            include_pet=include_pet,
            include_rigidity=include_rigidity,
            include_dice=include_dice,
            log_hard_dice=log_hard_dice,
            class_weights=class_weights,
            pet_cc_masks=pet_cc_masks,
        )
        loss.backward()
        optimizer.step()

        # step selection: swap the soft squared-MTV contribution for the hard
        # (nearest-counted) one, so best_disp is chosen on the MTV the scorer
        # actually sees rather than the bilinear proxy the optimizer descends
        sel_score = loss.item()
        if not np.isnan(logs["hard_mtv"]):
            sel_score += cfg.w_io_mtv * (logs["hard_mtv"] ** 2 - logs["mtv"] ** 2)
        # same substitution for TLG: swap the soft-mask proxy the optimizer
        # descends for the nearest-mask value the scorer actually reports
        if not np.isnan(logs["hard_tlg"]):
            sel_score += cfg.w_io_tlg * (logs["hard_tlg"] - logs["tlg"])
        if sel_score < best_loss:
            best_loss = sel_score
            best_disp = disp_unit.detach().clone()
            best_disp_i = i

        history.append({"step": i, "loss": loss.item(), "sel_score": sel_score, **logs})

        pbar.set_postfix(
            dice=f"{logs['hard_dice_ct']:.4f}",
            non_diff=f"{logs['jac']:.4f}",
            masked_jac=f"{logs['masked_jac']:.4f}",
            mtv=f"{logs['mtv']:.4f}",
            hard_mtv=f"{logs['hard_mtv']:.4f}",
            mtv_avg=f"{logs['mtv_avg']:.6f}",
            tlg=f"{logs['tlg']:.4f}",
            hard_tlg=f"{logs['hard_tlg']:.4f}",
            mtv_cc=f"{logs['mtv_cc']:.5f}",
            tlg_cc=f"{logs['tlg_cc']:.4f}",
            loss=f"{loss.item():.4f}",
            best=f"{best_loss:.4f}",
            best_i=best_disp_i,
        )

    if False and history_csv is not None and history:
        history_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(history_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    refined = best_disp
    return refined
