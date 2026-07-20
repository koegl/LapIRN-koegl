import contextlib
import os
from typing import Callable, Tuple

import mlflow
import my_data
import numpy as np
import torch
import tqdm
from config import TrainingConfig
from miccai2020_model_stage import (
    SpatialTransform_unit,
)
from torch.utils import data as torch_data


def create_datasets(
    config: TrainingConfig,
) -> Tuple[
    torch_data.ConcatDataset,
    my_data.PSMARegDataset,
    my_data.SyntheticSourceDataset | None,
    my_data.TubingenDataset | None,
    my_data.NLSTDataset | None,
    my_data.PSMARegDataset,
    my_data.TubingenDataset | None,
    my_data.NLSTDataset | None,
    dict,
]:

    train_ids, val_ids = my_data.get_train_val_split(
        data_dir=config.data_dir,
        split_path=config.split_path,
        val_fraction=config.val_fraction,
        tubingen=False,
        nlst=False,
    )
    synth_ids = [
        c
        for c, _ in my_data.list_single_session_sources(
            config.data_dir, exclude_case_ids=train_ids + val_ids
        )
    ]
    train_ids_tubingen, val_ids_tubingen = my_data.get_train_val_split(
        data_dir=config.data_dir,
        split_path=config.split_path,
        val_fraction=config.val_fraction,
        tubingen=True,
        nlst=False,
    )
    train_ids_nlst, val_ids_nlst = my_data.get_train_val_split(
        data_dir=config.data_dir,
        split_path=config.split_path,
        val_fraction=config.val_fraction,
        tubingen=False,
        nlst=True,
    )

    config_to_log = config.to_mlflow_params()
    config_to_log["train_indices"] = train_ids
    config_to_log["train_indices_synthetic"] = synth_ids
    config_to_log["train_indices_tubingen"] = train_ids_tubingen
    config_to_log["train_indices_nlst"] = train_ids_nlst

    config_to_log["val_indices"] = val_ids
    config_to_log["val_indices_tubingen"] = val_ids_tubingen
    config_to_log["val_indices_nlst"] = val_ids_nlst

    val_dataset_tubingen = None
    val_dataset_nlst = None
    train_dataset_synthetic = None
    train_dataset_tubingen = None
    train_dataset_nlst = None

    val_dataset = my_data.PSMARegDataset(
        case_ids=val_ids,
        cfg=config,
        augment=False,
        use_cache=config.use_cache_valid,
        include_intermediate_pairs=False,
        num_workers=config.num_workers,
    )

    train_dataset = my_data.PSMARegDataset(
        case_ids=train_ids,
        cfg=config,
        augment=config.augment,
        use_cache=config.use_cache_train_real,
        include_intermediate_pairs=True,
        num_workers=config.num_workers,
    )
    all_train_datasets = [train_dataset]

    if config.use_synthetic:
        train_dataset_synthetic = my_data.SyntheticSourceDataset(
            cfg=config,
            source_ids=synth_ids,
            repeat=config.synthetic_repeat,
            use_cache=config.use_cache_train_synthetic,
            num_workers=config.num_workers,
            augment=config.augment,
        )
        all_train_datasets.append(train_dataset_synthetic)

    if config.use_tubingen:
        train_dataset_tubingen = my_data.TubingenDataset(
            case_ids=train_ids_tubingen,
            cfg=config,
            augment=config.augment,
            use_cache=config.use_cache_train_real,
            num_workers=config.num_workers,
        )
        all_train_datasets.append(train_dataset_tubingen)

        val_dataset_tubingen = my_data.TubingenDataset(
            case_ids=val_ids_tubingen,
            cfg=config,
            augment=False,
            use_cache=config.use_cache_valid,
            num_workers=config.num_workers,
        )

    if config.use_nlst:
        train_dataset_nlst = my_data.NLSTDataset(
            case_ids=train_ids_nlst,
            cfg=config,
            augment=config.augment,
            use_cache=config.use_cache_train_real,
            num_workers=config.num_workers,
        )
        all_train_datasets.append(train_dataset_nlst)

        val_dataset_nlst = my_data.NLSTDataset(
            case_ids=val_ids_nlst,
            cfg=config,
            augment=False,
            use_cache=config.use_cache_valid,
            num_workers=config.num_workers,
        )

    train_combined = torch_data.ConcatDataset(all_train_datasets)

    return (
        train_combined,
        train_dataset,
        train_dataset_synthetic,
        train_dataset_tubingen,
        train_dataset_nlst,
        val_dataset,
        val_dataset_tubingen,
        val_dataset_nlst,
        config_to_log,
    )


def warmup_lr_factor(global_step: int, warmup_steps: int) -> float:
    """Linear LR warmup factor: ramps 0 -> 1 over the first `warmup_steps`
    training steps, then stays at 1. `warmup_steps <= 0` disables warmup."""
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, (global_step + 1) / warmup_steps)


def apply_warmup_lr(
    optimizer: torch.optim.Optimizer,
    base_lr: float,
    global_step: int,
    warmup_steps: int,
) -> float:
    """Set the optimizer LR to `base_lr` scaled by the linear warmup factor.
    Returns the LR applied (for logging)."""
    lr = base_lr * warmup_lr_factor(global_step, warmup_steps)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def add_jobid_to_mlflow_run() -> None:
    job_id = os.environ.get("SLURM_JOB_ID", "local")

    auto_name = str(mlflow.active_run().info.run_name)
    auto_name_no_id = auto_name.rsplit("-", 1)[0]

    new_name = f"{auto_name_no_id}-{job_id}"

    mlflow.set_tag("mlflow.runName", new_name)
    mlflow.set_tag("slurm_job_id", job_id)


def get_run_name() -> str:

    run_id = mlflow.active_run().info.run_id
    run_name = mlflow.get_run(run_id).info.run_name

    if run_name is None:
        raise ValueError("MLflow run name is None. Please ensure an active run exists.")

    return run_name


def optimizer_step_with_guard(
    loss: torch.Tensor,
    loss_scaled: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    is_step: bool,
    global_step: int,
    level: int,
) -> None:

    if torch.isfinite(loss):
        loss_scaled.backward()

        if is_step:
            # clip_grad_norm_ returns the PRE-clip norm and already scales the
            # applied gradients down to max_norm, so a large pre-clip norm is
            # safe to step on. Only skip on non-finite gradients (NaN/Inf);
            # skipping on magnitude here deadlocks (skipped step -> weights
            # unchanged -> same large norm -> skipped forever).
            total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=5.0
            )
            mlflow.log_metrics(
                {f"lvl{level}/grad_norm": total_norm.item()}, step=global_step
            )
            if not torch.isfinite(total_norm):
                tqdm.tqdm.write(
                    f"[lvl{level}] step {global_step}: non-finite grad_norm "
                    f"loss={loss.item():.4f} (skipped)"
                )
                optimizer.zero_grad()
            else:
                optimizer.step()
                optimizer.zero_grad()
    else:
        tqdm.tqdm.write(f"[lvl{level}] step {global_step}: non-finite loss (skipped)")
        optimizer.zero_grad()


def cycle(loader: torch.utils.data.DataLoader):
    while True:
        for batch in loader:
            yield batch


@contextlib.contextmanager
def track_peak_memory(label: str = ""):
    torch.cuda.reset_peak_memory_stats()
    try:
        yield
    finally:
        allocated = torch.cuda.max_memory_allocated() / 1024**3
        reserved = torch.cuda.max_memory_reserved() / 1024**3
        tag = f"[{label}] " if label else ""
        print(
            f"{tag}Peak allocated: {allocated:.2f} GiB | Peak reserved: {reserved:.2f} GiB"
        )


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    stage: str,
    config: TrainingConfig,
    lossall: np.ndarray,
) -> None:
    """Save a rolling resume checkpoint every config.checkpoint_interval epochs.

    Writes a single "latest" checkpoint per stage, overwritten each time, so
    disk usage stays flat over a long run. It holds model weights, optimizer
    state, current epoch and loss history -- everything needed to resume the
    stage after a wall-time kill. The final per-stage weight file (used to
    gate the next stage) is still saved separately by the train_lvlX function.

    Args:
        model: Model being trained.
        optimizer: Optimizer whose state is saved for resuming.
        epoch: Current epoch index.
        stage: Stage tag, one of "lvl1", "lvl2", "lvl3".
        config: Training configuration; must define checkpoint_interval.
        lossall: Running loss-history array to persist alongside weights.
    """
    if epoch % config.checkpoint_interval != 0:
        return

    config.model_save_dir.mkdir(parents=True, exist_ok=True)
    latest_path = (
        config.model_save_dir / f"{config.mlflow_experiment}_{stage}_latest.pth"
    )
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lossall": lossall,
        },
        latest_path,
    )
    tqdm.tqdm.write(f"[{stage}] checkpoint @ epoch {epoch} -> {latest_path.name}")


def downsample_label(label: torch.Tensor, scale_factor: float) -> torch.Tensor:
    """Downsample integer label map with nearest-neighbour (B, 1, D, H, W)."""
    return torch.nn.functional.interpolate(
        label.float(), scale_factor=scale_factor, mode="nearest"
    ).long()


def soft_dice_loss_binary(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """Soft dice for a single binary channel (B, 1, D, H, W)."""
    intersection = (pred * target).sum()
    cardinality = pred.sum() + target.sum()
    return 1.0 - (2.0 * intersection + eps) / (cardinality + eps)


def soft_dice_loss_multiclass(
    pred_onehot: torch.Tensor,
    target_onehot: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Mean foreground soft dice over classes present in target.

    Args:
        pred_onehot: (B, C, D, H, W) float, warped one-hot.
        target_onehot: (B, C, D, H, W) float, fixed one-hot.
        eps: Smoothing term.

    Returns:
        Scalar 1 - mean_foreground_dice.
    """
    # sum over spatial dims and batch: (C,)
    dims = (0, 2, 3, 4)
    intersection = (pred_onehot * target_onehot).sum(dim=dims)
    cardinality = pred_onehot.sum(dim=dims) + target_onehot.sum(dim=dims)
    dice_per_class = (2.0 * intersection + eps) / (cardinality + eps)
    # exclude background (class 0)
    return 1.0 - dice_per_class[1:].mean()


def to_one_hot(label: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert integer label volume to one-hot float tensor.

    Args:
        label: (B, 1, D, H, W) integer tensor.
        num_classes: Number of classes including background.

    Returns:
        (B, C, D, H, W) float tensor.
    """
    B, _, D, H, W = label.shape
    one_hot = torch.zeros(
        B, num_classes, D, H, W, device=label.device, dtype=torch.float32
    )
    return one_hot.scatter_(1, label.long(), 1.0)


def dice_loss_with_grad(
    moving_label: torch.Tensor,
    fixed_label: torch.Tensor,
    disp: torch.Tensor,
    grid: torch.Tensor,
    transform: SpatialTransform_unit,
    eps: float = 1e-5,
    return_each: bool = False,
) -> torch.Tensor | None:
    """Per-class soft dice loss with gradients flowing through disp.

    Builds one-hot only for classes present in fixed_label, warps each
    moving one-hot channel with bilinear interpolation, then computes
    dice per class and averages. Robust to variable class sets across subjects.

    Args:
        moving_label: (B, 1, D, H, W) integer tensor, moving labels.
        fixed_label: (B, 1, D, H, W) integer tensor, fixed labels.
        disp: (B, 3, D, H, W) displacement field (gradients flow through this).
        grid: level grid for SpatialTransform_unit.
        transform: SpatialTransform_unit instance.
        eps: Smoothing term.

    Returns:
        Scalar 1 - mean_foreground_dice.
    """

    classes = fixed_label.unique()
    classes = classes[classes != 0]  # exclude background

    if classes.numel() == 0:
        return None

    flow = disp.permute(0, 2, 3, 4, 1)  # (B, D, H, W, 3)

    dice_scores = []
    for c in classes:
        moving_c = (moving_label == c).float()
        fixed_c = (fixed_label == c).float()

        # skip if either label is empty: a class present in only one image
        # (e.g. cropped out of the moving by augmentation) yields dice~0 with
        # no usable registration gradient and only biases the mean.
        if fixed_c.sum() == 0 or moving_c.sum() == 0:
            continue

        warped_c = transform(moving_c, flow, grid)

        intersection = (warped_c * fixed_c).sum()
        cardinality = warped_c.sum() + fixed_c.sum()

        dice_scores.append((2.0 * intersection + eps) / (cardinality + eps))

    # every class was skipped (no overlapping labels present)
    if len(dice_scores) == 0:
        return None

    if return_each:
        return 1.0 - torch.stack(dice_scores)
    else:
        return 1.0 - torch.stack(dice_scores).mean()


def mtv_bias_loss(
    warped_pet_mask: torch.Tensor,
    moving_pet_mask: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """MTV bias: relative volume change of lesion mask after warping.

    Args:
        warped_pet_mask: (B, 1, D, H, W) float, warped moving lesion mask.
        moving_pet_mask: (B, 1, D, H, W) float, original moving lesion mask.
        eps: Smoothing term.

    Returns:
        Scalar absolute relative volume bias.
    """
    mtv_moving = moving_pet_mask.sum()
    mtv_warped = warped_pet_mask.sum()
    return torch.abs(mtv_warped - mtv_moving) / (mtv_moving + eps)


def tlg_bias_loss(
    warped_pet_image: torch.Tensor,
    warped_pet_mask: torch.Tensor,
    moving_pet_image: torch.Tensor,
    moving_pet_mask: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """TLG bias: relative change in lesion-weighted PET intensity after warping.

    Args:
        warped_pet_image: (B, 1, D, H, W) float, warped moving PET image.
        warped_pet_mask: (B, 1, D, H, W) float, warped moving lesion mask.
        moving_pet_image: (B, 1, D, H, W) float, original moving PET image.
        moving_pet_mask: (B, 1, D, H, W) float, original moving lesion mask.
        eps: Smoothing term.

    Returns:
        Scalar absolute relative TLG bias.
    """
    tlg_moving = (moving_pet_image * moving_pet_mask).sum()
    tlg_warped = (warped_pet_image * warped_pet_mask).sum()
    return torch.abs(tlg_warped - tlg_moving) / (tlg_moving + eps)


def masked_jac_det_loss(
    jac_det: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Penalise deviation of det(J) from 1 inside mask.

    Args:
        jac_det: (B, 1, D, H, W) Jacobian determinant field.
        mask: (B, 1, D, H, W) binary float mask (tumor region).
        eps: Smoothing term.

    Returns:
        Scalar mean squared deviation from 1 inside mask.
    """
    masked_det = jac_det * mask
    target = mask  # det(J)=1 inside mask
    return ((masked_det - target) ** 2).sum() / (mask.sum() + eps)


def orthonormality_loss(
    jac: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Penalise deviation of the Jacobian from orthonormality inside mask.

    Rigid motions have an orthonormal Jacobian (J^T J = I), which forbids local
    shear and anisotropic scaling. This penalises ||J^T J - I||_F^2 per voxel.
    Consumes the interior Jacobian returned by ``jacobian.jacobian_matrix``;
    boundary voxels contribute 0.

    Args:
        jac: (B, 3, 3, D-2, H-2, W-2) Jacobian J = I + du/dx, indexed [:, row, col].
        mask: (B, 1, D, H, W) binary float mask (e.g. bones).
        eps: Smoothing term.

    Returns:
        Scalar mean orthonormality error inside mask.
    """
    # (J^T J)_ij = sum_k jac[:, k, i] * jac[:, k, j]; jac indexed [b, row, col, d, h, w]
    jtj = torch.einsum("bkidhw,bkjdhw->bijdhw", jac, jac)
    eye = torch.eye(3, device=jac.device, dtype=jac.dtype).view(1, 3, 3, 1, 1, 1)
    m = jtj - eye
    err = (m * m).sum(dim=(1, 2))  # ||J^T J - I||_F^2, (B, D-2, H-2, W-2)

    err = torch.nn.functional.pad(err, (1, 1, 1, 1, 1, 1), value=0.0).unsqueeze(1)
    return (err * mask).sum() / (mask.sum() + eps)


def affine_loss(
    flow: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Penalise deviation from local affine-ness inside mask.

    A displacement field is affine iff all its second-order derivatives vanish,
    i.e. the Jacobian is constant over the region. This penalises the 3D bending
    energy ||d^2 u||^2 (all second partials, including mixed) per voxel, forcing
    the masked region to share a single transform. Combined with
    ``orthonormality_loss`` and ``masked_jac_det_loss`` this yields local
    rigidity. Boundary voxels contribute 0.

    Args:
        flow: (B, D, H, W, 3) displacement field in voxel units.
        mask: (B, 1, D, H, W) binary float mask (e.g. bones).
        eps: Smoothing term.

    Returns:
        Scalar mean bending energy inside mask.
    """
    energy = 0.0
    for c in range(3):
        f = flow[..., c]
        center = f[:, 1:-1, 1:-1, 1:-1]
        # pure second derivatives
        fxx = f[:, 2:, 1:-1, 1:-1] - 2 * center + f[:, :-2, 1:-1, 1:-1]
        fyy = f[:, 1:-1, 2:, 1:-1] - 2 * center + f[:, 1:-1, :-2, 1:-1]
        fzz = f[:, 1:-1, 1:-1, 2:] - 2 * center + f[:, 1:-1, 1:-1, :-2]
        # mixed second derivatives
        fxy = (
            f[:, 2:, 2:, 1:-1]
            - f[:, 2:, :-2, 1:-1]
            - f[:, :-2, 2:, 1:-1]
            + f[:, :-2, :-2, 1:-1]
        ) / 4
        fxz = (
            f[:, 2:, 1:-1, 2:]
            - f[:, 2:, 1:-1, :-2]
            - f[:, :-2, 1:-1, 2:]
            + f[:, :-2, 1:-1, :-2]
        ) / 4
        fyz = (
            f[:, 1:-1, 2:, 2:]
            - f[:, 1:-1, 2:, :-2]
            - f[:, 1:-1, :-2, 2:]
            + f[:, 1:-1, :-2, :-2]
        ) / 4
        energy = energy + (
            fxx * fxx + fyy * fyy + fzz * fzz + 2 * (fxy * fxy + fxz * fxz + fyz * fyz)
        )  # (B, D-2, H-2, W-2)

    energy = torch.nn.functional.pad(energy, (1, 1, 1, 1, 1, 1), value=0.0).unsqueeze(1)
    return (energy * mask).sum() / (mask.sum() + eps)


def enforce_rigidity_loss(
    jac_det: torch.Tensor,
    jac: torch.Tensor,
    flow: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-5,
):
    """Combined local-rigidity loss over a masked region (e.g. bones).

    Enforces the three Staring/Modersitzki conditions (summed with equal weight;
    scale the whole term with ``config.w_bone_rigidity`` at the call site):
      * properness    -- det(J) = 1   (``masked_jac_det_loss``)
      * orthonormality -- J^T J = I    (``orthonormality_loss``)
      * affine-ness    -- d^2 u = 0    (``affine_loss``)

    Args:
        jac_det: (B, 1, D, H, W) Jacobian determinant field.
        jac: (B, 3, 3, D-2, H-2, W-2) Jacobian from ``jacobian.jacobian_matrix``.
        flow: (B, D, H, W, 3) displacement field in voxel units.
        mask: (B, 1, D, H, W) binary float mask.
        eps: Smoothing term.

    Returns:
        (total, (det, ortho, affine)) scalar tensors.
    """
    det = masked_jac_det_loss(jac_det, mask, eps)
    ortho = orthonormality_loss(jac, mask, eps)
    affine = affine_loss(flow, mask, eps)
    total = det + ortho + affine
    return total, (det, ortho, affine)


def warp_binary_mask(
    mask: torch.Tensor,
    disp: torch.Tensor,
    grid: torch.Tensor,
    transform: SpatialTransform_unit,
) -> torch.Tensor:
    """Warp a binary mask with bilinear interpolation (differentiable).

    Args:
        mask: (B, 1, D, H, W) float binary mask.
        disp: (B, 3, D, H, W) displacement field.
        grid: level grid for SpatialTransform_unit.
        transform: SpatialTransform_unit instance.

    Returns:
        (B, 1, D, H, W) warped mask, values in [0, 1].
    """
    flow = disp.permute(0, 2, 3, 4, 1)
    return transform(mask.float(), flow, grid)


def affine_pet_iou(
    x_lbl_pet: torch.Tensor,
    y_lbl_pet: torch.Tensor,
    flow_affine: torch.Tensor,
    grid_full: torch.Tensor,
    transform_nearest: Callable,
    tumour_label: int = 1,
) -> float:
    """IoU of the affine-aligned moving PET mask vs fixed PET mask."""
    x_mask = (x_lbl_pet == tumour_label).float()
    x_mask_aff = transform_nearest(x_mask, flow_affine, grid_full)
    x_bin = x_mask_aff > 0.5
    y_bin = y_lbl_pet == tumour_label

    intersection = (x_bin & y_bin).sum().float()
    union = (x_bin | y_bin).sum().float()
    iou = (intersection / (union + 1e-8)).item()
    return iou
