import torch
from miccai2020_model_stage import (
    SpatialTransform_unit,
)


def downsample_label(label: torch.Tensor, scale_factor: float) -> torch.Tensor:
    """Downsample integer label map with nearest-neighbour (B, 1, D, H, W)."""
    return torch.nn.functional.interpolate(
        label.float(), scale_factor=scale_factor, mode="nearest"
    ).long()


def compute_ndv(jac_det: torch.Tensor) -> float:
    """Compute Non-Diffeomorphic Volume Percentage.

    Args:
        jac_det: (B, 1, D, H, W) Jacobian determinant field.

    Returns:
        Percentage of voxels with det(J) <= 0, as float in [0, 100].
    """
    with torch.no_grad():
        ndv = (jac_det <= 0).float().mean().item() * 100
    return ndv


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
) -> torch.Tensor:
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
        return torch.tensor(0.0, device=disp.device)

    flow = disp.permute(0, 2, 3, 4, 1)  # (B, D, H, W, 3)

    dice_scores = []
    for c in classes:
        moving_c = (moving_label == c).float()
        fixed_c = (fixed_label == c).float()

        # skip if both are empty (matches challenge convention)
        if fixed_c.sum() == 0 and moving_c.sum() == 0:
            continue

        warped_c = transform(moving_c, flow, grid)

        intersection = (warped_c * fixed_c).sum()
        cardinality = warped_c.sum() + fixed_c.sum()

        # if fixed is empty but moving is not (or vice versa), dice = 0
        if fixed_c.sum() == 0:
            dice_scores.append(torch.tensor(0.0, device=disp.device))
            continue

        dice_scores.append((2.0 * intersection + eps) / (cardinality + eps))

    return 1.0 - torch.stack(dice_scores).mean()
