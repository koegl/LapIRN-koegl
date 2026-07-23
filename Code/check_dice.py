"""Equivalence check: chunked dice vs the original per-class loop.

Run on the cluster:  python check_dice.py
"""

import torch
import utils  # noqa: E402
from miccai2020_model_stage import SpatialTransform_unit  # noqa: E402


def reference(
    moving_label, fixed_label, disp, grid, transform, eps=1e-5, class_weights=None
):
    """The original implementation, with the empty-class skip."""
    classes = fixed_label.unique()
    classes = classes[classes != 0]
    if classes.numel() == 0:
        return None
    flow = disp.permute(0, 2, 3, 4, 1)
    dice_scores, weights = [], []
    for c in classes:
        moving_c = (moving_label == c).float()
        fixed_c = (fixed_label == c).float()
        if fixed_c.sum() == 0 or moving_c.sum() == 0:
            continue
        warped_c = transform(moving_c, flow, grid)
        intersection = (warped_c * fixed_c).sum()
        cardinality = warped_c.sum() + fixed_c.sum()
        dice_scores.append((2.0 * intersection + eps) / (cardinality + eps))
        if class_weights is not None:
            weights.append(class_weights[int(c.item())])
    if len(dice_scores) == 0:
        return None
    dice_stack = torch.stack(dice_scores)
    if class_weights is not None:
        w = torch.stack(weights)
        w = w / (w.mean() + eps)
        return 1.0 - (w * dice_stack).sum() / w.sum()
    return 1.0 - dice_stack.mean()


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shape = (1, 1, 24, 28, 32)
    n_labels = 40

    moving = torch.randint(0, n_labels, shape, device=device).float()
    fixed = torch.randint(0, n_labels, shape, device=device).float()
    # make a few labels present in fixed but absent from moving, to exercise
    # the skip path
    moving[moving == 7] = 0
    moving[moving == 13] = 0

    transform = SpatialTransform_unit().to(device)
    d0, d1, d2 = shape[2:]
    lin = [torch.linspace(-1, 1, s, device=device) for s in (d0, d1, d2)]
    g0, g1, g2 = torch.meshgrid(*lin, indexing="ij")
    grid = torch.stack([g2, g1, g0], dim=-1).unsqueeze(0)

    for use_w in (False, True):
        cw = torch.rand(n_labels, device=device) if use_w else None

        disp_a = (0.02 * torch.randn((1, 3) + shape[2:], device=device)).requires_grad_(
            True
        )
        disp_b = disp_a.detach().clone().requires_grad_(True)

        ref = reference(moving, fixed, disp_a, grid, transform, class_weights=cw)
        new = utils.dice_loss_with_grad(
            moving, fixed, disp_b, grid, transform, class_weights=cw, chunk_size=7
        )

        ref.backward()
        new.backward()

        dv = (ref - new).abs().item()
        dg = (disp_a.grad - disp_b.grad).abs().max().item()
        print(
            f"class_weights={use_w}: value={ref.item():.8f} vs {new.item():.8f} "
            f"(|d|={dv:.3e})  max|dgrad|={dg:.3e}"
        )
        assert dv < 1e-5, "value mismatch"
        assert dg < 1e-5, "gradient mismatch"

    print("OK - chunked dice matches the reference in value and gradient")


if __name__ == "__main__":
    main()
