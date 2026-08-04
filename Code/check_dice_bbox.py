"""Compare `dice_loss_with_grad` against the bbox-cropped
`dice_loss_with_grad_bbox` (with and without checkpointing): peak memory, memory
retained for backward, runtime, and numerical agreement of both the loss and its
gradient w.r.t. the field.

No model and no dataset class are instantiated - a real label pair is read
straight off disk with nibabel, and the displacement is a smooth random field.
Falls back to synthetic ellipsoid blobs with --synthetic or when the files are
missing.

The interesting number is `retained`: the tensors autograd keeps alive from the
end of the forward pass until backward. That is what stacks up on top of the
rest of the lvl3 graph and what nvidia-smi shows as a step change around the
call.

Two control rows keep the agreement check honest. `margin=20` widens the crop:
if the crop extent were the cause of a disagreement, this would move it.
`ref, +1 ulp` runs the reference against itself with the displacement nudged by
one float32 ulp, which sets the noise floor - grid_sample's gradient is
piecewise constant in the sampling coordinate, so samples sitting within an ulp
of a cell boundary flip their local gradient under any change of coordinate
path. Variants are judged against that floor, not against zero.

Run:  python Code/check_dice_bbox.py
      python Code/check_dice_bbox.py --case 0006 --tp-fixed 00 --tp-moving 01
      python Code/check_dice_bbox.py --synthetic --shape 192 192 288
"""

import argparse
from pathlib import Path
from time import perf_counter
from typing import Callable, Dict, List, Tuple

import nibabel as nib
import numpy as np
import torch
import utils
from config import TrainingConfig
from Functions import generate_grid_unit

MiB = 1024.0**2


def load_label(path: Path, device: torch.device) -> torch.Tensor:
    """(1, 1, D, H, W) label tensor, loaded exactly as my_data.load_lbl does."""
    arr = nib.load(path).get_fdata().astype(np.uint8)
    return torch.from_numpy(arr).to(device)[None, None]


def make_label_map(
    shape: Tuple[int, int, int],
    n_labels: int,
    device: torch.device,
    seed: int = 0,
) -> torch.Tensor:
    """A (1, 1, D, H, W) int16 label map of `n_labels` overlapping ellipsoids.

    Radii are drawn per axis so the blobs range from compact (vertebra-like) to
    elongated (rib-like). Later labels overwrite earlier ones, which packs the
    volume the way real segmentations do.
    """
    d, h, w = shape
    gen = torch.Generator().manual_seed(seed)
    label_map = torch.zeros((d, h, w), dtype=torch.int16, device=device)

    for c in range(1, n_labels + 1):
        radii = torch.randint(4, 23, (3,), generator=gen).tolist()
        center = [
            torch.randint(r, dim - r, (1,), generator=gen).item()
            for r, dim in zip(radii, (d, h, w))
        ]

        lo = [max(0, ci - ri) for ci, ri in zip(center, radii)]
        hi = [min(dim, ci + ri + 1) for ci, ri, dim in zip(center, radii, (d, h, w))]

        axes = []
        for i in range(3):
            coord = torch.arange(lo[i], hi[i], device=device, dtype=torch.float32)
            axes.append(((coord - center[i]) / radii[i]) ** 2)

        dist = axes[0].view(-1, 1, 1) + axes[1].view(1, -1, 1) + axes[2].view(1, 1, -1)
        blob = dist <= 1.0

        window = label_map[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
        window[blob] = c

    return label_map.view(1, 1, d, h, w)


def make_disp(
    shape: Tuple[int, int, int],
    device: torch.device,
    max_vox: float,
    seed: int = 0,
) -> torch.Tensor:
    """A (1, 3, D, H, W) smooth field in normalized units, scaled so the largest
    displacement is about `max_vox` voxels along its own axis."""
    d, h, w = shape
    gen = torch.Generator().manual_seed(seed)
    coarse = torch.randn(
        (1, 3, max(d // 16, 2), max(h // 16, 2), max(w // 16, 2)), generator=gen
    ).to(device)
    field = torch.nn.functional.interpolate(
        coarse, size=(d, h, w), mode="trilinear", align_corners=True
    )
    field = field / field.abs().max()

    # component k of a grid_sample grid indexes spatial axis 2 - k, and a
    # normalized span of 2 covers N voxels
    per_axis = torch.tensor(
        [2.0 * max_vox / w, 2.0 * max_vox / h, 2.0 * max_vox / d], device=device
    )
    return field * per_axis.view(1, 3, 1, 1, 1)


def grad_metrics(
    reference: torch.Tensor, other: torch.Tensor
) -> Tuple[float, float, float, int]:
    """(max abs diff, relative L2, 1 - cosine, count of voxels over 1% of max).

    The max is reported but is a poor accuracy measure here: grid_sample's
    gradient is piecewise constant in the sampling coordinate and jumps at cell
    boundaries, so a sample sitting within an ulp of a boundary can land in a
    different cell in two mathematically equivalent coordinate paths and flip
    its whole local gradient. The norms say whether that is happening at a
    handful of voxels or everywhere.
    """
    diff = reference - other
    scale = reference.abs().max()
    max_abs = diff.abs().max().item()
    rel_l2 = (diff.norm() / (reference.norm() + 1e-30)).item()
    cos = torch.nn.functional.cosine_similarity(
        reference.flatten(), other.flatten(), dim=0
    ).item()
    n_big = int((diff.abs() > 0.01 * scale).sum().item())
    return max_abs, rel_l2, 1.0 - cos, n_big


def run_one(
    fn: Callable,
    moving: torch.Tensor,
    fixed: torch.Tensor,
    disp_raw: torch.Tensor,
    grid: torch.Tensor,
    cuda: bool,
    disp_scale: float = 1.0,
) -> Dict[str, object]:
    disp = (disp_raw * disp_scale).clone().requires_grad_(True)

    if cuda:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
    else:
        base = 0

    start = perf_counter()
    per_class = fn(moving, fixed, disp, grid, None, return_each=True)
    if cuda:
        torch.cuda.synchronize()
    forward_s = perf_counter() - start

    if cuda:
        forward_peak = torch.cuda.max_memory_allocated() - base
        retained = torch.cuda.memory_allocated() - base
    else:
        forward_peak = retained = 0

    loss = per_class.mean()

    start = perf_counter()
    loss.backward()
    if cuda:
        torch.cuda.synchronize()
    backward_s = perf_counter() - start

    total_peak = (torch.cuda.max_memory_allocated() - base) if cuda else 0

    return {
        "per_class": per_class.detach(),
        "loss": loss.item(),
        "grad": disp.grad.detach().clone(),
        "forward_peak": forward_peak,
        "retained": retained,
        "total_peak": total_peak,
        "forward_s": forward_s,
        "backward_s": backward_s,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="0006")
    parser.add_argument("--tp-fixed", default="00")
    parser.add_argument("--tp-moving", default="01")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--synthetic", action="store_true", help="use blob labels instead of real ones"
    )
    parser.add_argument("--shape", type=int, nargs=3, default=None)
    parser.add_argument("--n-labels", type=int, default=117)
    parser.add_argument(
        "--max-disp-vox",
        type=float,
        default=6.0,
        help="largest displacement in the field, in voxels",
    )
    parser.add_argument("--shift", type=int, default=4, help="synthetic pair offset")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")

    cfg = TrainingConfig()
    data_dir = args.data_dir if args.data_dir is not None else cfg.data_dir
    label_dir = Path(data_dir) / "labelsTr"
    fixed_path = label_dir / f"PSMARegPSMA_{args.case}_0000_{args.tp_fixed}.nii.gz"
    moving_path = label_dir / f"PSMARegPSMA_{args.case}_0000_{args.tp_moving}.nii.gz"

    use_real = not args.synthetic and fixed_path.exists() and moving_path.exists()

    print(f"device      : {device}")
    if use_real:
        print(f"source      : real labels, case {args.case}")
        print(f"  fixed     : {fixed_path.name}")
        print(f"  moving    : {moving_path.name}")
        # no affine pre-registration here, so the two timepoints sit further
        # apart than in training: union bboxes are wider and the measured
        # saving is a lower bound on what lvl3 sees
        print("  (raw timepoints, no affine prereg -> conservative bboxes)")
        fixed = load_label(fixed_path, device)
        moving = load_label(moving_path, device)
        shape = tuple(fixed.shape[-3:])
    else:
        if not args.synthetic:
            print(f"[!] {label_dir} not readable, falling back to synthetic labels")
        if args.shape is not None:
            shape = tuple(args.shape)
        elif cuda:
            shape = (192, 192, 288)
        else:
            shape = (48, 48, 72)  # CPU: correctness only, keep it quick
        print(f"source      : synthetic blobs, {args.n_labels} labels")
        fixed = make_label_map(shape, args.n_labels, device, seed=args.seed)
        # a rigid shift gives the pair the overlap (and the slightly different
        # bounding boxes) a real pair has after affine pre-registration
        moving = torch.roll(
            fixed, shifts=(args.shift, args.shift, -args.shift), dims=(2, 3, 4)
        )

    n_shared = torch.isin(fixed.unique(), moving.unique()).sum().item() - 1
    print(f"shape       : {shape}  ({np.prod(shape) / 1e6:.1f} M voxels)")
    print(f"labels      : {n_shared} present in both")
    print(f"occupancy   : {(fixed != 0).float().mean().item() * 100:.1f}% labelled")
    print(f"max disp    : {args.max_disp_vox} vox")
    if not cuda:
        print("\n[!] no CUDA: numerical agreement only, memory numbers are 0\n")

    disp_raw = make_disp(shape, device, args.max_disp_vox, seed=args.seed)

    grid = generate_grid_unit(shape)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).to(device).float()

    def bbox_ckpt(*call_args, **call_kwargs):
        return utils.dice_loss_with_grad_bbox(
            *call_args, use_checkpoint=True, **call_kwargs
        )

    def bbox_margin(margin: int) -> Callable:
        def call(*call_args, **call_kwargs):
            return utils.dice_loss_with_grad_bbox(
                *call_args, extra_margin=margin, **call_kwargs
            )

        return call

    # (name, fn, disp scale, include in the memory table)
    # The last two are controls, not candidates:
    #   margin=20  - if the crop extent were too tight, widening it would move
    #                the numbers. If it doesn't, the crop is not the cause.
    #   +1 ulp     - the reference against itself with the displacement nudged
    #                by one float32 ulp. This is the noise floor: no variant can
    #                be expected to agree more tightly than the loss agrees with
    #                itself under last-bit input noise.
    ULP = 2.0**-23
    entries: List[Tuple[str, Callable, float, bool]] = [
        ("dice_loss_with_grad", utils.dice_loss_with_grad, 1.0, True),
        ("bbox", utils.dice_loss_with_grad_bbox, 1.0, True),
        ("bbox + checkpoint", bbox_ckpt, 1.0, True),
        ("[control] margin=8", bbox_margin(8), 1.0, False),
        ("[control] margin=20", bbox_margin(20), 1.0, False),
        ("[control] margin=60", bbox_margin(60), 1.0, False),
        ("[control] margin=150", bbox_margin(150), 1.0, False),
        ("[control] ref, +1 ulp", utils.dice_loss_with_grad, 1.0 + ULP, False),
    ]

    results: Dict[str, Dict[str, object]] = {}
    for name, fn, scale, _ in entries:
        results[name] = run_one(
            fn, moving, fixed, disp_raw, grid, cuda, disp_scale=scale
        )

    variants = [(name, fn) for name, fn, _, show in entries if show]

    header = (
        f"{'variant':<26}{'fwd peak':>12}{'retained':>12}"
        f"{'fwd+bwd':>12}{'fwd s':>9}{'bwd s':>9}"
    )
    print("\n" + header)
    print("-" * len(header))
    for name, _ in variants:
        r = results[name]
        print(
            f"{name:<26}"
            f"{r['forward_peak'] / MiB:>11.0f}M"
            f"{r['retained'] / MiB:>11.0f}M"
            f"{r['total_peak'] / MiB:>11.0f}M"
            f"{r['forward_s']:>9.3f}"
            f"{r['backward_s']:>9.3f}"
        )

    base_name = entries[0][0]
    base = results[base_name]
    grad_scale = base["grad"].abs().max().item()
    n_grad = base["grad"].numel()

    print(f"\nagreement vs {base_name}   (max |grad| = {grad_scale:.3e})")
    print("-" * 92)
    print(
        f"{'variant':<24}{'per-class':>12}{'mean loss':>12}"
        f"{'grad max':>12}{'grad relL2':>12}{'1 - cos':>12}{'>1% of max':>12}"
    )

    metrics = {}
    for name, _, _, _ in entries[1:]:
        other = results[name]
        n_a, n_b = base["per_class"].numel(), other["per_class"].numel()
        per_class_diff = (
            (base["per_class"] - other["per_class"]).abs().max().item()
            if n_a == n_b
            else float("nan")
        )
        loss_diff = abs(base["loss"] - other["loss"])
        max_abs, rel_l2, one_minus_cos, n_big = grad_metrics(
            base["grad"], other["grad"]
        )
        metrics[name] = (n_a == n_b, loss_diff, max_abs, rel_l2, n_big)
        print(
            f"{name:<24}{per_class_diff:>12.2e}{loss_diff:>12.2e}"
            f"{max_abs:>12.2e}{rel_l2:>12.2e}{one_minus_cos:>12.2e}"
            f"{f'{n_big}/{n_grad}':>12}"
        )

    if cuda:
        print("\nretained memory")
        print("-" * 92)
        for name, _ in variants:
            r = results[name]
            factor = base["retained"] / r["retained"] if r["retained"] > 0 else 0.0
            print(f"{name:<24}{r['retained'] / MiB:>9.0f} MiB{factor:>9.1f}x")

    # the +1 ulp control sets the bar: a variant is fine if it disagrees with
    # the reference no more than the reference disagrees with itself under
    # last-bit input noise
    floor_max = metrics["[control] ref, +1 ulp"][2]
    floor_l2 = metrics["[control] ref, +1 ulp"][3]
    print(
        f"\nnoise floor (ref vs ref +1 ulp): grad max {floor_max:.2e}, "
        f"relL2 {floor_l2:.2e}"
    )

    all_ok = True
    for name, _ in variants[1:]:
        same_classes, loss_diff, max_abs, rel_l2, _ = metrics[name]
        ok = (
            same_classes
            and loss_diff < 1e-5
            and max_abs <= max(3.0 * floor_max, 1e-9)
            and rel_l2 <= max(3.0 * floor_l2, 1e-9)
        )
        all_ok &= ok
        print(f"  {name:<24} {'ok' if ok else 'OFF'}")

    print("\n" + ("PASS: all variants within the noise floor" if all_ok else "FAIL"))


if __name__ == "__main__":
    main()
