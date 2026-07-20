"""Sanity-check NCC_fast / multi_resolution_NCC_fast against the reference
NCC / multi_resolution_NCC: agreement in value, in gradient, and speedup.

Run:  python Code/check_ncc_fast.py
"""

from pathlib import Path
from time import time
from typing import Tuple

import numpy as np
import torch
from config import TrainingConfig
from miccai2020_model_stage import (
    NCC,
    NCC_fast,
    multi_resolution_NCC,
    multi_resolution_NCC_fast,
)


def make_inputs(
    shape: Tuple[int, int, int],
    device: torch.device,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Smooth, correlated volumes — closer to real CT than white noise, which
    has near-zero local variance and hides normalisation bugs."""
    torch.manual_seed(seed)
    base = torch.rand(1, 1, *shape, device=device)
    base = torch.nn.functional.avg_pool3d(base, 5, stride=1, padding=2)
    I = base + 0.1 * torch.rand(1, 1, *shape, device=device)
    J = base.roll(shifts=2, dims=2) + 0.1 * torch.rand(1, 1, *shape, device=device)
    return I.contiguous(), J.contiguous()


def load_real_pair(
    fixed_path: Path, moving_path: Path, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """A real preprocessed pair — mostly air, which is the regime where the
    single-pass variance formula is weakest."""
    import my_data

    def load(path: Path) -> torch.Tensor:
        arr = my_data.nib.load(str(path)).get_fdata().astype(np.float32)
        arr = my_data.norm_ct(arr)
        return torch.from_numpy(arr)[None, None].to(device).float()

    return load(moving_path), load(fixed_path)


def report_pair(I: torch.Tensor, J: torch.Tensor, win: int, label: str) -> None:
    print(f"\n=== {label} ===")
    frac_air = (I < 0.01).float().mean().item()
    print(f"shape={tuple(I.shape[2:])}  fraction of near-zero voxels={frac_air:.3f}")

    ref = multi_resolution_NCC(win=win, scale=3)(I, J).item()
    fast = multi_resolution_NCC_fast(win=win, scale=3).loss(I, J).item()
    rel = abs(ref - fast) / max(abs(ref), 1e-12)
    print(f"loss   ref={ref:+.8f}  fast={fast:+.8f}  rel_diff={rel:.3e}")

    a = I.clone().requires_grad_(True)
    multi_resolution_NCC(win=win, scale=3)(a, J).backward()
    g_ref = a.grad.detach()

    b = I.clone().requires_grad_(True)
    multi_resolution_NCC_fast(win=win, scale=3).loss(b, J).backward()
    g_fast = b.grad.detach()

    cos = torch.nn.functional.cosine_similarity(
        g_ref.flatten(), g_fast.flatten(), dim=0
    ).item()
    ratio = (g_fast.abs().max() / g_ref.abs().max().clamp(min=1e-12)).item()
    print(f"grad   cosine={cos:.8f}  max|g_fast|/max|g_ref|={ratio:.4f}")


def compare_values(shape: Tuple[int, int, int], device: torch.device) -> None:
    print(f"\n=== single-scale NCC vs NCC_fast  (shape={shape}) ===")
    I, J = make_inputs(shape, device)
    for win in (5, 7, 9):
        ref = NCC(win=win)(I, J).item()
        fast = NCC_fast(win=win).loss(I, J).item()
        rel = abs(ref - fast) / max(abs(ref), 1e-12)
        print(f"win={win}:  ref={ref:+.8f}  fast={fast:+.8f}  rel_diff={rel:.3e}")


def compare_multires(
    shape: Tuple[int, int, int], win: int, device: torch.device
) -> None:
    print(f"\n=== multi_resolution (win={win}, scale=3, shape={shape}) ===")
    I, J = make_inputs(shape, device)
    ref = multi_resolution_NCC(win=win, scale=3)(I, J).item()
    fast = multi_resolution_NCC_fast(win=win, scale=3).loss(I, J).item()
    rel = abs(ref - fast) / max(abs(ref), 1e-12)
    print(f"ref={ref:+.8f}  fast={fast:+.8f}  rel_diff={rel:.3e}")


def compare_grads(shape: Tuple[int, int, int], win: int, device: torch.device) -> None:
    """Values matching is not enough — the loss is only useful via its gradient."""
    print(f"\n=== gradient wrt moving image (win={win}, scale=3) ===")
    I, J = make_inputs(shape, device)

    a = I.clone().requires_grad_(True)
    multi_resolution_NCC(win=win, scale=3)(a, J).backward()
    g_ref = a.grad.detach()

    b = I.clone().requires_grad_(True)
    multi_resolution_NCC_fast(win=win, scale=3).loss(b, J).backward()
    g_fast = b.grad.detach()

    denom = g_ref.abs().max().clamp(min=1e-12)
    max_abs = (g_ref - g_fast).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(
        g_ref.flatten(), g_fast.flatten(), dim=0
    ).item()
    print(f"max|dg|={max_abs:.3e}  (rel to max|g_ref|={denom.item():.3e})")
    print(f"cosine similarity={cos:.8f}")


def benchmark(
    shape: Tuple[int, int, int],
    win: int,
    device: torch.device,
    n_iter: int = 20,
) -> None:
    print(f"\n=== timing (win={win}, scale=3, shape={shape}, n={n_iter}) ===")
    I, J = make_inputs(shape, device)
    I = I.requires_grad_(True)

    ref = multi_resolution_NCC(win=win, scale=3)
    fast = multi_resolution_NCC_fast(win=win, scale=3)

    def run(fn, backward: bool) -> float:
        for _ in range(3):  # warmup
            out = fn(I, J)
            if backward:
                out.backward()
                I.grad = None
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time()
        for _ in range(n_iter):
            out = fn(I, J)
            if backward:
                out.backward()
                I.grad = None
        if device.type == "cuda":
            torch.cuda.synchronize()
        return (time() - start) / n_iter

    for backward in (False, True):
        tag = "fwd+bwd" if backward else "fwd"
        t_ref = run(lambda a, b: ref(a, b), backward)
        t_fast = run(lambda a, b: fast.loss(a, b), backward)
        print(
            f"{tag:8s}  ref={t_ref * 1e3:8.2f} ms   "
            f"fast={t_fast * 1e3:8.2f} ms   speedup={t_ref / t_fast:.2f}x"
        )


def main() -> None:
    cfg = TrainingConfig()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  lvl3_ncc_win={cfg.lvl3_ncc_win}")

    small = (48, 48, 64)
    full = tuple(int(s) for s in cfg.img_shape)

    compare_values(small, device)
    compare_multires(small, cfg.lvl3_ncc_win, device)
    compare_grads(small, cfg.lvl3_ncc_win, device)

    # the one that counts: a real, mostly-air NLST pair at full resolution
    fixed = Path(
        "/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/imagesTr/PSMARegPSMA_0002_0000_00.nii.gz"
    )
    moving = Path(
        "/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/imagesTr/PSMARegPSMA_0002_0001_00.nii.gz"
    )
    if fixed.exists() and moving.exists():
        I, J = load_real_pair(fixed, moving, device)
        report_pair(I, J, cfg.lvl3_ncc_win, "real NLST pair (full res)")
    else:
        print(f"\nskipping real-data check, not found: {fixed}")

    if device.type == "cuda":
        benchmark(full, cfg.lvl3_ncc_win, device)
    else:
        print("\nskipping full-resolution benchmark on cpu")

    print(
        "\nnote: the reference NCC clamps I_var/J_var to >=0 and cc to <=1;\n"
        "NCC_fast does neither, so exact agreement is not expected. Large\n"
        "gradient differences (cosine << 1) mean the clamps are actually\n"
        "firing on your data and the fast version is not a drop-in swap."
    )


if __name__ == "__main__":
    main()
