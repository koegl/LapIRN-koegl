"""Compare the fast NCC against the reference at every level used in training:

    lvl1  NCC              win=lvl1_ncc_win            img_shape_4
    lvl2  multi_res scale2 win=lvl2_ncc_win            img_shape_2
    lvl3  multi_res scale3 win=lvl3_ncc_win            img_shape

Reports value agreement, gradient agreement and speedup per level.

Run:  python Code/check_ncc_fast.py
"""

from pathlib import Path
from time import time
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from config import TrainingConfig
from miccai2020_model_stage import (
    NCC,
    NCC_fast,
    multi_resolution_NCC,
    multi_resolution_NCC_fast,
)

# a real preprocessed pair; mostly air, which is where the single-pass
# variance formula is weakest. falls back to synthetic data if missing.
real_fixed = Path(
    "/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/imagesTr/PSMARegPSMA_0002_0000_00.nii.gz"
)
real_moving = Path(
    "/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/imagesTr/PSMARegPSMA_0002_0001_00.nii.gz"
)

n_iter = 20


class LevelSpec:
    def __init__(
        self,
        name: str,
        shape: Tuple[int, int, int],
        win: int,
        scale: Optional[int],
    ) -> None:
        self.name = name
        self.shape = shape
        self.win = win
        # scale None -> single-scale NCC / NCC_fast, as level1 uses
        self.scale = scale

    def build(self) -> Tuple[torch.nn.Module, torch.nn.Module]:
        if self.scale is None:
            return NCC(win=self.win), NCC_fast(win=self.win)
        return (
            multi_resolution_NCC(win=self.win, scale=self.scale),
            multi_resolution_NCC_fast(win=self.win, scale=self.scale),
        )

    def describe(self) -> str:
        kind = "NCC" if self.scale is None else f"multi_res scale={self.scale}"
        return f"{self.name}: {kind}, win={self.win}, shape={self.shape}"


def make_synthetic(
    shape: Tuple[int, int, int], device: torch.device, seed: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    base = torch.rand(1, 1, *shape, device=device)
    base = F.avg_pool3d(base, 5, stride=1, padding=2)
    I = base + 0.1 * torch.rand(1, 1, *shape, device=device)
    J = base.roll(shifts=2, dims=2) + 0.1 * torch.rand(1, 1, *shape, device=device)
    return I.contiguous(), J.contiguous()


def load_real_pair(
    device: torch.device,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if not (real_fixed.exists() and real_moving.exists()):
        return None

    import my_data

    def load(path: Path) -> torch.Tensor:
        arr = my_data.nib.load(str(path)).get_fdata().astype(np.float32)
        arr = my_data.norm_ct(arr)
        return torch.from_numpy(arr)[None, None].to(device).float()

    return load(real_moving), load(real_fixed)


def to_shape(vol: torch.Tensor, shape: Tuple[int, int, int]) -> torch.Tensor:
    if tuple(vol.shape[2:]) == shape:
        return vol
    return F.interpolate(vol, size=shape, mode="trilinear", align_corners=False)


def compare(
    ref: torch.nn.Module,
    fast: torch.nn.Module,
    I: torch.Tensor,
    J: torch.Tensor,
) -> None:
    v_ref = ref(I, J).item()
    v_fast = fast(I, J).item()
    rel = abs(v_ref - v_fast) / max(abs(v_ref), 1e-12)
    print(f"  loss   ref={v_ref:+.8f}  fast={v_fast:+.8f}  rel_diff={rel:.3e}")

    a = I.clone().requires_grad_(True)
    ref(a, J).backward()
    g_ref = a.grad.detach()

    b = I.clone().requires_grad_(True)
    fast(b, J).backward()
    g_fast = b.grad.detach()

    cos = F.cosine_similarity(g_ref.flatten(), g_fast.flatten(), dim=0).item()
    ratio = (g_fast.abs().max() / g_ref.abs().max().clamp(min=1e-12)).item()
    print(f"  grad   cosine={cos:.8f}  max|g_fast|/max|g_ref|={ratio:.4f}")


def time_fn(
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    I: torch.Tensor,
    J: torch.Tensor,
    backward: bool,
    device: torch.device,
) -> float:
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


def benchmark(
    ref: torch.nn.Module,
    fast: torch.nn.Module,
    I: torch.Tensor,
    J: torch.Tensor,
    device: torch.device,
) -> None:
    I = I.clone().requires_grad_(True)
    for backward in (False, True):
        tag = "fwd+bwd" if backward else "fwd"
        t_ref = time_fn(ref, I, J, backward, device)
        t_fast = time_fn(fast, I, J, backward, device)
        print(
            f"  {tag:8s} ref={t_ref * 1e3:8.2f} ms  "
            f"fast={t_fast * 1e3:8.2f} ms  speedup={t_ref / t_fast:.2f}x"
        )


def main() -> None:
    cfg = TrainingConfig()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    levels: List[LevelSpec] = [
        LevelSpec(
            "lvl1", tuple(int(s) for s in cfg.img_shape_4), cfg.lvl1_ncc_win, None
        ),
        LevelSpec("lvl2", tuple(int(s) for s in cfg.img_shape_2), cfg.lvl2_ncc_win, 2),
        LevelSpec("lvl3", tuple(int(s) for s in cfg.img_shape), cfg.lvl3_ncc_win, 3),
    ]

    real = load_real_pair(device)
    source = "real pair" if real is not None else "synthetic"
    print(f"device={device}  data={source}  n_iter={n_iter}")
    if real is None:
        print(f"  (real pair not found at {real_fixed})")

    for spec in levels:
        print(f"\n=== {spec.describe()} ===")

        if real is not None:
            I = to_shape(real[0], spec.shape)
            J = to_shape(real[1], spec.shape)
        else:
            I, J = make_synthetic(spec.shape, device)
        print(f"  fraction of near-zero voxels={(I < 0.01).float().mean().item():.3f}")

        ref, fast = spec.build()
        ref = ref.to(device)
        fast = fast.to(device)

        compare(ref, fast, I, J)

        if device.type == "cuda":
            benchmark(ref, fast, I, J, device)
        else:
            print("  skipping timing on cpu")


if __name__ == "__main__":
    main()
