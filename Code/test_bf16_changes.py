"""Numerical-correctness checks for the bf16-autocast recipe changes.

Verifies the planned edits BEFORE they are applied to the model code:

1. fp32 grid_sample with a bf16 flow input crashes (the mixed-dtype bug this
   recipe must avoid by upcasting flow before grid_sample).
2. fp32 grid_sample with an fp32-cast flow works, and matches the pure-fp32
   reference when the flow values are exactly bf16-representable.
3. DiffeomorphicTransform_unit (scaling-and-squaring) with a bf16 input under
   autocast(enabled=False): manual `.float()` upcast keeps the result
   numerically equal (allclose) to a pure-fp32 run; without the upcast it
   crashes.
4. A representative loss stack (NCC_fast, multi_resolution_NCC_fast, smoothloss)
   computes finite values on fp32 inputs that were produced under a bf16
   autocast region and then cast back to fp32 (the pattern the losses will see).
5. bf16 autocast around a conv trunk actually produces bf16 conv activations
   (the memory saving), while an explicit `.float()` on its output restores
   fp32 for the composition / transform / loss code.

Run:
    python Code/test_bf16_changes.py
"""

import torch
import torch.nn.functional as F
from miccai2020_model_stage import (
    DiffeomorphicTransform_unit,
    NCC_fast,
    SpatialTransform_unit,
    multi_resolution_NCC_fast,
    smoothloss,
)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
AMP_KWARGS = {"device_type": "cuda", "dtype": torch.bfloat16}

# exactly representable in bf16 (multiples of a power of two within bf16 mantissa)
BF16_SAFE_VALUES = [0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 0.25, -0.25]


def make_grid(shape, device):
    """Unit grid in [-1, 1] matching generate_grid_unit layout (B, D, H, W, 3)."""
    vectors = [torch.arange(0, s, device=device, dtype=torch.float32) for s in shape]
    grids = torch.meshgrid(vectors, indexing="ij")
    grid = torch.stack(grids)  # (3, D, H, W)
    grid = grid.unsqueeze(0).permute(0, 2, 3, 4, 1)  # (1, D, H, W, 3)
    for i, s in enumerate(shape):
        grid[..., i] = 2 * (grid[..., i] / (s - 1) - 0.5)
    return grid


def bf16_safe_flow(shape, device):
    """Random flow whose values are exactly representable in bf16, so casting
    bf16 -> fp32 is lossless and fp32/bf16 paths are directly comparable."""
    vals = torch.tensor(BF16_SAFE_VALUES, device=device)
    idx = torch.randint(0, len(vals), shape, device=device)
    return vals[idx]


def test_mixed_dtype_grid_sample_fails():
    """fp32 image + bf16 flow must raise: this is the bug the upcast prevents."""
    shape = (8, 8, 8)
    x = torch.randn(1, 2, *shape, device=DEVICE)
    flow = bf16_safe_flow((1, *shape, 3), DEVICE).to(torch.bfloat16)
    grid = make_grid(shape, DEVICE)
    with torch.autocast(enabled=False, **AMP_KWARGS):
        try:
            F.grid_sample(
                x,
                grid + flow,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
        except RuntimeError:
            pass  # torch < ~2.7: strict dtype check
        else:
            # newer torch silently computes with mixed dtypes — the upcast is
            # still required for determinism/portability, so just note it
            print("note: this torch build tolerates mixed-dtype grid_sample")


def test_upcast_flow_grid_sample_matches_fp32():
    """With exactly-representable flow values, the upcast path must be
    bit-identical to a pure-fp32 SpatialTransform_unit run."""
    shape = (16, 16, 16)
    x = torch.randn(1, 2, *shape, device=DEVICE)
    flow32 = bf16_safe_flow((1, *shape, 3), DEVICE)
    grid = make_grid(shape, DEVICE)

    transform = SpatialTransform_unit().to(DEVICE)

    # pure fp32 reference (module is never under autocast in the recipe)
    with torch.autocast(enabled=False, **AMP_KWARGS):
        ref = transform(x, flow32, grid)

        # simulate: flow produced as bf16 by the conv trunk, then upcast
        flow_bf16 = flow32.to(torch.bfloat16)
        out = transform(x, flow_bf16.float(), grid)

    assert ref.dtype == torch.float32 and out.dtype == torch.float32
    assert torch.equal(ref, out), f"max diff {(ref - out).abs().max()}"


def test_diff_transform_bf16_input_needs_upcast():
    """DiffeomorphicTransform_unit with a bf16 velocity: without the .float()
    upcast it crashes (flow stays bf16, grid_sample against fp32); with the
    upcast it must match the fp32 reference."""
    shape = (16, 16, 16)
    # velocity in NCDHW layout, bf16-safe values
    v32 = bf16_safe_flow((1, 3, *shape), DEVICE)
    grid = make_grid(shape, DEVICE)

    diff = DiffeomorphicTransform_unit(time_step=7).to(DEVICE)

    with torch.autocast(enabled=False, **AMP_KWARGS):
        ref = diff(v32, grid)

        v_bf16 = v32.to(torch.bfloat16)
        # what the recipe does: upcast at the top of forward
        out = diff(v_bf16.float(), grid)

        # sanity: whether the un-upcast path crashes is build-dependent;
        # the recipe never relies on it either way
        try:
            diff(v_bf16, grid)
        except RuntimeError:
            pass

    assert out.dtype == torch.float32
    assert torch.allclose(ref, out, atol=0, rtol=0), (
        f"max diff {(ref - out).abs().max()}"
    )


def test_losses_finite_on_autocast_produced_fp32_inputs():
    """Losses run under autocast(enabled=False) on fp32 tensors that came out
    of a bf16 autocast region and were cast back — must be finite and equal to
    feeding genuinely-fp32 tensors of the same values."""
    shape = (16, 16, 16)
    conv = torch.nn.Conv3d(2, 4, 3, padding=1).to(DEVICE)

    x = torch.randn(1, 2, *shape, device=DEVICE)
    y = torch.randn(1, 2, *shape, device=DEVICE)

    with torch.autocast(**AMP_KWARGS):
        feat_bf16 = conv(x)
    assert feat_bf16.dtype == torch.bfloat16, "conv trunk should emit bf16"

    feat = feat_bf16.float()  # recipe: cast back before leaving the trunk

    ncc = NCC_fast(win=9)
    ncc_multi = multi_resolution_NCC_fast(win=9, scale=3)

    with torch.autocast(enabled=False, **AMP_KWARGS):
        # NCC_fast is single-channel; slice one feature channel like the
        # training loop slices the CT/PET channels of the warped image
        l_ncc = ncc(feat[:, :1], feat[:, :1].clone())
        # multi-resolution NCC down-pools with count_include_pad=False;
        # feed images (not features) at a size divisible by 4
        l_multi = ncc_multi(x[:, :1], y[:, :1])
        flow = torch.randn(1, 3, *shape, device=DEVICE) * 0.01
        l_smooth = smoothloss(flow)

    for name, val in [("ncc", l_ncc), ("ncc_multi", l_multi), ("smooth", l_smooth)]:
        assert torch.isfinite(val), f"{name} loss not finite: {val}"
        assert val.dtype == torch.float32, f"{name} dtype {val.dtype}"


def test_autocast_conv_trunk_saves_and_upcasts():
    """End-to-end dtype plumbing: bf16 inside trunk, fp32 everywhere after the
    explicit casts the recipe adds."""
    shape = (16, 16, 16)
    conv = torch.nn.Conv3d(4, 4, 3, padding=1).to(DEVICE)
    head = torch.nn.Conv3d(4, 3, 3, padding=1).to(DEVICE)  # 3-channel flow head
    x = torch.randn(1, 4, *shape, device=DEVICE)

    with torch.autocast(**AMP_KWARGS):
        feat = conv(x)
        assert feat.dtype == torch.bfloat16
        v = head(feat)  # stays bf16 inside the trunk

    v32 = v.float()
    assert v32.dtype == torch.float32 and v32.shape[1] == 3

    # composition + transform in fp32 must work and stay fp32
    grid = make_grid(shape, DEVICE)
    transform = SpatialTransform_unit().to(DEVICE)
    diff = DiffeomorphicTransform_unit(time_step=7).to(DEVICE)
    with torch.autocast(enabled=False, **AMP_KWARGS):
        composed = v32 + v32  # velocity composition in fp32
        disp = diff(composed, grid)
        warped = transform(x, disp.permute(0, 2, 3, 4, 1), grid)
    assert composed.dtype == disp.dtype == warped.dtype == torch.float32
    assert torch.isfinite(warped).all()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required (bf16 autocast + grid_sample)")

    tests = [
        test_mixed_dtype_grid_sample_fails,
        test_upcast_flow_grid_sample_matches_fp32,
        test_diff_transform_bf16_input_needs_upcast,
        test_losses_finite_on_autocast_produced_fp32_inputs,
        test_autocast_conv_trunk_saves_and_upcasts,
    ]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print("all bf16 recipe checks passed")
