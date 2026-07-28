"""Parity check for the unified IO objective.

Guards the invariant that the training-time unrolled inner loss
(`unrolled_io_loss`, all term groups on) equals the deploy-time IO loss
(`compute_io_loss`) term-for-term, so the net is seeded for exactly the
trajectory that `run_io` descends at test time. The only difference between the
two is the hard-Dice log (computed under no_grad, never added to the loss), so
the scalar losses must match bit-for-bit given identical inputs.

Run:
    python3 Code/test_io_objective_parity.py
or under pytest:
    pytest Code/test_io_objective_parity.py
"""

import numpy as np
import torch

import config as config_mod
import instance_opt
import synthetic
from Functions import generate_grid_unit
from miccai2020_model_stage import (
    NCC,
    SpatialTransform_unit,
    SpatialTransformNearest_unit,
)


def _build_dummy_batch(img_shape, device, seed=0):
    """A small, structured synthetic case: a displacement field, moving/fixed
    two-channel (CT, PET) images, and label maps that contain a bone label and a
    PET-tumor label so that every term (dice, PET/TLG, tumor-jac, rigidity)
    actually contributes a non-trivial value."""
    g = torch.Generator(device=device).manual_seed(seed)
    H, W, D = img_shape

    # small smooth-ish displacement in unit-flow coordinates, needs grad because
    # the real objective is differentiated w.r.t. it
    disp_unit = 0.05 * torch.randn((1, 3, H, W, D), generator=g, device=device)
    disp_unit.requires_grad_(True)

    # moving image: channel 0 = CT, channel 1 = PET
    x_moving = torch.rand((1, 2, H, W, D), generator=g, device=device)
    y = torch.rand((1, 2, H, W, D), generator=g, device=device)

    # label maps. Seed a known bone value and a PET-tumor (==1) region so the
    # rigidity / PET terms are exercised rather than short-circuiting to zero.
    bone_val = float(synthetic.BONE_LABEL_VALUES[0])
    x_lbl_ct = torch.randint(0, 6, (1, 1, H, W, D), generator=g, device=device).float()
    y_lbl_ct = torch.randint(0, 6, (1, 1, H, W, D), generator=g, device=device).float()
    x_lbl_ct[..., : H // 3, :, :] = bone_val
    y_lbl_ct[..., : H // 3, :, :] = bone_val

    x_lbl_pet = torch.zeros((1, 1, H, W, D), device=device)
    x_lbl_pet[..., : H // 4, : W // 4, : D // 4] = 1.0

    return disp_unit, x_moving, y, x_lbl_ct, x_lbl_pet, y_lbl_ct


def run_parity(class_weights=None, seed=0):
    device = torch.device("cpu")
    cfg = config_mod.TrainingConfig()
    # shrink the volume so the check runs in a second on CPU; even dims keep the
    # half-res SVF machinery happy even though we feed disp_unit directly here.
    cfg.img_shape = (24, 24, 24)

    disp_unit, x_moving, y, x_lbl_ct, x_lbl_pet, y_lbl_ct = _build_dummy_batch(
        cfg.img_shape, device, seed=seed
    )

    transform = SpatialTransform_unit().to(device)
    transform_nearest = SpatialTransformNearest_unit().to(device)
    grid = (
        torch.from_numpy(
            np.reshape(generate_grid_unit(cfg.img_shape), (1,) + cfg.img_shape + (3,))
        )
        .to(device)
        .float()
    )
    loss_ncc = NCC(cfg.lvl3_ncc_win)
    bone_values = torch.tensor(
        synthetic.BONE_LABEL_VALUES, dtype=torch.float32, device=device
    )
    ncc_weight = cfg.w_ct

    # deploy-time objective (what run_io descends)
    loss_deploy, logs = instance_opt.compute_io_loss(
        disp_unit,
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
        class_weights=class_weights,
    )

    # training-time unrolled inner loss, all term groups on
    loss_unroll = instance_opt.unrolled_io_loss(
        disp_unit,
        y[:, 0:1],
        x_moving,
        x_lbl_ct,
        y_lbl_ct,
        transform,
        grid,
        cfg,
        loss_ncc,
        ncc_weight=ncc_weight,
        class_weights=class_weights,
        x_lbl_pet=x_lbl_pet,
        bone_values=bone_values,
        include_pet=True,
        include_rigidity=True,
    )

    return loss_deploy, loss_unroll, logs


def test_io_objective_parity():
    # both the default (no class weights, matching real inference) and the
    # class-weighted path must agree
    for label, class_weights in [
        ("class_weights=None", None),
        ("class_weights=set", torch.linspace(0.5, 1.5, 118)),
    ]:
        loss_deploy, loss_unroll, _ = run_parity(class_weights=class_weights)
        assert torch.allclose(loss_deploy, loss_unroll, rtol=0, atol=1e-6), (
            f"{label}: deploy {loss_deploy.item():.8f} != unroll {loss_unroll.item():.8f}"
        )


def test_ct_only_subset_differs():
    """Sanity: turning the PET/rigidity groups off must actually change the loss
    (otherwise the flags would be silently inert)."""
    device = torch.device("cpu")
    cfg = config_mod.TrainingConfig()
    cfg.img_shape = (24, 24, 24)
    disp_unit, x_moving, y, x_lbl_ct, x_lbl_pet, y_lbl_ct = _build_dummy_batch(
        cfg.img_shape, device
    )
    transform = SpatialTransform_unit().to(device)
    grid = (
        torch.from_numpy(
            np.reshape(generate_grid_unit(cfg.img_shape), (1,) + cfg.img_shape + (3,))
        )
        .to(device)
        .float()
    )
    loss_ncc = NCC(cfg.lvl3_ncc_win)
    bone_values = torch.tensor(
        synthetic.BONE_LABEL_VALUES, dtype=torch.float32, device=device
    )
    common = dict(
        transform=transform,
        grid=grid,
        cfg=cfg,
        loss_ncc=loss_ncc,
        ncc_weight=cfg.w_ct,
        x_lbl_pet=x_lbl_pet,
        bone_values=bone_values,
    )
    full = instance_opt.unrolled_io_loss(
        disp_unit, y[:, 0:1], x_moving, x_lbl_ct, y_lbl_ct,
        include_pet=True, include_rigidity=True, **common,
    )
    ct_only = instance_opt.unrolled_io_loss(
        disp_unit, y[:, 0:1], x_moving, x_lbl_ct, y_lbl_ct,
        include_pet=False, include_rigidity=False, **common,
    )
    assert not torch.allclose(full, ct_only), (
        "PET/rigidity flags had no effect on the loss"
    )


if __name__ == "__main__":
    for label, cw in [
        ("class_weights=None", None),
        ("class_weights=set", torch.linspace(0.5, 1.5, 118)),
    ]:
        loss_deploy, loss_unroll, logs = run_parity(class_weights=cw)
        diff = (loss_deploy - loss_unroll).abs().item()
        status = "OK" if diff < 1e-6 else "MISMATCH"
        print(
            f"[{status}] {label:22s} deploy={loss_deploy.item():.8f} "
            f"unroll={loss_unroll.item():.8f} |diff|={diff:.2e}"
        )
    print(
        "logs (deploy):",
        {k: round(v, 5) for k, v in logs.items()},
    )
    test_io_objective_parity()
    test_ct_only_subset_differs()
    print("all parity assertions passed")
