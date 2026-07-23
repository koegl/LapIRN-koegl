"""Register two CTs with the trained LapIRN model.

Minimal version of inference.py: affine prereg + model inference only
(no polyaffine, no instance optimisation, no metrics/csvs). PET channels are
filled with zeros because the model expects CT+PET input.

Saves the warped moving CT and the warped moving labels in the fixed image
space.
"""

from pathlib import Path
from typing import Dict, Tuple

import affine_reg
import Functions
import miccai2020_model_stage
import my_data
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from config import TrainingConfig

# TotalSegmentator ribs: rib_left_1..12 and rib_right_1..12
RIB_LABEL_IDS: Tuple[int, ...] = tuple(range(92, 116))


class SpatialTransformer(torch.nn.Module):
    """
    N-D Spatial Transformer
    Obtained from https://github.com/voxelmorph/voxelmorph
    """

    def __init__(self, size, mode="bilinear"):
        super().__init__()
        self.mode = mode
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(*vectors, indexing="ij")
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        self.register_buffer("grid", grid)

    def forward(self, src: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        new_locs = self.grid + flow
        shape = flow.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]
        return F.grid_sample(src, new_locs, align_corners=False, mode=self.mode)


def create_model(
    device: torch.device, cfg: TrainingConfig, model_path: Path
) -> torch.nn.Module:
    model_lvl1 = miccai2020_model_stage.Miccai2020_LDR_laplacian_unit_add_lvl1(
        in_channel=cfg.in_channel,
        n_classes=cfg.n_classes,
        start_channel=cfg.start_channel,
        is_train=True,
        imgshape=cfg.img_shape_4,
        range_flow=cfg.range_flow,
    ).to(device)
    model_lvl2 = miccai2020_model_stage.Miccai2020_LDR_laplacian_unit_add_lvl2(
        in_channel=cfg.in_channel,
        n_classes=cfg.n_classes,
        start_channel=cfg.start_channel,
        is_train=True,
        imgshape=cfg.img_shape_2,
        range_flow=cfg.range_flow,
        model_lvl1=model_lvl1,
    ).to(device)
    model = miccai2020_model_stage.Miccai2020_LDR_laplacian_unit_add_lvl3(
        in_channel=cfg.in_channel,
        n_classes=cfg.n_classes,
        start_channel=cfg.start_channel,
        is_train=True,
        imgshape=cfg.img_shape,
        range_flow=cfg.range_flow,
        model_lvl2=model_lvl2,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model


def build_input(
    ct_path: Path,
    device: torch.device,
) -> Tuple[torch.Tensor, np.ndarray]:
    """Build the 2-channel (CT, dummy zero PET) network input for one image.

    Returns the (1, 2, H, W, D) tensor and the raw (unnormalised, bed-removed)
    CT array, so the warped CT can be saved in HU.
    """
    ct_raw = nib.load(str(ct_path)).get_fdata().astype(np.float32)

    mask = my_data.get_body_mask(ct_raw)
    ct_raw = my_data.apply_body_mask(
        ct_raw, mask, fill_value=float(np.percentile(ct_raw, 0.5))
    )

    # dummy PET: all zeros, same grid as the CT
    pet_raw = np.zeros_like(ct_raw)

    ct = torch.from_numpy(my_data.norm_ct(ct_raw)).unsqueeze(0)
    pet = torch.from_numpy(my_data.norm_pet(pet_raw)).unsqueeze(0)
    volume = torch.cat([ct, pet], dim=0).unsqueeze(0).to(device).float()

    return volume, ct_raw


def load_labels(label_path: Path, device: torch.device) -> torch.Tensor:
    arr = nib.load(str(label_path)).get_fdata().astype(np.int16)
    labels = torch.from_numpy(arr)[None, None].to(device).float()
    return labels


def save_like(
    arr: np.ndarray,
    reference_path: Path,
    out_path: Path,
    dtype,
) -> None:
    """Save an array using the header/affine of a reference image."""
    ref = nib.load(str(reference_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(arr.astype(dtype), ref.affine, ref.header)
    img.set_data_dtype(dtype)
    nib.save(img, str(out_path))
    print(f"saved → {out_path}")


def dice_per_label(
    pred: torch.Tensor,
    target: torch.Tensor,
    label_ids: range = range(1, 118),
) -> Dict[int, float]:
    """Per-label hard dice. Labels absent from both pred and target get nan so
    they drop out of the means."""
    scores: Dict[int, float] = {}
    for lbl in label_ids:
        p = pred == lbl
        t = target == lbl
        volume_sum = p.sum() + t.sum()
        if volume_sum == 0:
            scores[lbl] = float("nan")
        else:
            scores[lbl] = (2.0 * (p & t).sum() / volume_sum).item()
    return scores


def save_dice_csv(
    scores: Dict[int, float],
    out_path: Path,
    rib_ids: Tuple[int, ...] = RIB_LABEL_IDS,
) -> None:
    """label,dice csv: mean over all present labels first, per-label rows next,
    mean over the rib labels last."""
    present = [v for v in scores.values() if not np.isnan(v)]
    rib_present = [
        scores[lbl] for lbl in rib_ids if lbl in scores and not np.isnan(scores[lbl])
    ]

    rows = [("mean", float(np.mean(present)) if present else float("nan"))]
    rows += [(str(lbl), scores[lbl]) for lbl in sorted(scores)]
    rows += [("rib_mean", float(np.mean(rib_present)) if rib_present else float("nan"))]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["label", "dice"]).to_csv(out_path, index=False)
    print(f"saved → {out_path}")


def multilabel_dice(
    pred: torch.Tensor,
    target: torch.Tensor,
    label_ids: range = range(1, 118),
) -> float:
    """Mean Dice over a fixed label set. pred/target: (H, W, D) int."""
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


def register(
    fixed_ct_path: Path,
    fixed_label_path: Path,
    moving_ct_path: Path,
    moving_label_path: Path,
    out_dir: Path,
    model_path: Path,
    cfg: TrainingConfig,
    device: torch.device,
) -> None:
    model = create_model(device, cfg, model_path)

    grid_full = Functions.generate_grid_unit(cfg.img_shape)
    grid_full = (
        torch.from_numpy(np.reshape(grid_full, (1,) + grid_full.shape))
        .to(device)
        .float()
    )
    transform = miccai2020_model_stage.SpatialTransform_unit().to(device)

    X, moving_ct_raw = build_input(moving_ct_path, device)
    Y, _ = build_input(fixed_ct_path, device)

    for name, vol in (("moving", X), ("fixed", Y)):
        if tuple(vol.shape[2:]) != tuple(cfg.img_shape):
            raise ValueError(
                f"{name} image has shape {tuple(vol.shape[2:])}, "
                f"expected {tuple(cfg.img_shape)}"
            )

    moving_labels = load_labels(moving_label_path, device)
    fixed_labels = load_labels(fixed_label_path, device)

    # --- affine preregistration ---
    dvf = affine_reg.compute_affine_dvf(
        fixed_ct_path=fixed_ct_path,
        moving_ct_path=moving_ct_path,
        make_lowres_ants_image_fn=affine_reg.make_lowres_ants_image,
        preprocess_ct_fn=affine_reg.preprocess_ct,
        ants_affine_to_fullres_voxel_disp_fn=affine_reg.ants_affine_to_fullres_voxel_disp,
    )
    dvf_tensor = affine_reg.dvf_to_tensor(dvf, device)
    h, w, d = cfg.img_shape
    d_h = dvf_tensor[:, 0] / (h / 2.0)
    d_w = dvf_tensor[:, 1] / (w / 2.0)
    d_d = dvf_tensor[:, 2] / (d / 2.0)
    flow_affine = torch.stack([d_d, d_w, d_h], dim=1).permute(0, 2, 3, 4, 1)

    X_affine = transform(X, flow_affine, grid_full)

    # --- model inference ---
    with torch.no_grad():
        F_X_Y, _, _, _, _, _, _ = model(X_affine, Y)

    # --- compose affine + deformable into one full-resolution field ---
    deform_grid = grid_full + F_X_Y.permute(0, 2, 3, 4, 1)
    affine_grid = grid_full + flow_affine

    composed_grid = torch.nn.functional.grid_sample(
        affine_grid.permute(0, 4, 1, 2, 3),
        deform_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    ).permute(0, 2, 3, 4, 1)

    total_unit_flow = (composed_grid - grid_full).permute(0, 4, 1, 2, 3)
    total_voxel = Functions.transform_unit_flow_to_flow_cuda(
        total_unit_flow.permute(0, 2, 3, 4, 1).clone()
    ).permute(0, 4, 1, 2, 3)

    # voxelmorph-style transformer expects (dx, dy, dz) in image axis order
    disp = total_voxel.flip(1)

    st_linear = SpatialTransformer(size=cfg.img_shape, mode="bilinear").to(device)
    st_nearest = SpatialTransformer(size=cfg.img_shape, mode="nearest").to(device)

    moving_ct_hu = torch.from_numpy(moving_ct_raw)[None, None].to(device).float()
    warped_ct = st_linear(moving_ct_hu, disp)
    warped_labels = st_nearest(moving_labels, disp)

    dice_before = multilabel_dice(
        moving_labels[0, 0].round().long(), fixed_labels[0, 0].round().long()
    )
    dice_after = multilabel_dice(
        warped_labels[0, 0].round().long(), fixed_labels[0, 0].round().long()
    )
    print(f"dice before={dice_before:.4f};\tafter={dice_after:.4f}")

    scores = dice_per_label(
        warped_labels[0, 0].round().long(), fixed_labels[0, 0].round().long()
    )
    save_dice_csv(scores, out_dir / "dice_per_label.csv")

    save_like(
        warped_ct[0, 0].detach().cpu().numpy(),
        reference_path=fixed_ct_path,
        out_path=out_dir / "warped_moving_ct.nii.gz",
        dtype=np.float32,
    )
    save_like(
        warped_labels[0, 0].round().detach().cpu().numpy(),
        reference_path=fixed_label_path,
        out_path=out_dir / "warped_moving_labels.nii.gz",
        dtype=np.int16,
    )


def main() -> None:
    # --- variables (define here, no argparse) ---
    fixed_ct_path = Path(
        "/home/iml/fryderyk.koegl/code/LapIRN-koegl/lisa/preprocessed/sub-0L4e4-f1XHo_ses-20180604_sequ-6_acq-cor_ce-ContrastAgent_part-axial_ct.nii.gz"
    )
    fixed_label_path = Path(
        "/home/iml/fryderyk.koegl/code/LapIRN-koegl/lisa/preprocessed/sub-0L4e4-f1XHo_ses-20180604_sequ-6_acq-cor_ce-ContrastAgent_part-axial_ct_label.nii.gz"
    )
    moving_ct_path = Path(
        "/home/iml/fryderyk.koegl/code/LapIRN-koegl/lisa/preprocessed/sub-0L4e4-f1XHo_ses-20180919_sequ-6_acq-cor_ce-ContrastAgent_part-axial_ct.nii.gz"
    )
    moving_label_path = Path(
        "/home/iml/fryderyk.koegl/code/LapIRN-koegl/lisa/preprocessed/sub-0L4e4-f1XHo_ses-20180919_sequ-6_acq-cor_ce-ContrastAgent_part-axial_ct_label.nii.gz"
    )

    model_ori_name = "exultant-hawk-38756587"
    model_path = Path(
        f"/home/iml/fryderyk.koegl/data/PSMAReg/models/PSMAReg_LapIRN_{model_ori_name}_stagelvl3_best.pth"
    )
    out_dir = Path("/home/iml/fryderyk.koegl/code/LapIRN-koegl/lisa/registered")

    cfg = TrainingConfig()
    cfg.in_channel = 4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    register(
        fixed_ct_path=fixed_ct_path,
        fixed_label_path=fixed_label_path,
        moving_ct_path=moving_ct_path,
        moving_label_path=moving_label_path,
        out_dir=out_dir,
        model_path=model_path,
        cfg=cfg,
        device=device,
    )


if __name__ == "__main__":
    main()
