from pathlib import Path

import config
import my_data
import nibabel as nib
import numpy as np
import synthetic
import torch
from Functions import generate_grid_unit
from miccai2020_model_stage import SpatialTransform_unit


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = config.TrainingConfig()  # your config
    out_dir = Path("tmp/synthetic_dataset_check")
    out_dir.mkdir(parents=True, exist_ok=True)

    bone_label_values = ...  # your 60 values
    synth_ids = ["0002"]  # a few single-session case ids to test

    ds = my_data.SyntheticSourceDataset(cfg=cfg, source_ids=synth_ids)
    item = ds[0]

    y = item["y"][None].to(device).float()
    y_lbl_ct = item["y_label_ct"][None].to(device)
    y_lbl_pet = item["y_label_pet"][None].to(device)

    moving, m_lbl_ct, m_lbl_pet, gt_unit = synthetic.generate_synthetic_moving(
        source=y,
        source_label_ct=y_lbl_ct,
        source_label_pet=y_lbl_pet,
        bone_label_values=synthetic.BONE_LABEL_VALUES,
        device=device,
    )

    # --- warp moving back to fixed using gt_unit through the REAL transform ----
    transform = SpatialTransform_unit().to(device)
    for param in transform.parameters():
        param.requires_grad = False

    shape = (y.shape[2], y.shape[3], y.shape[4])
    grid_full = generate_grid_unit(shape)
    grid_full = (
        torch.from_numpy(np.reshape(grid_full, (1,) + grid_full.shape))
        .to(device)
        .float()
    )

    recon = transform(moving, gt_unit.permute(0, 2, 3, 4, 1), grid_full)

    diff_moving = (moving - y).abs().mean().item()
    diff_recon = (recon - y).abs().mean().item()
    print(f"moving vs fixed  (should be > 0):        {diff_moving:.4f}")
    print(f"recon  vs fixed  (should be small ~0.0): {diff_recon:.4f}")

    # --- save CT channel of each for Slicer ------------------------------------
    affine = np.eye(4)
    for name, vol in [("fixed", y), ("moving", moving), ("recon", recon)]:
        nib.save(
            nib.Nifti1Image(vol.squeeze()[0].cpu().numpy(), affine),
            (out_dir / f"{name}_ct.nii.gz").as_posix(),
        )


if __name__ == "__main__":
    main()
