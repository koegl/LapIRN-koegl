import os

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import affine_reg
import Functions
import miccai2020_model_stage
import my_data
import numpy as np
import torch
from config import TrainingConfig


def debug_asym_crop() -> None:
    config = TrainingConfig()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_ids, _ = my_data.get_train_val_split(
        data_dir=config.data_dir,
        split_path=config.split_path,
        val_fraction=config.val_fraction,
    )

    # unaugmented pair -> no flip, no symmetric crop (clean baseline)
    ds = my_data.PSMARegDataset(case_ids=train_ids, cfg=config, augment=False)
    batch = ds[0]

    # force a deterministic moving-only crop (distinct head/feet so they're tellable)
    crop_head_moving = 00
    crop_feet_moving = 100

    moving = {"x": batch["x"].clone()}
    moving = my_data.apply_z_crop(moving, ["x"], crop_head_moving, crop_feet_moving)

    X = moving["x"].unsqueeze(0).to(device).float()
    Y = batch["y"].unsqueeze(0).to(device).float()

    grid_full = Functions.generate_grid_unit(config.img_shape)
    grid_full = (
        torch.from_numpy(np.reshape(grid_full, (1,) + grid_full.shape))
        .to(device)
        .float()
    )

    transform = miccai2020_model_stage.SpatialTransform_unit().to(device)

    # affine flow for the *uncropped* pair: aug params neutral here because the
    # asymmetric moving crop must NOT be reflected in the DVF
    flow_affine = affine_reg.create_affine_flow(
        config=config,
        device=device,
        case_id=batch["case_id"],
        tp_x=batch["tp_x"],
        tp_y=batch["tp_y"],
        aug_flipped=False,
        aug_crop_head=0,
        aug_crop_feet=0,
    )

    X_affine = transform(X, flow_affine, grid_full)

    out_dir = config.save_dir / "debug_asym"
    my_data.save_volume(
        volume=X[:, 0:1], out_dir=out_dir, epoch=0, name="x_moving_cropped_ct"
    )
    my_data.save_volume(
        volume=X_affine[:, 0:1], out_dir=out_dir, epoch=0, name="x_affine_ct"
    )
    my_data.save_volume(volume=Y[:, 0:1], out_dir=out_dir, epoch=0, name="y_ct")

    print(f"crop_head_moving={crop_head_moving}, crop_feet_moving={crop_feet_moving}")
    print(f"X shape: {tuple(X.shape)}")
    print(f"saved to: {out_dir}")


def main() -> None:

    # from Functions import generate_grid, generate_grid_unit

    # g1 = generate_grid((4, 4, 4))
    # g2 = generate_grid_unit((4, 4, 4))
    # print("generate_grid range:", g1.min(), g1.max())
    # print("generate_grid_unit range:", g2.min(), g2.max())

    # return

    config = TrainingConfig()

    train_ids, val_ids = my_data.get_train_val_split(
        data_dir=config.data_dir,
        split_path=config.split_path,
        val_fraction=config.val_fraction,
    )

    ds_no_flip = my_data.PSMARegDataset(case_ids=train_ids, cfg=config, augment=False)

    ds_flip = my_data.PSMARegDataset(case_ids=train_ids, cfg=config, augment=True)

    batch_no_flip = ds_no_flip[0]
    batch_flip = ds_flip[0]

    x_no_flip = batch_no_flip["x"]
    x_flip = batch_flip["x"]

    print("max diff:", (x_no_flip - x_flip).abs().max().item())
    print("are identical:", torch.allclose(x_no_flip, x_flip))

    # Check if flipping axis 0 of x_no_flip matches x_flip
    for axis in [0, 1, 2, 3]:
        flipped = torch.flip(x_no_flip, dims=[axis])
        print(
            f"matches flip on dim {axis}:", torch.allclose(flipped, x_flip, atol=1e-5)
        )


if __name__ == "__main__":
    # main()
    debug_asym_crop()
