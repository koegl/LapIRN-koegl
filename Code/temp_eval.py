from pathlib import Path

import config
import level3
import my_data
import torch
from miccai2020_model_stage import (
    Miccai2020_LDR_laplacian_unit_add_lvl1,
    Miccai2020_LDR_laplacian_unit_add_lvl2,
    Miccai2020_LDR_laplacian_unit_add_lvl3,
)
from torch.utils import data as torch_data


def main() -> None:
    cfg = config.TrainingConfig()
    device = torch.device("cuda")

    ckpt_lvl2 = Path(
        "/lustre/groups/iml/data/PSMAReg/models/PSMAReg_LapIRN_unleashed-sloth-539_stagelvl2_best.pth"
    )
    ckpt_lvl3 = Path(
        "/lustre/groups/iml/data/PSMAReg/models/PSMAReg_LapIRN_redolent-loon-722_stagelvl3_best.pth"
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if cfg.use_lvl1:
        model_lvl1 = Miccai2020_LDR_laplacian_unit_add_lvl1(
            in_channel=cfg.in_channel,
            n_classes=cfg.n_classes,
            start_channel=cfg.start_channel,
            is_train=True,
            imgshape=cfg.img_shape_4,
            range_flow=cfg.range_flow,
        ).to(device)
    else:
        model_lvl1 = None

    model_lvl2 = Miccai2020_LDR_laplacian_unit_add_lvl2(
        in_channel=cfg.in_channel,
        n_classes=cfg.n_classes,
        start_channel=cfg.start_channel,
        is_train=True,
        imgshape=cfg.img_shape_2,
        range_flow=cfg.range_flow,
        model_lvl1=model_lvl1,
    ).to(device)

    model_lvl2.load_state_dict(torch.load(ckpt_lvl2))

    for param in model_lvl2.parameters():
        param.requires_grad = False

    model = Miccai2020_LDR_laplacian_unit_add_lvl3(
        in_channel=cfg.in_channel,
        n_classes=cfg.n_classes,
        start_channel=cfg.start_channel,
        is_train=True,
        imgshape=cfg.img_shape,
        range_flow=cfg.range_flow,
        model_lvl2=model_lvl2,
    ).to(device)

    state = torch.load(ckpt_lvl3, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

    _, val_ids = my_data.get_train_val_split(
        data_dir=cfg.data_dir, split_path=cfg.split_path, val_fraction=cfg.val_fraction
    )
    val_dataset = my_data.PSMARegDataset(
        case_ids=val_ids,
        cfg=cfg,
        augment=False,
        use_cache=cfg.use_cache_valid,
        include_intermediate_pairs=False,
        num_workers=cfg.num_workers,
    )
    valid_generator = torch_data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    for zero in (False, True):
        model.debug_zero_residual = zero
        dice = level3.evaluate_lvl3(model, valid_generator, cfg, device)  # see step 4
        tag = "zero-residual" if zero else "trained"
        print(f"{tag:>14} val dice_ct: {dice:.4f}")


if __name__ == "__main__":
    main()
