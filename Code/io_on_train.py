"""Run instance optimization on every training pair and save the refined
residual fields as self-distillation targets for finetune_on_io.py.

The IO runs in the RESIDUAL frame: base = F_X_Y (the network output on the
affine-preregistered pair), moving = X_prereg, labels affine-pre-warped. This
is the same frame the training-time unrolled IO uses (level3.io_inner_loss_fn),
NOT the deploy-time total-field frame of inference.process_subject — so the
refined field is directly comparable to F_X_Y and can supervise it with the
existing loss_dvf = ((F_X_Y - gt_unit)**2).mean() without any affine
decomposition. The price is that the volume terms do not see det(A); the
unrolled-IO path accepts the same trade.

Augmentation is OFF: the saved field for (case_id, tp_x, tp_y) is only valid
for the un-augmented pair, so finetune_on_io.py must also run augment=False.

Usage:
    python io_on_train.py --model-path /path/to/..._stagelvl3_best_combined.pth
"""

import argparse
import csv
from pathlib import Path
from typing import Tuple

import affine_reg
import instance_opt
import my_data
import numpy as np
import torch
import tqdm
from config import DATA_PATH, TrainingConfig
from Functions import generate_grid_unit
from inference import create_model
from miccai2020_model_stage import (
    SpatialTransform_unit,
    SpatialTransformNearest_unit,
)
from torch.utils import data as torch_data


def dvf_filename(case_id: str, tp_x: str, tp_y: str) -> str:
    return f"dvf_{case_id}_{tp_x}_{tp_y}.pt"


def prereg_batch(
    batch: dict,
    cfg: TrainingConfig,
    device: torch.device,
    transform: torch.nn.Module,
    transform_nearest: torch.nn.Module,
    grid_full: torch.Tensor,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Affine-preregister the moving image and labels of a real (un-augmented)
    train batch, mirroring the real branch of level3.train_lvl3.

    Returns (X_prereg, X_lbl_ct_prereg, X_lbl_pet_prereg, Y, Y_lbl_ct,
    Y_lbl_pet), all on device. Labels are nearest-warped into the affine frame,
    which is what both run_io (residual-frame call) and the unrolled IO
    consume."""
    if cfg.use_poly_affine:
        raise NotImplementedError(
            "io_on_train / finetune_on_io only support the plain affine prereg"
        )

    X = batch["x"].to(device).float()
    Y = batch["y"].to(device).float()
    X_lbl_ct = batch["x_label_ct"].to(device)
    X_lbl_pet = batch["x_label_pet"].to(device)
    Y_lbl_ct = batch["y_label_ct"].to(device)
    Y_lbl_pet = batch["y_label_pet"].to(device)

    if "_" in batch["case_id"][0]:
        case_id_x, case_id_y = batch["case_id"][0].split("_")
    else:
        case_id_x = batch["case_id"][0]
        case_id_y = batch["case_id"][0]

    flow_prereg = affine_reg.create_affine_flow(
        config=cfg,
        device=device,
        case_id_x=case_id_x,
        case_id_y=case_id_y,
        tp_x=batch["tp_x"][0],
        tp_y=batch["tp_y"][0],
        aug_flipped=batch["aug_flipped"],
        aug_crop_head=batch["aug_crop_head"],
        aug_crop_feet=batch["aug_crop_feet"],
        aug_crop_head_fixed=batch["aug_crop_head_fixed"],
        aug_crop_feet_fixed=batch["aug_crop_feet_fixed"],
    )

    X_prereg = transform(X, flow_prereg, grid_full)
    X_lbl_ct_prereg = transform_nearest(X_lbl_ct.float(), flow_prereg, grid_full)
    X_lbl_pet_prereg = transform_nearest(X_lbl_pet.float(), flow_prereg, grid_full)

    return X_prereg, X_lbl_ct_prereg, X_lbl_pet_prereg, Y, Y_lbl_ct, Y_lbl_pet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        default="/home/iml/fryderyk.koegl/data/PSMAReg/models/PSMAReg_LapIRN_auspicious-sloth-39469081_stagelvl3_best_combined.pth",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DATA_PATH / "PSMAReg/io_train_dvfs",
        help="where the refined fields (one .pt per pair) are written",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="recompute pairs whose output file already exists",
    )
    parser.add_argument(
        "--include-pet",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--include-rigidity",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    cfg = TrainingConfig()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_ids, _ = my_data.get_train_val_split(
        data_dir=cfg.data_dir,
        split_path=cfg.split_path,
        val_fraction=cfg.val_fraction,
        tubingen=False,
        nlst=False,
        abdomen=False,
    )

    # augment=False: the saved field must correspond to the deterministic pair
    dataset = my_data.PSMARegDataset(
        case_ids=train_ids,
        cfg=cfg,
        augment=False,
        use_cache=False,
        include_intermediate_pairs=True,
        num_workers=cfg.num_workers,
    )
    loader = torch_data.DataLoader(dataset, batch_size=1, shuffle=False)

    model = create_model(device, cfg, args.model_path)

    transform = SpatialTransform_unit().to(device)
    transform_nearest = SpatialTransformNearest_unit().to(device)

    grid_full = generate_grid_unit(cfg.img_shape)
    grid_full = (
        torch.from_numpy(np.reshape(grid_full, (1,) + grid_full.shape))
        .to(device)
        .float()
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "io_summary.csv"
    write_header = not summary_path.exists()
    summary_file = open(summary_path, "a", newline="")
    summary = csv.writer(summary_file)
    if write_header:
        summary.writerow(
            ["case_id", "tp_x", "tp_y", "dice_base", "dice_refined", "file"]
        )

    for batch in tqdm.tqdm(loader, desc="IO on train pairs", ncols=120):
        case_id = batch["case_id"][0]
        tp_x = batch["tp_x"][0]
        tp_y = batch["tp_y"][0]
        out_path = args.out_dir / dvf_filename(case_id, tp_x, tp_y)
        if out_path.exists() and not args.overwrite:
            tqdm.tqdm.write(f"{out_path.name}: exists, skipping")
            continue

        X_prereg, X_lbl_ct, X_lbl_pet, Y, Y_lbl_ct, _ = prereg_batch(
            batch, cfg, device, transform, transform_nearest, grid_full
        )

        with torch.no_grad():
            F_X_Y, _, _, _, _, _, _ = model(X_prereg, Y)

        # run_io needs autograd on its inner velocity leaf: no no_grad here
        refined = instance_opt.run_io(
            Y,
            F_X_Y,
            X_prereg,
            X_lbl_ct,
            X_lbl_pet,
            Y_lbl_ct,
            transform,
            transform_nearest,
            grid_full,
            cfg,
            device,
            include_pet=args.include_pet,
            include_rigidity=args.include_rigidity,
        )
        refined = refined.detach()

        # diagnostic: hard mean dice before/after, both on the prereg labels so
        # the two numbers are comparable (double interpolation on both)
        with torch.no_grad():
            warped_base = instance_opt.warp_label(
                X_lbl_ct, F_X_Y, grid_full, transform_nearest
            )
            warped_ref = instance_opt.warp_label(
                X_lbl_ct, refined, grid_full, transform_nearest
            )
            target = Y_lbl_ct[0, 0].round().long()
            dice_base = instance_opt.multilabel_dice(
                warped_base[0, 0].round().long(), target
            )
            dice_ref = instance_opt.multilabel_dice(
                warped_ref[0, 0].round().long(), target
            )

        # fp16 halves the disk cost; unit-flow values are O(1e-2), where fp16
        # keeps ~3 significant digits — far below the IO refinement itself
        torch.save(
            {
                "disp_unit": refined.squeeze(0).to(torch.float16).cpu(),
                "case_id": case_id,
                "tp_x": tp_x,
                "tp_y": tp_y,
                "model_path": str(args.model_path),
            },
            out_path,
        )
        summary.writerow(
            [case_id, tp_x, tp_y, f"{dice_base:.4f}", f"{dice_ref:.4f}", out_path.name]
        )
        summary_file.flush()
        tqdm.tqdm.write(
            f"{case_id} {tp_x}->{tp_y}: dice {dice_base:.4f} -> {dice_ref:.4f}"
        )

    summary_file.close()
    print(f"Done. Fields in {args.out_dir}, summary in {summary_path}")


if __name__ == "__main__":
    main()
