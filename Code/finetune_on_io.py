"""Self-distillation fine-tuning: regress the lvl3 network output onto the
IO-refined fields produced by io_on_train.py.

The refined fields live in the RESIDUAL frame (see io_on_train.py), so the
supervision is exactly the synthetic-branch DVF loss of level3:
    loss = ((F_X_Y - gt_unit) ** 2).mean()
and no other term. The IO field already encodes every objective term the IO
descended (dice, NCC, tumour volume, rigidity, fold barrier), so regressing
onto it distills all of them at once; re-adding the feed-forward losses would
double-count them at their old (pre-IO) weights.

Both this script and io_on_train.py run the dataset with augment=False: a
saved field is only valid for the deterministic, un-augmented pair.

Usage:
    python finetune_on_io.py \
        --model-path /path/to/..._stagelvl3_best_combined.pth \
        --dvf-dir /path/to/io_train_dvfs
"""

import argparse
from pathlib import Path

import io_on_train
import my_data
import numpy as np
import torch
import tqdm
import utils
from config import DATA_PATH, TrainingConfig
from Functions import generate_grid, generate_grid_unit
from inference import create_model
from level3 import evaluate_lvl3
from miccai2020_model_stage import (
    SpatialTransform_unit,
    SpatialTransformNearest_unit,
    build_similarity_loss,
    smoothloss,
)
from torch.utils import data as torch_data


class IODistillDataset(torch_data.Dataset):
    """Real train pairs (augment=False) + the IO-refined residual field as
    `gt_unit`. Pairs without a saved field are dropped with a warning, so a
    partially finished io_on_train run can already be trained on."""

    def __init__(self, base: my_data.PSMARegDataset, dvf_dir: Path) -> None:
        self.base = base
        self.dvf_dir = dvf_dir
        self.indices = []
        missing = 0
        for i, (case_id, tp_x, tp_y) in enumerate(base.pairs):
            if (dvf_dir / io_on_train.dvf_filename(case_id, tp_x, tp_y)).exists():
                self.indices.append(i)
            else:
                missing += 1
        if missing:
            print(
                f"[finetune_on_io] {missing}/{len(base.pairs)} pairs have no "
                f"refined field in {dvf_dir} and are skipped"
            )
        if not self.indices:
            raise FileNotFoundError(f"no refined fields found in {dvf_dir}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict:
        data = self.base[self.indices[index]]
        payload = torch.load(
            self.dvf_dir
            / io_on_train.dvf_filename(data["case_id"], data["tp_x"], data["tp_y"]),
            map_location="cpu",
        )
        data["gt_unit"] = payload["disp_unit"].float()
        return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        default="/home/iml/fryderyk.koegl/data/PSMAReg/models/PSMAReg_LapIRN_auspicious-sloth-39469081_stagelvl3_best_combined.pth",
    )
    parser.add_argument(
        "--dvf-dir", type=Path, default=DATA_PATH / "PSMAReg/io_train_dvfs"
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="lower than lr_lvl3 on purpose: this is a fine-tune of a "
        "converged checkpoint, not a fresh level",
    )
    parser.add_argument(
        "--val-interval", type=int, default=2, help="run evaluate_lvl3 every N epochs"
    )
    args = parser.parse_args()

    cfg = TrainingConfig()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_ids, val_ids = my_data.get_train_val_split(
        data_dir=cfg.data_dir,
        split_path=cfg.split_path,
        val_fraction=cfg.val_fraction,
        tubingen=False,
        nlst=False,
        abdomen=False,
    )

    # augment=False: the refined fields were computed on the un-augmented pairs
    train_base = my_data.PSMARegDataset(
        case_ids=train_ids,
        cfg=cfg,
        augment=False,
        use_cache=cfg.use_cache_train_real,
        include_intermediate_pairs=True,
        num_workers=cfg.num_workers,
    )
    train_dataset = IODistillDataset(train_base, args.dvf_dir)
    train_loader = torch_data.DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True
    )

    val_dataset = my_data.PSMARegDataset(
        case_ids=val_ids,
        cfg=cfg,
        augment=False,
        use_cache=cfg.use_cache_valid,
        include_intermediate_pairs=False,
        num_workers=cfg.num_workers,
    )
    val_loader = torch_data.DataLoader(val_dataset, batch_size=cfg.batch_size)

    # the whole model (lvl2 submodule included) is trainable: the checkpoint
    # comes from after the lvl3 unfreeze, and the distillation target is a
    # refinement of the full pyramid's output
    model = create_model(device, cfg, args.model_path)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    transform = SpatialTransform_unit().to(device)
    transform_nearest = SpatialTransformNearest_unit().to(device)
    for p in list(transform.parameters()) + list(transform_nearest.parameters()):
        p.requires_grad = False

    grid = generate_grid(cfg.img_shape)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).to(device).float()
    grid_full = generate_grid_unit(cfg.img_shape)
    grid_full = (
        torch.from_numpy(np.reshape(grid_full, (1,) + grid_full.shape))
        .to(device)
        .float()
    )

    # only needed by evaluate_lvl3, not by the distillation loss itself
    loss_similarity_ct = build_similarity_loss(cfg, level=3)
    loss_similarity_pet = build_similarity_loss(cfg, level=3)

    cfg.model_save_dir.mkdir(parents=True, exist_ok=True)
    experiment_name = "IO_finetune"
    best_combined = float("-inf")

    with utils.start_logging_run(cfg):
        utils.overwrite_run_name(experiment_name)
        run_name = utils.get_run_name()
        last_path = (
            cfg.model_save_dir / f"{experiment_name}_{run_name}_iodistill_last.pth"
        )
        best_path = (
            cfg.model_save_dir
            / f"{experiment_name}_{run_name}_iodistill_best_combined.pth"
        )

        utils.log_config(
            {
                **cfg.to_mlflow_params(),
                "finetune_model_path": str(args.model_path),
                "finetune_dvf_dir": str(args.dvf_dir),
                "finetune_lr": args.lr,
                "finetune_epochs": args.epochs,
                "finetune_n_pairs": len(train_dataset),
            }
        )

        global_step = 0
        pbar_outer = tqdm.tqdm(range(args.epochs), desc="iodistill epoch")
        for epoch in pbar_outer:
            model.train()
            epoch_loss = 0.0
            optimizer.zero_grad()
            pbar = tqdm.tqdm(train_loader, desc=f"iodistill epoch {epoch}")
            for step, batch in enumerate(pbar):
                X_prereg, _, _, Y, _, _ = io_on_train.prereg_batch(
                    batch, cfg, device, transform, transform_nearest, grid_full
                )
                gt_unit = batch["gt_unit"].to(device).float()

                F_X_Y, _, _, _, _, _, _ = model(X_prereg, Y)

                loss_dvf = ((F_X_Y - gt_unit) ** 2).mean()
                (loss_dvf / cfg.accumulation_steps).backward()

                if (step + 1) % cfg.accumulation_steps == 0 or (step + 1) == len(
                    train_loader
                ):
                    optimizer.step()
                    optimizer.zero_grad()

                epoch_loss += loss_dvf.item()
                pbar.set_postfix(dvf=f"{loss_dvf.item():.8f}")
                utils.log_metrics({"finetune/dvf": loss_dvf.item()}, step=global_step)
                global_step += 1

            utils.log_metrics(
                {"finetune/dvf_epoch": epoch_loss / len(train_loader)}, step=global_step
            )

            is_last = epoch == args.epochs - 1
            if epoch % args.val_interval == 0 or is_last:
                val_metrics = evaluate_lvl3(
                    model,
                    val_loader,
                    cfg,
                    device,
                    loss_similarity_ct,
                    loss_similarity_pet,
                    smoothloss,
                    transform,
                    grid,
                    epoch,
                    args.val_interval,
                    saved_initial=True,
                    is_last=is_last,
                )
                # same selection as level3.train_lvl3: replica of the official
                # ranking (accuracy / tumour-bias / regularity, 0.4/0.4/0.2
                # weighted geometric mean). HIGHER is better.
                selection = utils.challenge_selection_score(
                    cfg,
                    dice_ct_loss=val_metrics["dice_ct"],
                    hd95=val_metrics["hd95"],
                    mtv_bias=val_metrics["mtv_bias"],
                    tlg_bias=val_metrics["tlg_bias"],
                    ndv_percent=val_metrics["ndv"],
                )
                combined_score = selection["final"]
                utils.log_metrics(
                    {
                        **{f"finetune_val/{k}": v for k, v in val_metrics.items()},
                        **{
                            f"finetune_val/sel_{k}": v
                            for k, v in selection.items()
                            if not (isinstance(v, float) and np.isnan(v))
                        },
                    },
                    step=global_step,
                )
                print(f"epoch {epoch} val: {val_metrics}")
                print(f"epoch {epoch} combined score: {combined_score:.4f}")

                torch.save(model.state_dict(), last_path)
                # NaN never satisfies `>`, so a round with a missing component
                # simply does not save (same convention as train_lvl3)
                if combined_score > best_combined:
                    best_combined = combined_score
                    torch.save(model.state_dict(), best_path)
                    print(
                        f"  new best combined {best_combined:.4f} -> {best_path.name}"
                    )

    print(f"Done. last: {last_path}\n      best: {best_path}")


if __name__ == "__main__":
    main()
