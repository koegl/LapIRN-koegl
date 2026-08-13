import json
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import affine_reg
import Functions
import instance_opt
import miccai2020_model_stage
import my_data
import ndv_official
import nibabel as nib
import numpy as np
import pandas as pd
import poly_affine_reg
import torch
import torch.nn.functional as F
import tqdm
import utils
from config import TrainingConfig

os.environ["TOTALSEG_WEIGHTS_PATH"] = (
    "/home/iml/fryderyk.koegl/code/LapIRN-koegl/totalsegmentator_weights/nnunet/results"
)
sys.path.insert(0, "/home/iml/fryderyk.koegl/code/autopet-3-submission")
import datetime

import hd95_official
import totalsegmentator.python_api as totalseg

import main as autopet


def load_split(split_path: Path) -> Tuple[List[str], List[str]]:
    with open(split_path, "r") as f:
        split = json.load(f)
    train = split["train"]
    val = split["val"]
    return train, val


def load_io_labels(
    label_dir: Path,
    case_id: str,
    device: torch.device,
    ct_template: str = "{case_id}_0000_{tp}",
    pet_template: str = "{case_id}_0001_{tp}",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    def load(path: Path) -> torch.Tensor:
        arr = my_data.nib.load(str(path)).get_fdata().astype(np.int16)
        label = torch.from_numpy(arr)[None, None].to(device).float()
        return label

    def p(label_dir: Path, template: str, tp: str) -> Path:
        stem = template.format(case_id=case_id, tp=tp)
        path = label_dir / f"{stem}.nii.gz"
        if not path.exists():
            path = label_dir / f"{stem}.nii"
        return path

    x_lbl_ct = load(p(label_dir, ct_template, "01"))
    y_lbl_ct = load(p(label_dir, ct_template, "00"))
    x_lbl_pet = load(p(label_dir, pet_template, "01"))
    y_lbl_pet = load(p(label_dir, pet_template, "00"))

    labels = (x_lbl_ct, x_lbl_pet, y_lbl_ct, y_lbl_pet)
    return labels


def build_pet_predictor(device: torch.device):
    model_folder = (
        "/home/iml/fryderyk.koegl/code/autopet-3-submission/"
        "autoPET-3-LesionTracer/Dataset222_AutoPETIII_2024/"
        "autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres_bs3"
    )
    folds = (0, 1, 2, 3, 4)
    predictor = autopet.build_predictor(model_folder, folds, device)
    return predictor


def predict_pet_labels(
    predictor,
    ct_path: Path,
    pet_path: Path,
    device: torch.device,
) -> torch.Tensor:
    seg = autopet.run_inference_in_memory(predictor, str(ct_path), str(pet_path))
    label = torch.from_numpy(seg.astype("int16"))[None, None].to(device).float()
    return label


def compute_ndv_official(
    disp_voxel: np.ndarray,
    mask: np.ndarray,
) -> float:
    """percent_ndv matching the challenge scorer.
    disp_voxel: (3, H, W, D) displacement, voxel units, official component order.
    mask: (H, W, D) body mask.
    """
    identity = ndv_official.get_identity_grid(disp_voxel)
    trans = disp_voxel + identity
    jac_dets = ndv_official.calc_jac_dets(trans)
    mask_inner = mask[1:-1, 1:-1, 1:-1] > 0
    _, _, non_diff_volume, _ = ndv_official.calc_measurements(jac_dets, mask_inner)
    total_voxels = float(mask_inner.sum())
    percent_ndv = non_diff_volume / total_voxels * 100.0
    return float(percent_ndv)


def compute_hd95_official(
    fixed_labels: np.ndarray,
    moving_labels: np.ndarray,
    warped_labels: np.ndarray,
    spacing_mm: Tuple[float, float, float],
) -> float:
    hd95 = hd95_official.compute_average_ct_label_hd95(
        fixed_labels,
        moving_labels,
        warped_labels,
        spacing_mm,
    )
    return float(hd95)


# --- variables (define here, no argparse) ---
val_image_dir = Path("/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/imagesTs")
val_cache_dir = Path("/home/iml/fryderyk.koegl/data/PSMAReg/affine_cache_val")
val_subjects = [
    "0001",
    "0003",
    "0005",
    "0007",
    "0008",
    "0009",
    "0013",
    "0021",
    "0024",
    "0029",
    "0031",
    "0033",
    "0034",
    "0035",
    "0036",
    "0038",
    "0039",
    "0042",
    "0047",
    "0048",
    "4000",
    "4003",
    "4011",
    "4013",
    "4022",
    "4032",
    "4034",
    "4036",
    "4040",
    "4041",
    "4047",
    "4048",
    "4053",
    "4054",
    "4055",
    "4063",
    "4064",
    "4065",
    "4067",
    "4068",
]
split_path = Path("/home/iml/fryderyk.koegl/code/LapIRN-koegl/split.json")
my_val_image_dir = Path(
    "/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/imagesTr"
)
my_val_seg_dir = Path("/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/labelsTr")

results_csv_my_val_dice = Path(
    "/home/iml/fryderyk.koegl/code/LapIRN-koegl/submission_results/csvs/results_my_val_dice.csv"
)
results_csv_official_val_dice = Path(
    "/home/iml/fryderyk.koegl/code/LapIRN-koegl/submission_results/csvs/results_official_val_dice.csv"
)
results_csv_my_val_dice_per_label = Path(
    "/home/iml/fryderyk.koegl/code/LapIRN-koegl/submission_results/csvs/results_my_val_dice_per_label.csv"
)
results_csv_official_val_dice_per_label = Path(
    "/home/iml/fryderyk.koegl/code/LapIRN-koegl/submission_results/csvs/results_official_val_dice_per_label.csv"
)
results_csv_official_mtv = Path(
    "/home/iml/fryderyk.koegl/code/LapIRN-koegl/submission_results/csvs/results_official_val_mtv.csv"
)
results_csv_official_tlg = Path(
    "/home/iml/fryderyk.koegl/code/LapIRN-koegl/submission_results/csvs/results_official_val_tlg.csv"
)
results_csv_my_val_mtv = Path(
    "/home/iml/fryderyk.koegl/code/LapIRN-koegl/submission_results/csvs/results_my_val_mtv.csv"
)
results_csv_my_val_tlg = Path(
    "/home/iml/fryderyk.koegl/code/LapIRN-koegl/submission_results/csvs/results_my_val_tlg.csv"
)
results_csv_official_ndv = Path(
    "/home/iml/fryderyk.koegl/code/LapIRN-koegl/submission_results/csvs/results_official_val_ndv.csv"
)
results_csv_my_val_ndv = Path(
    "/home/iml/fryderyk.koegl/code/LapIRN-koegl/submission_results/csvs/results_my_val_ndv.csv"
)
results_csv_official_hd95 = Path(
    "/home/iml/fryderyk.koegl/code/LapIRN-koegl/submission_results/csvs/results_official_val_hd95.csv"
)
results_csv_my_val_hd95 = Path(
    "/home/iml/fryderyk.koegl/code/LapIRN-koegl/submission_results/csvs/results_my_val_hd95.csv"
)

# the before-registration row is only written when a csv is created from
# scratch. the hd95 csvs already exist, so set this to print the
# before-registration hd95 values as comma-separated values in the terminal
# instead (paste them into the existing csv by hand).
print_hd95_before_to_terminal: bool = False


def predict_ct_labels(
    ct_path: Path,
    device: torch.device,
) -> torch.Tensor:
    ct_img = nib.load(str(ct_path))
    seg_img = totalseg.totalsegmentator(
        ct_img,
        task="total",
        fast=False,
        device="gpu" if device.type == "cuda" else "cpu",
        quiet=True,
    )
    arr = seg_img.get_fdata().astype("int16")
    label = torch.from_numpy(arr)[None, None].to(device).float()
    return label


def compress_to_zip(source_dir: Path, output_zip: Path) -> None:
    files = list(source_dir.rglob("*"))
    files = [f for f in files if f.is_file()]
    files = [f for f in files if f.name.startswith("disp_00")]

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in files:
            arcname = file.relative_to(source_dir)
            zf.write(file, arcname)
            print(f"Added: {arcname}")

    shutil.rmtree(source_dir)

    print(f"\nDone! Zip saved to: {output_zip}")


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


def load_val_pair_OLD(val_image_dir: Path, case_id: str) -> Dict[str, torch.Tensor]:
    """Load fixed (00) + moving (01) val pair."""

    def load_ct(tp: str) -> np.ndarray:
        path = val_image_dir / f"PSMARegPSMA_{case_id}_0000_{tp}.nii.gz"
        return my_data.nib.load(str(path)).get_fdata().astype(np.float32)

    def load_pet(tp: str) -> np.ndarray:
        path = val_image_dir / f"PSMARegPSMA_{case_id}_0001_{tp}.nii.gz"
        return my_data.nib.load(str(path)).get_fdata().astype(np.float32)

    x_ct_raw = load_ct("01")
    y_ct_raw = load_ct("00")

    x_mask = my_data.get_body_mask(x_ct_raw)
    y_mask = my_data.get_body_mask(y_ct_raw)

    x_ct_raw = my_data.apply_body_mask(
        x_ct_raw, x_mask, fill_value=float(np.percentile(x_ct_raw, 0.5))
    )
    y_ct_raw = my_data.apply_body_mask(
        y_ct_raw, y_mask, fill_value=float(np.percentile(y_ct_raw, 0.5))
    )

    x_pet_raw = my_data.apply_body_mask(load_pet("01"), x_mask, fill_value=0.0)
    y_pet_raw = my_data.apply_body_mask(load_pet("00"), y_mask, fill_value=0.0)

    def t(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(arr).unsqueeze(0)

    x = torch.cat(
        [t(my_data.norm_ct_OLD(x_ct_raw)), t(my_data.norm_pet(x_pet_raw))], dim=0
    )
    y = torch.cat(
        [t(my_data.norm_ct_OLD(y_ct_raw)), t(my_data.norm_pet(y_pet_raw))], dim=0
    )
    pair = {"x": x.float(), "y": y.float()}
    return pair


def load_val_pair(val_image_dir: Path, case_id: str) -> Dict[str, torch.Tensor]:
    """Load fixed (00) + moving (01) val pair."""

    def load_ct(tp: str) -> np.ndarray:
        path = val_image_dir / f"PSMARegPSMA_{case_id}_0000_{tp}.nii.gz"
        return my_data.nib.load(str(path)).get_fdata().astype(np.float32)

    def load_pet(tp: str) -> np.ndarray:
        path = val_image_dir / f"PSMARegPSMA_{case_id}_0001_{tp}.nii.gz"
        return my_data.nib.load(str(path)).get_fdata().astype(np.float32)

    x_ct_raw = load_ct("01")
    y_ct_raw = load_ct("00")

    x_mask = my_data.get_body_mask(x_ct_raw)
    y_mask = my_data.get_body_mask(y_ct_raw)

    x_ct_raw = my_data.apply_body_mask(x_ct_raw, x_mask, fill_value=my_data.CT_AIR_HU)
    y_ct_raw = my_data.apply_body_mask(y_ct_raw, y_mask, fill_value=my_data.CT_AIR_HU)

    x_pet_raw = my_data.apply_body_mask(load_pet("01"), x_mask, fill_value=0.0)
    y_pet_raw = my_data.apply_body_mask(load_pet("00"), y_mask, fill_value=0.0)

    def t(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(arr).unsqueeze(0)

    x = torch.cat([t(my_data.norm_ct(x_ct_raw)), t(my_data.norm_pet(x_pet_raw))], dim=0)
    y = torch.cat([t(my_data.norm_ct(y_ct_raw)), t(my_data.norm_pet(y_pet_raw))], dim=0)
    pair = {"x": x.float(), "y": y.float()}
    return pair


def create_model(
    device: torch.device, cfg: TrainingConfig, model_path: Path
) -> torch.nn.Module:
    if cfg.use_lvl1:
        model_lvl1 = miccai2020_model_stage.Miccai2020_LDR_laplacian_unit_add_lvl1(
            in_channel=cfg.in_channel,
            n_classes=cfg.n_classes,
            start_channel=cfg.start_channel,
            is_train=True,
            imgshape=cfg.img_shape_4,
            range_flow=cfg.range_flow,
            cost_volume_mode=cfg.cost_volume_mode,
            cost_volume_radius=cfg.cost_volume_radius,
            cost_volume_dilation=cfg.cost_volume_dilation,
            cost_volume_feat_channels=cfg.cost_volume_feat_channels,
            cost_volume_out_channels=cfg.cost_volume_out_channels,
            n_resblocks=cfg.n_resblocks,
            resblock_expansion=cfg.resblock_expansion,
        ).to(device)
    else:
        model_lvl1 = None

    model_lvl2 = miccai2020_model_stage.Miccai2020_LDR_laplacian_unit_add_lvl2(
        in_channel=cfg.in_channel,
        n_classes=cfg.n_classes,
        start_channel=cfg.start_channel,
        is_train=True,
        imgshape=cfg.img_shape_2,
        range_flow=cfg.range_flow,
        model_lvl1=model_lvl1,
        n_resblocks=cfg.n_resblocks,
        resblock_expansion=cfg.resblock_expansion,
    ).to(device)
    model = miccai2020_model_stage.Miccai2020_LDR_laplacian_unit_add_lvl3(
        in_channel=cfg.in_channel,
        n_classes=cfg.n_classes,
        start_channel=cfg.start_channel,
        is_train=True,
        imgshape=cfg.img_shape,
        range_flow=cfg.range_flow,
        model_lvl2=model_lvl2,
        n_resblocks=cfg.n_resblocks,
        resblock_expansion=cfg.resblock_expansion,
        # must match the checkpoint: a seg-head checkpoint carries seg_head.*
        # weights that load_state_dict would otherwise reject
        use_seg_pet_head=cfg.use_seg_pet_head,
        seg_pet_head_channels=cfg.seg_pet_head_channels,
        use_bone_head=cfg.use_seg_bone_head,
        bone_head_channels=cfg.seg_bone_head_channels,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model


def save_disp(disp_half: torch.Tensor, out_dir: Path, case_id: str) -> None:
    disp_np = disp_half.squeeze(0).cpu().numpy().astype(np.float32)
    disp_np = disp_np[::-1].copy()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"disp_{case_id}_00_{case_id}_01.nii.gz"
    my_data.nib.save(my_data.nib.Nifti1Image(disp_np, np.eye(4)), str(out_path))


def dice_per_label(
    pred: torch.Tensor,
    target: torch.Tensor,
    label_ids: range = range(1, 118),
    empty_value: float = float("nan"),
) -> Dict[int, float]:
    """Per-label hard dice. `empty_value` is returned for labels absent from
    both pred and target: float('nan') = official convention (dropped from the
    mean), 0.0 = 'my' convention (counted)."""
    scores: Dict[int, float] = {}
    for lbl in label_ids:
        p = pred == lbl
        t = target == lbl
        volume_sum = p.sum() + t.sum()
        if volume_sum == 0:
            scores[lbl] = empty_value
        else:
            scores[lbl] = (2.0 * (p & t).sum() / volume_sum).item()
    return scores


def build_per_label_df(
    per_case: Dict[str, Dict[int, float]],
    label_ids: range = range(1, 118),
) -> pd.DataFrame:
    data: Dict[str, Dict] = {}
    for case_id, scores in per_case.items():
        col: Dict = {lbl: scores.get(lbl, float("nan")) for lbl in label_ids}
        present = [v for v in col.values() if not np.isnan(v)]
        col["mean_dice"] = float(np.mean(present)) if present else float("nan")
        data[case_id] = col
    df = pd.DataFrame(data)
    df.index.name = "label"
    return df


def worst_labels(df: pd.DataFrame, n: int = 20) -> pd.Series:
    label_rows = [r for r in df.index if r != "mean_dice"]
    means = df.loc[label_rows].mean(axis=1)
    worst = means.sort_values().head(n)
    return worst


def multilabel_dice(
    pred: torch.Tensor,
    target: torch.Tensor,
    label_ids: range = range(1, 118),
) -> float:
    """Mean Dice over a fixed label set, matching the official scorer.
    pred/target: (H, W, D) int."""
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


def append_results_to_csv(
    csv_path: Path,
    model_name: str,
    dices: Dict[str, float],
    dices_before: Dict[str, float],
) -> None:
    """Append one experiment row, writing a 'before_registration' row first
    if the CSV does not yet exist."""
    rows = []

    if not csv_path.exists():
        avg_before = float(np.nanmean(list(dices_before.values())))
        before_row: Dict[str, object] = {
            "model": "before_registration",
            "avg_dice": avg_before,
        }
        before_row.update({f"dice_{k}": v for k, v in dices_before.items()})
        rows.append(before_row)

    avg_dice = float(np.nanmean(list(dices.values())))
    row: Dict[str, object] = {"model": model_name, "avg_dice": avg_dice}
    row.update({f"dice_{k}": v for k, v in dices.items()})
    rows.append(row)

    new_df = pd.DataFrame(rows)

    if csv_path.exists():
        existing_df = pd.read_csv(csv_path)
        averaged_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        averaged_df = new_df

    averaged_df.to_csv(csv_path, index=False)
    tqdm.tqdm.write(f"results saved → {csv_path}")


def print_metric_row_as_csv(
    model_name: str,
    values: Dict[str, float],
    metric_name: str,
) -> None:
    """Print one metric row as comma-separated values (header + values), so it
    can be pasted into an already-existing csv by hand."""
    avg = float(np.nanmean(list(values.values())))
    header = ["model", f"avg_{metric_name}"] + [
        f"{metric_name}_{k}" for k in values.keys()
    ]
    row = [model_name, f"{avg}"] + [f"{v}" for v in values.values()]
    tqdm.tqdm.write(f"--- {metric_name} row for '{model_name}' ---")
    tqdm.tqdm.write(",".join(header))
    tqdm.tqdm.write(",".join(row))


def append_metric_to_csv(
    csv_path: Path,
    model_name: str,
    values: Dict[str, float],
    metric_name: str,
    values_before: Dict[str, float] | None = None,
) -> None:
    """Append one experiment row for a per-case metric, writing a
    'before_registration' row first if the CSV does not yet exist."""
    rows = []

    if values_before is not None and not csv_path.exists():
        avg_before = float(np.nanmean(list(values_before.values())))
        before_row: Dict[str, object] = {
            "model": "before_registration",
            f"avg_{metric_name}": avg_before,
        }
        before_row.update({f"{metric_name}_{k}": v for k, v in values_before.items()})
        rows.append(before_row)

    avg = float(np.nanmean(list(values.values())))
    row: Dict[str, object] = {"model": model_name, f"avg_{metric_name}": avg}
    row.update({f"{metric_name}_{k}": v for k, v in values.items()})
    rows.append(row)

    new_df = pd.DataFrame(rows)

    if csv_path.exists():
        existing_df = pd.read_csv(csv_path)
        averaged_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        averaged_df = new_df

    averaged_df.to_csv(csv_path, index=False)
    tqdm.tqdm.write(f"results saved → {csv_path}")


def evaluate_split(
    subjects: List[str],
    image_dir: Path,
    seg_dir: Path,
    seg_dir_fast: Path,
    results_csv: Path,
    model_name: str,
    out_dir: Path,
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    model_path: Path,
    transform: torch.nn.Module,
    transform_nearest: torch.nn.Module,
    grid_full: torch.Tensor,
    cfg: TrainingConfig,
    device: torch.device,
    use_io: bool,
    include_pet: bool,
    include_rigidity: bool,
    use_class_weights: bool,
    use_polyaffine: bool,
    io_lr: float,
    io_it: int,
    desc: str,
    mtv_csv: Path,
    tlg_csv: Path,
    ndv_csv: Path,
    hd95_csv: Path,
    per_label_csv: Path,
    ct_label_template: str = "PSMARegPSMA_{case_id}_0000_{tp}",
    pet_label_template: str = "PSMARegPSMA_{case_id}_0001_{tp}",
    skip_model: bool = False,
) -> None:
    dices: Dict[str, float] = {}
    dices_before: Dict[str, float] = {}
    mtvs: Dict[str, float] = {}
    tlgs: Dict[str, float] = {}
    ndvs: Dict[str, float] = {}
    hd95s: Dict[str, float] = {}
    hd95s_before: Dict[str, float] = {}
    per_case: Dict[str, Dict[int, float]] = {}
    for case_id in tqdm.tqdm(subjects, desc=desc, ncols=150):
        (
            dice_after,
            dice_before,
            mtv,
            tlg,
            ndv,
            hd95,
            hd95_before,
            per_label,
        ) = process_subject(
            case_id,
            image_dir,
            out_dir,
            model,
            model_path,
            transform,
            grid_full,
            cfg,
            device,
            transform_nearest,
            seg_dir,
            model_name=model_name,
            use_io=use_io,
            include_pet=include_pet,
            include_rigidity=include_rigidity,
            use_class_weights=use_class_weights,
            use_polyaffine=use_polyaffine,
            skip_model=skip_model,
            seg_dir_fast=seg_dir_fast,
            ct_label_template=ct_label_template,
            pet_label_template=pet_label_template,
            io_lr=io_lr,
            io_it=io_it,
        )
        dices[case_id] = dice_after
        dices_before[case_id] = dice_before
        mtvs[case_id] = mtv
        tlgs[case_id] = tlg
        ndvs[case_id] = ndv
        hd95s[case_id] = hd95
        hd95s_before[case_id] = hd95_before
        per_case[case_id] = per_label

    df = build_per_label_df(per_case)
    df.to_csv(per_label_csv)
    tqdm.tqdm.write(f"worst labels:\n{worst_labels(df)}")

    avg = float(np.nanmean(list(dices.values())))
    tqdm.tqdm.write(f"[{desc}] average dice: {avg:.4f}")
    append_results_to_csv(results_csv, model_name, dices, dices_before=dices_before)
    append_metric_to_csv(mtv_csv, model_name, mtvs, "mtv")
    append_metric_to_csv(tlg_csv, model_name, tlgs, "tlg")
    append_metric_to_csv(ndv_csv, model_name, ndvs, "ndv")
    append_metric_to_csv(
        hd95_csv, model_name, hd95s, "hd95", values_before=hd95s_before
    )
    if print_hd95_before_to_terminal:
        print_metric_row_as_csv("before_registration", hd95s_before, "hd95")


image_pair_memory_cache = {}


def is_old_model(model_path: Path) -> bool:
    ref = datetime.datetime(
        2026, 8, 4, 14, 48, 42, tzinfo=datetime.timezone(datetime.timedelta(hours=2))
    )
    mtime = datetime.datetime.fromtimestamp(
        model_path.stat().st_mtime, tz=datetime.timezone.utc
    )

    return mtime < ref


def process_subject(
    case_id: str,
    val_image_dir: Path,
    out_dir: Path,
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    model_path: Path,
    transform: torch.nn.Module,
    grid_full: torch.Tensor,
    cfg: TrainingConfig,
    device: torch.device,
    transform_nearest: torch.nn.Module,
    seg_dir: Path,
    model_name: str,
    use_io: bool = False,
    include_pet: bool = True,
    include_rigidity: bool = True,
    use_class_weights: bool = False,
    use_polyaffine: bool = False,
    skip_model: bool = False,
    seg_dir_fast: Optional[Path] = None,
    ct_label_template: str = "PSMARegPSMA_{case_id}_0000_{tp}",
    pet_label_template: str = "PSMARegPSMA_{case_id}_0001_{tp}",
    io_lr: float = 1e-1,
    io_it: int = 100,
) -> Tuple[float, float, float, float, float, float, float, Dict[int, float]]:

    if case_id in image_pair_memory_cache:
        pair = image_pair_memory_cache[case_id]
    else:
        if is_old_model(model_path):
            pair = load_val_pair_OLD(val_image_dir, case_id)
        else:
            pair = load_val_pair(val_image_dir, case_id)

        image_pair_memory_cache[case_id] = pair
    X = pair["x"].unsqueeze(0).to(device).float()
    Y = pair["y"].unsqueeze(0).to(device).float()

    fixed_ct_path = val_image_dir / f"PSMARegPSMA_{case_id}_0000_00.nii.gz"
    moving_ct_path = val_image_dir / f"PSMARegPSMA_{case_id}_0000_01.nii.gz"

    def get_affine_dvf_canon() -> np.ndarray:
        return affine_reg.get_affine_dvf(
            case_id_x=case_id,
            case_id_y=case_id,
            tp_x="01",
            tp_y="00",
            fixed_ct_path=fixed_ct_path,
            moving_ct_path=moving_ct_path,
            make_lowres_ants_image_fn=affine_reg.make_lowres_ants_image,
            preprocess_ct_fn=affine_reg.preprocess_ct,
            ants_affine_to_fullres_voxel_disp_fn=affine_reg.ants_affine_to_fullres_voxel_disp,
        )

    dvf = get_affine_dvf_canon()
    dvf = affine_reg.apply_augmentation_to_dvf(
        dvf=dvf, flipped=False, crop_head=0, crop_feet=0
    )
    dvf_tensor = affine_reg.dvf_to_tensor(dvf, device)
    h, w, d = cfg.img_shape
    d_h = dvf_tensor[:, 0] / (h / 2.0)
    d_w = dvf_tensor[:, 1] / (w / 2.0)
    d_d = dvf_tensor[:, 2] / (d / 2.0)
    flow_affine = torch.stack([d_d, d_w, d_h], dim=1).permute(0, 2, 3, 4, 1)

    if use_polyaffine:

        def resolve_seg_path(tp: str) -> Path:
            stem = ct_label_template.format(case_id=case_id, tp=tp)
            path = seg_dir / f"{stem}.nii"
            if not path.exists():
                path = seg_dir / f"{stem}.nii.gz"
            return path

        poly_dvf = poly_affine_reg.get_polyaffine_dvf(
            case_id_x=case_id,
            case_id_y=case_id,
            tp_x="01",
            tp_y="00",
            fixed_seg_path=resolve_seg_path("00"),
            moving_seg_path=resolve_seg_path("01"),
            get_affine_dvf_fn=get_affine_dvf_canon,
            cfg=cfg,
            device=device,
        )
        flow_poly = poly_affine_reg.create_polyaffine_flow(
            poly_dvf=poly_dvf,
            aug_flipped=False,
            aug_crop_head=0,
            aug_crop_feet=0,
            cfg=cfg,
            device=device,
        )
        flow_affine = poly_affine_reg.compose_flows(flow_affine, flow_poly, grid_full)

    X_affine = transform(X, flow_affine, grid_full)

    if skip_model:
        # affine-only / polyaffine-only baseline: identity deformable field
        F_X_Y = torch.zeros_like(flow_affine.permute(0, 4, 1, 2, 3))
        warped = X_affine
    else:
        models = model if isinstance(model, (list, tuple)) else [model]
        with torch.no_grad():
            if len(models) == 1:
                F_X_Y, warped, _, _, _, _, _ = models[0](X_affine, Y)
            else:
                # log-euclidean mean of the ensemble: average the stationary
                # velocity fields, then integrate the mean once. averaging the
                # already-integrated displacements can fold, the velocities cannot.
                velocity = None
                for m in models:
                    _, _, _, v, _, _, _ = m(X_affine, Y)
                    velocity = v if velocity is None else velocity + v
                velocity = velocity / len(models)
                F_X_Y = models[0].diff_transform(velocity, grid_full)
                warped = transform(X_affine, F_X_Y.permute(0, 2, 3, 4, 1), grid_full)

    if use_io and not skip_model:
        x_lbl_ct, x_lbl_pet, y_lbl_ct, y_lbl_pet = load_io_labels(
            seg_dir_fast,
            case_id,
            device,
            ct_template=ct_label_template,
            pet_template=pet_label_template,
        )
        x_lbl_ct = transform_nearest(x_lbl_ct, flow_affine, grid_full)
        x_lbl_pet = transform_nearest(x_lbl_pet, flow_affine, grid_full)

        F_X_Y = instance_opt.run_io(
            Y,
            F_X_Y,
            X_affine,
            x_lbl_ct,
            x_lbl_pet,
            y_lbl_ct,
            transform,
            transform_nearest,
            grid_full,
            cfg,
            device,
            include_pet=include_pet,
            include_rigidity=include_rigidity,
            use_class_weights=use_class_weights,
            lr=io_lr,
            n_steps=io_it,
        )

    if False:
        # --- DIAGNOSTIC: replicate IO hard_dice path (transform_nearest, unit flow) ---
        x_lbl_ct_diag, _, y_lbl_ct_diag, _ = load_io_labels(
            seg_dir_fast,
            pet_label_dir,
            case_id,
            device,
            ct_template=ct_label_template,
            pet_template=pet_label_template,
        )
        x_lbl_ct_diag = transform_nearest(x_lbl_ct_diag, flow_affine, grid_full)
        warped_diag = transform_nearest(
            x_lbl_ct_diag, F_X_Y.permute(0, 2, 3, 4, 1), grid_full
        )
        dice_diag = multilabel_dice(
            warped_diag[0, 0].round().long(), y_lbl_ct_diag[0, 0].round().long()
        )
        tqdm.tqdm.write(f"{case_id}: dice diag (IO path replica)={dice_diag:.4f}")

    deform_grid = grid_full + F_X_Y.permute(0, 2, 3, 4, 1)
    affine_grid = grid_full + flow_affine

    affine_grid_ch = affine_grid.permute(0, 4, 1, 2, 3)
    composed_grid = torch.nn.functional.grid_sample(
        affine_grid_ch,
        deform_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    ).permute(0, 2, 3, 4, 1)

    total_unit_flow = (composed_grid - grid_full).permute(0, 4, 1, 2, 3)

    total_voxel = Functions.transform_unit_flow_to_flow_cuda(
        total_unit_flow.permute(0, 2, 3, 4, 1).clone()
    ).permute(0, 4, 1, 2, 3)

    their_st = SpatialTransformer(size=cfg.img_shape, mode="nearest").to(device)

    def load_seg(tp: str) -> torch.Tensor:
        stem = ct_label_template.format(case_id=case_id, tp=tp)
        path = seg_dir / f"{stem}.nii"
        if not path.exists():
            path = seg_dir / f"{stem}.nii.gz"
        arr = my_data.nib.load(str(path)).get_fdata().astype(np.int16)
        seg = torch.from_numpy(arr)[None, None].to(device).float()
        return seg

    seg_moving = load_seg("01")
    seg_fixed = load_seg("00")
    seg_fixed_i = seg_fixed[0, 0].round().long()

    # --- evaluation displacement: full-res, no half-res round-trip ---
    disp_eval = total_voxel
    disp_their = disp_eval.flip(1)
    seg_their = their_st(seg_moving, disp_their)
    dice_their = multilabel_dice(seg_their[0, 0].round().long(), seg_fixed_i)

    per_label = dice_per_label(
        seg_their[0, 0].round().long(), seg_fixed_i, empty_value=float("nan")
    )

    disp_voxel = disp_eval[0].detach().cpu().numpy().astype(np.float32)
    ndv_mask = Y[0, 0].detach().cpu().numpy() > 0
    ndv = compute_ndv_official(disp_voxel, ndv_mask)

    dice_before = multilabel_dice(seg_moving[0, 0].round().long(), seg_fixed_i)

    # --- same after-registration dice, but on the fast segmentations used by IO ---
    dice_after_fast = float("nan")
    if seg_dir_fast is not None:
        stem_m = ct_label_template.format(case_id=case_id, tp="01")
        stem_f = ct_label_template.format(case_id=case_id, tp="00")

        def resolve_fast(stem: str) -> Path:
            p = seg_dir_fast / f"{stem}.nii.gz"
            return p if p.exists() else seg_dir_fast / f"{stem}.nii"

        fast_m = (
            my_data.nib.load(str(resolve_fast(stem_m))).get_fdata().astype(np.int16)
        )
        fast_f = (
            my_data.nib.load(str(resolve_fast(stem_f))).get_fdata().astype(np.int16)
        )
        seg_moving_fast = torch.from_numpy(fast_m)[None, None].to(device).float()
        seg_fixed_fast = torch.from_numpy(fast_f)[None, None].round().long().to(device)
        seg_their_fast = their_st(seg_moving_fast, disp_their)
        dice_after_fast = multilabel_dice(
            seg_their_fast[0, 0].round().long(), seg_fixed_fast
        )

    tqdm.tqdm.write(
        f"{case_id}: before={dice_before:.4f}\t"
        f"after_fast={dice_after_fast:.4f}\t"
        f"after={dice_their:.4f}"
    )

    fixed_seg_stem = ct_label_template.format(case_id=case_id, tp="00")
    fixed_seg_path = seg_dir / f"{fixed_seg_stem}.nii"
    if not fixed_seg_path.exists():
        fixed_seg_path = seg_dir / f"{fixed_seg_stem}.nii.gz"
    fixed_nii = my_data.nib.load(str(fixed_seg_path))
    spacing_mm = tuple(float(z) for z in fixed_nii.header.get_zooms()[:3])

    fixed_np = seg_fixed_i.cpu().numpy().astype(np.int16)
    warped_np = seg_their[0, 0].round().long().cpu().numpy().astype(np.int16)
    moving_np = seg_moving[0, 0].round().long().cpu().numpy().astype(np.int16)
    hd95 = compute_hd95_official(fixed_np, moving_np, warped_np, spacing_mm)
    # before registration: the "warped" labels are just the moving ones
    hd95_before = compute_hd95_official(fixed_np, moving_np, moving_np, spacing_mm)

    # tqdm.tqdm.write(f"{case_id}: hd95 before={hd95_before:.4f}\thd95 after={hd95:.4f}")

    their_st_lin = SpatialTransformer(size=cfg.img_shape, mode="bilinear").to(device)

    def load_pet_mask(tp: str) -> torch.Tensor:
        stem = pet_label_template.format(case_id=case_id, tp=tp)
        path = seg_dir_fast / f"{stem}.nii.gz"
        if not path.exists():
            path = seg_dir_fast / f"{stem}.nii"
        arr = my_data.nib.load(str(path)).get_fdata()
        mask = (torch.from_numpy(arr)[None, None].to(device).float() > 0).float()
        return mask

    moving_pet_mask = load_pet_mask("01")
    moving_pet_image = X[:, 1:2]
    warped_pet_mask = their_st(moving_pet_mask, disp_their)
    warped_pet_image = their_st_lin(moving_pet_image, disp_their)

    mtv = utils.mtv_bias_loss(warped_pet_mask, moving_pet_mask).item()
    tlg = utils.tlg_bias_loss(
        warped_pet_image, warped_pet_mask, moving_pet_image, moving_pet_mask
    ).item()

    if False:
        # --- DEBUG: verify submitted field via evaluator pipeline ---
        debug_dir = out_dir / "debug_compose"
        X_moving = X[:, 0:1]
        X_submitted = transform(X, total_unit_flow.permute(0, 2, 3, 4, 1), grid_full)[
            :, 0:1
        ]
        Y_fixed = Y[:, 0:1]
        my_data.save_volume(
            X_moving, out_dir=debug_dir, epoch=0, name=f"{case_id}_0_moving"
        )
        my_data.save_volume(
            X_submitted, out_dir=debug_dir, epoch=0, name=f"{case_id}_1_submitted"
        )
        my_data.save_volume(
            Y_fixed, out_dir=debug_dir, epoch=0, name=f"{case_id}_2_fixed"
        )
        # --- END DEBUG ---

    if False:
        # --- DEBUG: save X, Y, warped X, ct label, warped ct label ---
        model_name_clean = model_name.replace(".", "_").replace(" ", "")

        debug_dir = out_dir / "overlap_debug" / model_name_clean
        debug_dir.mkdir(parents=True, exist_ok=True)

        def save_ct(vol: torch.Tensor, name: str) -> None:
            arr = vol[0, 0].detach().cpu().numpy().astype(np.float32)
            path = debug_dir / f"{case_id}_{name}.nii.gz"
            my_data.nib.save(my_data.nib.Nifti1Image(arr, np.eye(4)), str(path))

        save_ct(X, f"{case_id}_X_{model_name_clean}")
        save_ct(Y, f"{case_id}_Y_{model_name_clean}")
        save_ct(warped, f"{case_id}_warped_X_{model_name_clean}")
        save_ct(seg_moving, f"{case_id}_ct_label_moving_{model_name_clean}")
        save_ct(seg_fixed, f"{case_id}_ct_label_fixed_{model_name_clean}")
        save_ct(
            seg_their.round(),
            f"{case_id}_ct_label_moving_warped_{model_name_clean}",
        )

    # submission format: half-resolution displacement field
    disp_half = torch.nn.functional.interpolate(
        total_voxel, scale_factor=0.5, mode="trilinear", align_corners=False
    )
    save_disp(disp_half, out_dir / f"submission_{model_name}", case_id)

    return dice_their, dice_before, mtv, tlg, ndv, hd95, hd95_before, per_label


CONFIGS_REPLACEMENTS: Dict[str, Dict[str, float | bool]] = {
    "secretive-dolphin-38622192": {
        "start_channel": 7,
        "w_ct": 5.0,
        "w_dice_ct_lvl3": 5.0,
        "w_pet": 0.0,
        "w_dice_pet": 0.0,
        "w_bone_rigidity": 2.0,
        "w_jacobian_tumor": 2.0,
        "w_tlg": 2.0,
        "w_jacobian": 2000.0,
        "w_smooth": 2.0,
        "range_flow": 0.4,
        "use_poly_affine": False,
    },
    "sincere-finch-38813192": {
        "start_channel": 7,
        "w_ct": 5.0,
        "w_dice_ct_lvl3": 5.0,
        "w_pet": 0.0,
        "w_dice_pet": 0.0,
        "w_bone_rigidity": 20.0,
        "w_jacobian_tumor": 20.0,
        "w_tlg": 20.0,
        "w_jacobian": 2000.0,
        "w_smooth": 2.0,
        "range_flow": 0.4,
        "use_poly_affine": False,
    },
    "worried-elk-38863657": {
        "start_channel": 7,
        "w_ct": 5.0,
        "w_dice_ct_lvl3": 5.0,
        "w_pet": 0.0,
        "w_dice_pet": 0.0,
        "w_bone_rigidity": 10.0,
        "w_jacobian_tumor": 10.0,
        "w_tlg": 10.0,
        "w_jacobian": 2000.0,
        "w_smooth": 2.0,
        "range_flow": 0.4,
        "use_poly_affine": False,
    },
    "charming-trout-38863973": {
        "start_channel": 7,
        "w_ct": 5.0,
        "w_dice_ct_lvl3": 5.0,
        "w_pet": 0.0,
        "w_dice_pet": 0.0,
        "w_bone_rigidity": 5.0,
        "w_jacobian_tumor": 5.0,
        "w_tlg": 5.0,
        "w_jacobian": 2000.0,
        "w_smooth": 2.0,
        "range_flow": 0.4,
        "use_poly_affine": False,
    },
}


def update_config_from_dict(cfg: TrainingConfig, model_name: str) -> None:

    if model_name not in CONFIGS_REPLACEMENTS:
        print(
            f"Warning: model_name '{model_name}' not found in CONFIGS_REPLACEMENTS. Using default config."
        )
        return

    for key, value in CONFIGS_REPLACEMENTS[model_name].items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
        else:
            raise ValueError(f"Invalid config key: {key}")


models_to_evaluate = [
    # "PSMAReg_LapIRN_polite-snake-38577202_stagelvl3_best.pth",
    # "PSMAReg_LapIRN_exultant-hawk-38756587_stagelvl3_best.pth",
    # "PSMAReg_LapIRN_rumbling-yak-38789486_stagelvl3_best.pth",
    # "PSMAReg_LapIRN_secretive-dolphin-38622192_stagelvl3_best.pth",
    # "PSMAReg_LapIRN_victorious-flea-38622412_stagelvl3_best.pth",
    # "PSMAReg_LapIRN_intelligent-fish-38730451_stagelvl3_best.pth",
    # "PSMAReg_LapIRN_popular-sloth-38758804_stagelvl3_best.pth",
    # "PSMAReg_LapIRN_nosy-doe-38788231_stagelvl3_best.pth",
    # "PSMAReg_LapIRN_sincere-finch-38813192_stagelvl3_best.pth",
    # "PSMAReg_LapIRN_worried-elk-38863657_stagelvl3_best.pth",
    # "PSMAReg_LapIRN_charming-trout-38863973_stagelvl3_best.pth",
    # "PSMAReg_LapIRN_rebellious-stork-38993360_stagelvl3_best.pth",
    "PSMAReg_LapIRN_nervous-pig-39235734_stagelvl3_best_combined.pth",
]

# each inner list is one ensemble: the velocity fields of its models are
# averaged with equal weights (1/n) and integrated once into a displacement.
models_to_average = []
# models_to_average = [["nosy-doe-38788231", "intelligent-fish-38730451"]]


def short_model_name(model_full_name: str) -> str:

    if "combined" in model_full_name:
        short_name = model_full_name.split("_")[2] + "_combined"
    elif "tumour" in model_full_name:
        short_name = model_full_name.split("_")[2] + "_tumour"
    else:
        short_name = model_full_name.split("_")[2]

    return short_name


def model_path_for(model_full_name: str) -> Path:
    return Path(f"/home/iml/fryderyk.koegl/data/PSMAReg/models/{model_full_name}")


def main() -> None:
    cfg = TrainingConfig()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg.cache_dir = val_cache_dir
    cfg.in_channel = 4
    cfg.start_channel = 7
    cfg.resblock_expansion = 1
    cfg.n_resblocks = 5

    # --- what to evaluate ---
    eval_official: bool = True
    eval_my_val: bool = False
    baselines_done: bool = False

    # single models first, then the ensembles
    eval_jobs: List[List[str]] = [[name] for name in models_to_evaluate]
    eval_jobs += [list(names) for names in models_to_average]

    for model_ori_names in eval_jobs:
        if len(model_ori_names) == 1:
            # update_config_from_dict(cfg, model_ori_names[0])
            model_name = short_model_name(model_ori_names[0])
        else:
            short_model_ori_names = [short_model_name(n) for n in model_ori_names]
            model_name = "averaged_" + "_".join(short_model_ori_names)

        out_dir = Path("/home/iml/fryderyk.koegl/code/LapIRN-koegl/submission_results")
        # evaluation metrics use the high-quality segmentations
        official_seg_dir = Path(
            "/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/labelsTs"
        )
        # IO optimization only has access to the fast segmentations
        official_seg_dir_fast = Path(
            "/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/labelsTs"
        )

        # note: create_model reads mutable cfg fields (start_channel, cost_volume_*),
        # so a per-model update_config_from_dict has to happen before each build
        models = [
            create_model(device, cfg, model_path_for(name)) for name in model_ori_names
        ]
        model: Union[torch.nn.Module, List[torch.nn.Module]] = (
            models[0] if len(models) == 1 else models
        )

        grid_full = Functions.generate_grid_unit(cfg.img_shape)
        grid_full = (
            torch.from_numpy(np.reshape(grid_full, (1,) + grid_full.shape))
            .to(device)
            .float()
        )

        transform = miccai2020_model_stage.SpatialTransform_unit().to(device)
        transform_nearest = miccai2020_model_stage.SpatialTransformNearest_unit().to(
            device
        )

        # PET labels (IO and evaluation) come from io_labels_pet: pet_{case_id}_{tp}
        # pet_label_dir = Path("/home/iml/fryderyk.koegl/data/PSMAReg/io_labels_pet")

        use_io: bool = False
        include_pet: bool = True
        include_rigidity: bool = True
        use_class_weights = False
        use_polyaffine: bool = False
        io_lr: float = 2e-1
        io_it: float = 60
        if use_io:
            model_name += "_IO_"
            model_name += f"lr{io_lr:.1e}_it{io_it}_wncc{cfg.w_ct:.2f}_wJac{cfg.w_jacobian:.2f}_wSmooth{cfg.w_smooth:.2f}_wCT{cfg.w_ct:.2f}_wPET{cfg.w_dice_pet:.2f}_wDiceCT{cfg.w_dice_ct_lvl3:.2f}_wDicePET{cfg.w_dice_pet:.2f}_wTLG{cfg.w_tlg:.2f}_wMaskedJac{cfg.w_jacobian_tumor:.2f}_wBoneRigid{cfg.w_bone_rigidity:.2f} "
            print("warning using IO")
            if use_class_weights:
                model_name += "_classweights"
                print("warning using class weights")

        if use_polyaffine:
            model_name += "_polyaffine"
            print("warning using polyaffine prereg")

        # --- baselines: affine-only and affine+polyaffine, no model inference.
        # each runs once, only if its per-label csv is missing.
        def run_baseline(baseline_name: str, baseline_polyaffine: bool) -> None:
            if eval_official:
                per_label_csv = results_csv_official_val_dice_per_label.with_stem(
                    results_csv_official_val_dice_per_label.stem + baseline_name
                )
                per_label_csv = Path(
                    per_label_csv.as_posix().replace("csvs/", "csvs/per_label/")
                )
                if per_label_csv.exists():
                    print(f"skipping baseline '{baseline_name}' (official val): exists")
                else:
                    print(f"running baseline '{baseline_name}' on official val")
                    evaluate_split(
                        subjects=val_subjects,
                        image_dir=val_image_dir,
                        seg_dir=official_seg_dir,
                        seg_dir_fast=official_seg_dir_fast,
                        results_csv=results_csv_official_val_dice,
                        model_name=baseline_name,
                        out_dir=out_dir / f"baseline_{baseline_name}",
                        model=models[0],
                        transform=transform,
                        transform_nearest=transform_nearest,
                        grid_full=grid_full,
                        cfg=cfg,
                        device=device,
                        use_io=False,
                        include_pet=include_pet,
                        include_rigidity=include_rigidity,
                        use_class_weights=False,
                        use_polyaffine=baseline_polyaffine,
                        skip_model=True,
                        ct_label_template="PSMARegPSMA_{case_id}_0000_{tp}",
                        pet_label_template="PSMARegPSMA_{case_id}_0001_{tp}",
                        io_lr=io_lr,
                        io_it=io_it,
                        desc=f"official val [{baseline_name}]",
                        mtv_csv=results_csv_official_mtv,
                        tlg_csv=results_csv_official_tlg,
                        ndv_csv=results_csv_official_ndv,
                        hd95_csv=results_csv_official_hd95,
                        per_label_csv=per_label_csv,
                    )

            if eval_my_val:
                per_label_csv = results_csv_my_val_dice_per_label.with_stem(
                    results_csv_my_val_dice_per_label.stem + baseline_name
                )
                per_label_csv = Path(
                    per_label_csv.as_posix().replace("csvs/", "csvs/per_label/")
                )
                if per_label_csv.exists():
                    print(f"skipping baseline '{baseline_name}' (my val): exists")
                else:
                    print(f"running baseline '{baseline_name}' on my val")
                    _, my_val_subjects = load_split(split_path)
                    evaluate_split(
                        subjects=my_val_subjects,
                        image_dir=my_val_image_dir,
                        seg_dir_fast=my_val_seg_dir,
                        seg_dir=my_val_seg_dir,
                        results_csv=results_csv_my_val_dice,
                        model_name=baseline_name,
                        out_dir=out_dir / "my_val" / f"baseline_{baseline_name}",
                        model=models[0],
                        transform=transform,
                        transform_nearest=transform_nearest,
                        grid_full=grid_full,
                        cfg=cfg,
                        device=device,
                        use_io=False,
                        include_pet=include_pet,
                        include_rigidity=include_rigidity,
                        use_class_weights=False,
                        use_polyaffine=baseline_polyaffine,
                        skip_model=True,
                        ct_label_template="PSMARegPSMA_{case_id}_0000_{tp}",
                        pet_label_template="PSMARegPSMA_{case_id}_0001_{tp}",
                        io_lr=io_lr,
                        io_it=io_it,
                        desc=f"my val [{baseline_name}]",
                        mtv_csv=results_csv_my_val_mtv,
                        tlg_csv=results_csv_my_val_tlg,
                        ndv_csv=results_csv_my_val_ndv,
                        hd95_csv=results_csv_my_val_hd95,
                        per_label_csv=per_label_csv,
                    )

        if baselines_done is False:
            # run_baseline("affine", baseline_polyaffine=False)
            # run_baseline("polyaffine", baseline_polyaffine=True)
            baselines_done = True

        if eval_official:
            # print("warning: reducing number of my val subjects")

            per_label_dice_csv = results_csv_official_val_dice_per_label.with_stem(
                results_csv_official_val_dice_per_label.stem + model_name
            )
            per_label_dice_csv = Path(
                per_label_dice_csv.as_posix().replace("csvs/", "csvs/per_label/")
            )
            evaluate_split(
                subjects=val_subjects,
                image_dir=val_image_dir,
                seg_dir=official_seg_dir,
                seg_dir_fast=official_seg_dir_fast,
                results_csv=results_csv_official_val_dice,
                model_name=model_name,
                model_path=model_path_for(model_ori_names[0]),
                out_dir=out_dir,
                model=model,
                transform=transform,
                transform_nearest=transform_nearest,
                grid_full=grid_full,
                cfg=cfg,
                device=device,
                use_io=use_io,
                include_pet=include_pet,
                include_rigidity=include_rigidity,
                use_class_weights=use_class_weights,
                use_polyaffine=use_polyaffine,
                ct_label_template="PSMARegPSMA_{case_id}_0000_{tp}",
                pet_label_template="PSMARegPSMA_{case_id}_0001_{tp}",
                io_lr=io_lr,
                io_it=io_it,
                desc="official val",
                mtv_csv=results_csv_official_mtv,
                tlg_csv=results_csv_official_tlg,
                ndv_csv=results_csv_official_ndv,
                hd95_csv=results_csv_official_hd95,
                per_label_csv=per_label_dice_csv,
            )
            compress_to_zip(
                out_dir / f"submission_{model_name}",
                out_dir / f"submission_{model_name}.zip",
            )

        if eval_my_val:
            per_label_dice_csv = results_csv_my_val_dice_per_label.with_stem(
                results_csv_my_val_dice_per_label.stem + model_name
            )
            per_label_dice_csv = Path(
                per_label_dice_csv.as_posix().replace("csvs/", "csvs/per_label/")
            )
            _, my_val_subjects = load_split(split_path)
            # my_val_subjects = my_val_subjects[0:1]
            # print("warning: reducing number of my val subjects")
            evaluate_split(
                subjects=my_val_subjects,
                image_dir=my_val_image_dir,
                seg_dir=my_val_seg_dir,
                results_csv=results_csv_my_val_dice,
                model_name=model_name,
                out_dir=out_dir / "my_val",
                model=model,
                transform=transform,
                transform_nearest=transform_nearest,
                grid_full=grid_full,
                cfg=cfg,
                device=device,
                use_io=use_io,
                include_pet=include_pet,
                include_rigidity=include_rigidity,
                use_class_weights=use_class_weights,
                use_polyaffine=use_polyaffine,
                io_it=io_it,
                seg_dir_fast=my_val_seg_dir,
                pet_label_dir=my_val_seg_dir,
                ct_label_template="PSMARegPSMA_{case_id}_0000_{tp}",
                pet_label_template="PSMARegPSMA_{case_id}_0001_{tp}",
                io_lr=io_lr,
                desc="my val",
                mtv_csv=results_csv_my_val_mtv,
                tlg_csv=results_csv_my_val_tlg,
                ndv_csv=results_csv_my_val_ndv,
                hd95_csv=results_csv_my_val_hd95,
                per_label_csv=per_label_dice_csv,
            )


if __name__ == "__main__":
    main()
