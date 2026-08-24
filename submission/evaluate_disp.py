"""Score a directory of container-produced displacement fields.

Runs on the HOST (not inside the container), in the repo's own environment:

    python submission/evaluate_disp.py submission/validation_predictions

It reads every `disp_<id>_00_<id>_01.nii.gz` in the directory and reproduces the
metrics of Code/inference.py:process_subject -- dice, HD95, NDV, MTV bias and
TLG bias -- by warping the reference segmentations with the field exactly the
way the organizers' scorer does. Nothing about the field's provenance matters,
so this scores the container output, an old submission or a baseline alike.

Fields at less than full resolution are upsampled first, mirroring the scorer.

`--pet-seg-dir` additionally scores the container's PET lesion masks against the
reference labels, so a segmentation regression shows up here rather than only as
a worse registration much later.

`--compare` prints the corresponding row of an earlier chase_leaderboard run
next to the new numbers. Those were produced from a half-resolution field, so
small differences are expected; large ones are not.
"""

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "Code"))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import hd95_official  # noqa: E402
import my_data  # noqa: E402
import ndv_official  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import utils  # noqa: E402
from config import TrainingConfig  # noqa: E402

DISP_RE = re.compile(r"^disp_(?P<id>\w+?)_00_(?P=id)_01\.nii\.gz$")

DEFAULT_IMAGE_DIR = Path(
    "/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/imagesTs"
)
DEFAULT_SEG_DIR = Path("/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/labelsTs")
DEFAULT_COMPARE_DIR = REPO_ROOT / "submission_results" / "csvs" / "chase_leaderboard"

CT_TEMPLATE = "PSMARegPSMA_{case_id}_0000_{tp}"
PET_TEMPLATE = "PSMARegPSMA_{case_id}_0001_{tp}"

METRICS = ["dice", "dice_in_crop", "dice_out_crop", "dice_before", "hd95",
           "hd95_before", "ndv", "mtv", "tlg"]
SEG_METRICS = ["pet_dice", "pet_pred_vox", "pet_ref_vox"]


def binary_dice(gt: np.ndarray, pred: np.ndarray) -> float:
    """Lesion dice, matching autopet-3-submission/inference_all.py:dice_score --
    two empty masks score 1.0 rather than NaN, which is how the 0.5260 / 0.4576
    numbers those models were compared on were computed."""
    gt, pred = gt > 0, pred > 0
    total = gt.sum() + pred.sum()
    if total == 0:
        return 1.0
    return float(2.0 * np.logical_and(gt, pred).sum() / total)


def score_segmentations(
    pet_seg_dir: Path, label_dir: Path, case_ids: List[str]
) -> Dict[str, Dict[str, float]]:
    """PET lesion dice per case, predicted vs reference labels.

    Only the moving timepoint (01) is segmented, so this is one row per case and
    merges straight into the registration metric table.
    """
    out: Dict[str, Dict[str, float]] = {}
    for case_id in case_ids:
        name = f"{PET_TEMPLATE.format(case_id=case_id, tp='01')}.nii.gz"
        pred_path, ref_path = pet_seg_dir / name, label_dir / name
        if not pred_path.exists() or not ref_path.exists():
            continue
        pred = nib.load(str(pred_path)).get_fdata()
        ref = nib.load(str(ref_path)).get_fdata()
        out[case_id] = {
            "pet_dice": binary_dice(ref, pred),
            "pet_pred_vox": float((pred > 0).sum()),
            "pet_ref_vox": float((ref > 0).sum()),
        }
    return out


class SpatialTransformer(torch.nn.Module):
    """The organizers' warper (voxelmorph). Copied from Code/inference.py so the
    scoring path here is byte-for-byte the one used there."""

    def __init__(self, size, mode="bilinear"):
        super().__init__()
        self.mode = mode
        vectors = [torch.arange(0, s) for s in size]
        grid = torch.stack(torch.meshgrid(*vectors, indexing="ij"))
        self.register_buffer("grid", grid.unsqueeze(0).type(torch.FloatTensor))

    def forward(self, src: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        new_locs = self.grid + flow
        shape = flow.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        new_locs = new_locs.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]
        return torch.nn.functional.grid_sample(
            src, new_locs, align_corners=False, mode=self.mode
        )


def multilabel_dice(
    pred: torch.Tensor,
    target: torch.Tensor,
    label_ids: range = range(1, 118),
    present_only: bool = False,
) -> float:
    """Mean dice over the FIXED label set, matching the official scorer and
    Code/inference.py:multilabel_dice. Labels absent from both volumes score 0,
    not "skip" -- averaging only over present labels would inflate the result.

    present_only=True skips absent labels instead, and is ONLY for comparing
    sub-volumes. A z-band holds a fraction of the 117 structures, so scoring the
    missing ones as 0 makes the result a measure of how many labels the band
    happens to contain rather than of registration quality. Never use it for a
    whole-volume score -- it is not what the scorer computes.
    """
    dices = []
    for lbl in label_ids:
        p, t = pred == lbl, target == lbl
        volume_sum = p.sum() + t.sum()
        if volume_sum == 0:
            if not present_only:
                dices.append(0.0)
            continue
        dices.append((2.0 * (p & t).sum() / volume_sum).item())
    return float(np.mean(dices)) if dices else float("nan")


def compute_ndv(disp_voxel: np.ndarray, mask: np.ndarray) -> float:
    identity = ndv_official.get_identity_grid(disp_voxel)
    jac_dets = ndv_official.calc_jac_dets(disp_voxel + identity)
    mask_inner = mask[1:-1, 1:-1, 1:-1] > 0
    _, _, non_diff_volume, _ = ndv_official.calc_measurements(jac_dets, mask_inner)
    return float(non_diff_volume / float(mask_inner.sum()) * 100.0)


def compute_hd95(
    fixed: np.ndarray,
    moving: np.ndarray,
    warped: np.ndarray,
    spacing: Tuple[float, ...],
) -> float:
    """Delegates to the official implementation, as Code/inference.py does."""
    return float(
        hd95_official.compute_average_ct_label_hd95(fixed, moving, warped, spacing)
    )


def load_field(path: Path, full_shape: Tuple[int, int, int], device) -> torch.Tensor:
    """Read a submission-format field -> (1, 3, H, W, D), full-res voxel units.

    The stored channel order is already what the scorer's warper consumes, so no
    flip happens here; `disp_official` below re-derives the other convention.
    """
    arr = nib.load(str(path)).get_fdata().astype(np.float32)
    if arr.ndim != 4 or arr.shape[0] != 3:
        raise ValueError(
            f"{path.name}: expected channel-first (3,X,Y,Z), got {arr.shape}"
        )
    disp = torch.from_numpy(arr).unsqueeze(0).to(device).float()
    if tuple(disp.shape[2:]) != tuple(full_shape):
        # the organizers upsample a sub-resolution field; magnitudes are already
        # in full-res voxel units, so a plain trilinear resize is exact
        print(f"  {path.name}: {tuple(disp.shape[2:])} -> {full_shape} (upsampled)")
        disp = torch.nn.functional.interpolate(
            disp, size=full_shape, mode="trilinear", align_corners=False
        )
    return disp


def load_images(
    image_dir: Path, case_id: str, device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Moving/fixed 2-channel volumes, normalised as in Code/inference.py:load_val_pair."""

    def load(mod: str, tp: str) -> np.ndarray:
        name = (CT_TEMPLATE if mod == "ct" else PET_TEMPLATE).format(
            case_id=case_id, tp=tp
        )
        return (
            nib.load(str(image_dir / f"{name}.nii.gz")).get_fdata().astype(np.float32)
        )

    x_ct, y_ct = load("ct", "01"), load("ct", "00")
    x_mask, y_mask = my_data.get_body_mask(x_ct), my_data.get_body_mask(y_ct)
    x_ct = my_data.apply_body_mask(x_ct, x_mask, fill_value=my_data.CT_AIR_HU)
    y_ct = my_data.apply_body_mask(y_ct, y_mask, fill_value=my_data.CT_AIR_HU)
    x_pet = my_data.apply_body_mask(load("pet", "01"), x_mask, fill_value=0.0)
    y_pet = my_data.apply_body_mask(load("pet", "00"), y_mask, fill_value=0.0)

    def stack(ct: np.ndarray, pet: np.ndarray) -> torch.Tensor:
        return torch.stack(
            [
                torch.from_numpy(my_data.norm_ct(ct)),
                torch.from_numpy(my_data.norm_pet(pet)),
            ]
        ).unsqueeze(0)

    return stack(x_ct, x_pet).to(device).float(), stack(y_ct, y_pet).to(device).float()


def load_label(
    seg_dir: Path, template: str, case_id: str, tp: str, device
) -> torch.Tensor:
    stem = template.format(case_id=case_id, tp=tp)
    path = seg_dir / f"{stem}.nii.gz"
    if not path.exists():
        path = seg_dir / f"{stem}.nii"
    arr = nib.load(str(path)).get_fdata().astype(np.int16)
    return torch.from_numpy(arr)[None, None].to(device).float()


def evaluate_case(
    case_id: str,
    disp_path: Path,
    image_dir: Path,
    seg_dir: Path,
    cfg: TrainingConfig,
    device,
) -> Dict[str, float]:
    full_shape = tuple(cfg.img_shape)
    warp_nearest = SpatialTransformer(size=full_shape, mode="nearest").to(device)
    warp_linear = SpatialTransformer(size=full_shape, mode="bilinear").to(device)

    disp_their = load_field(disp_path, full_shape, device)
    # NDV uses the other component order (channel 0 <-> last spatial axis), the
    # one Code/inference.py calls the "official" order
    disp_official = disp_their.flip(1)

    X, Y = load_images(image_dir, case_id, device)

    seg_moving = load_label(seg_dir, CT_TEMPLATE, case_id, "01", device)
    seg_fixed = load_label(seg_dir, CT_TEMPLATE, case_id, "00", device)
    seg_warped = warp_nearest(seg_moving, disp_their)

    fixed_i = seg_fixed[0, 0].round().long()
    warped_i = seg_warped[0, 0].round().long()
    dice = multilabel_dice(warped_i, fixed_i)
    dice_before = multilabel_dice(seg_moving[0, 0].round().long(), fixed_i)

    # Inside vs outside the axial band the IO dice term could see. The CT labels
    # the container produces are cropped to cfg.io_seg_crop, so the term
    # supervises only that band while this score covers the whole volume. If the
    # in-band dice improves and the out-of-band dice does not, the crop is what
    # limits the term -- not its weight or the label quality.
    lo, hi = cfg.io_seg_crop, full_shape[2] - cfg.io_seg_crop
    dice_in = multilabel_dice(
        warped_i[:, :, lo:hi], fixed_i[:, :, lo:hi], present_only=True
    )
    dice_out = multilabel_dice(
        torch.cat([warped_i[:, :, :lo], warped_i[:, :, hi:]], dim=2),
        torch.cat([fixed_i[:, :, :lo], fixed_i[:, :, hi:]], dim=2),
        present_only=True,
    )

    spacing = nib.load(
        str(seg_dir / f"{CT_TEMPLATE.format(case_id=case_id, tp='00')}.nii.gz")
    ).header.get_zooms()[:3]
    fixed_np = fixed_i.cpu().numpy().astype(np.int16)
    moving_np = seg_moving[0, 0].round().long().cpu().numpy().astype(np.int16)
    warped_np = seg_warped[0, 0].round().long().cpu().numpy().astype(np.int16)
    hd95 = compute_hd95(
        fixed_np, moving_np, warped_np, tuple(float(s) for s in spacing)
    )
    hd95_before = compute_hd95(
        fixed_np, moving_np, moving_np, tuple(float(s) for s in spacing)
    )

    ndv = compute_ndv(
        disp_official[0].cpu().numpy().astype(np.float32),
        Y[0, 0].cpu().numpy() > 0,
    )

    moving_pet_mask = (
        load_label(seg_dir, PET_TEMPLATE, case_id, "01", device) > 0
    ).float()
    moving_pet_image = X[:, 1:2]
    warped_pet_mask = warp_nearest(moving_pet_mask, disp_their)
    warped_pet_image = warp_linear(moving_pet_image, disp_their)
    mtv = utils.mtv_bias_loss(warped_pet_mask, moving_pet_mask).item()
    tlg = utils.tlg_bias_loss(
        warped_pet_image, warped_pet_mask, moving_pet_image, moving_pet_mask
    ).item()

    return {
        "dice": dice,
        "dice_in_crop": dice_in,
        "dice_out_crop": dice_out,
        "dice_before": dice_before,
        "hd95": hd95,
        "hd95_before": hd95_before,
        "ndv": ndv,
        "mtv": mtv,
        "tlg": tlg,
    }


def load_reference(compare_dir: Path, model_row: str) -> Dict[str, Dict[str, float]]:
    """Per-case values of an earlier run, keyed metric -> case_id -> value."""
    out: Dict[str, Dict[str, float]] = {}
    for metric, fname in (
        ("dice", "results_official_val_dice.csv"),
        ("hd95", "results_official_val_hd95.csv"),
        ("ndv", "results_official_val_ndv.csv"),
        ("mtv", "results_official_val_mtv.csv"),
        ("tlg", "results_official_val_tlg.csv"),
    ):
        path = compare_dir / fname
        if not path.exists():
            continue
        with path.open() as fh:
            for row in csv.DictReader(fh):
                if row[next(iter(row))] != model_row:
                    continue
                values = {}
                for key, value in row.items():
                    m = re.match(rf"^{metric}_(\w+)$", key or "")
                    if m and value not in (None, ""):
                        values[m.group(1)] = float(value)
                out[metric] = values
                break
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("disp_dir", type=Path, help="directory of disp_*.nii.gz")
    p.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    p.add_argument(
        "--label-dir",
        type=Path,
        default=DEFAULT_SEG_DIR,
        help="reference labels: the CT segmentations the registration metrics "
        "warp, and the PET masks --pet-seg-dir is scored against",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="per-case results plus a trailing mean row "
        "(default: <disp_dir>/metrics.csv)",
    )
    p.add_argument(
        "--compare",
        metavar="MODEL_ROW",
        default=None,
        help="model row in the chase_leaderboard CSVs to print alongside, "
        "e.g. auspicious-sloth-39469081_combined",
    )
    p.add_argument("--compare-dir", type=Path, default=DEFAULT_COMPARE_DIR)
    p.add_argument(
        "--pet-seg-dir",
        type=Path,
        default=None,
        help="container-written PET lesion masks, scored against --label-dir",
    )
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    cfg = TrainingConfig()

    cases: List[Tuple[str, Path]] = []
    for path in sorted(args.disp_dir.iterdir()):
        m = DISP_RE.match(path.name)
        if m:
            cases.append((m.group("id"), path))
    if not cases:
        raise SystemExit(f"no disp_<id>_00_<id>_01.nii.gz files in {args.disp_dir}")

    reference = load_reference(args.compare_dir, args.compare) if args.compare else {}

    print(f"scoring {len(cases)} case(s) from {args.disp_dir}\n")

    # PET segmentation scores become extra columns of the same table rather than
    # a separate block, so one CSV holds everything. Scored up front (it is only
    # mask loading) so the columns are known before the slow loop starts and
    # rows can stream as they are computed -- HD95 alone is ~20 s per case.
    columns = list(METRICS)
    seg_scores: Dict[str, Dict[str, float]] = {}
    if args.pet_seg_dir:
        seg_scores = score_segmentations(
            args.pet_seg_dir, args.label_dir, [c for c, _ in cases]
        )
        if seg_scores:
            columns += SEG_METRICS
        else:
            print(f"no PET masks matched in {args.pet_seg_dir}\n")

    def fmt(metric: str, value: float) -> str:
        return f"{value:>12.0f}" if metric.endswith("_vox") else f"{value:>12.4f}"

    header = f"{'case':>6}  " + "  ".join(f"{m:>12}" for m in columns)
    print(header)
    print("-" * len(header))

    results: Dict[str, Dict[str, float]] = {}
    for case_id, path in cases:
        res = evaluate_case(case_id, path, args.image_dir, args.label_dir, cfg, device)
        res.update(seg_scores.get(case_id, {}))
        results[case_id] = res
        print(
            f"{case_id:>6}  "
            + "  ".join(fmt(m, res.get(m, float("nan"))) for m in columns),
            flush=True,
        )

    print("-" * len(header))
    means = {
        m: float(np.nanmean([r.get(m, float("nan")) for r in results.values()]))
        for m in columns
    }
    print(f"{'mean':>6}  " + "  ".join(fmt(m, means[m]) for m in columns))

    if any(m in columns for m in SEG_METRICS):
        print(
            "\nPET dice reference: nnU-Net 0.5260 / autopet 0.4576 on the held-out\n"
            "split. Cases where both masks are empty score 1.0, so a val set with few\n"
            "lesions reads higher than those numbers -- read pet_*_vox alongside it."
        )

    if reference:
        common = [m for m in METRICS if m in reference]
        ref_means = {
            m: float(
                np.nanmean([reference[m][c] for c in results if c in reference[m]])
            )
            for m in common
        }
        print(f"\nreference: {args.compare}  (chase_leaderboard, half-res field)")
        print(f"{'':>6}  " + "  ".join(f"{m:>12}" for m in common))
        print(f"{'ref':>6}  " + "  ".join(f"{ref_means[m]:>12.4f}" for m in common))
        print(
            f"{'delta':>6}  "
            + "  ".join(f"{means[m] - ref_means[m]:>+12.4f}" for m in common)
        )

    out_csv = args.out_csv or args.disp_dir / "metrics.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["case"] + columns)
        for case_id, res in results.items():
            writer.writerow([case_id] + [res.get(m, "") for m in columns])
        writer.writerow(["mean"] + [means[m] for m in columns])
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
