"""Offline diagnostics on an already-submitted displacement field.

Two questions this answers, both from a submission zip instead of a fresh
inference run (which is expensive):

  1. Which CT labels actually drive the HD95 mean? HD95 is an unweighted mean
     over labels, so a handful of grossly displaced labels can dominate a score
     that looks like a uniform boundary problem. The per-label dump separates
     "broad 3-8 mm bulk" (a boundary-refinement problem, worth a differentiable
     surface loss) from "long tail" (a capture-range / label-matching problem,
     which no boundary loss fixes).

  2. How much of the MTV / TLG bias currently cancels across lesions? Both
     metrics are global sums over the whole lesion mask, so a lesion that
     expands can hide one that contracts. If sum_c |bias_c| >> |sum_c bias_c|
     the reported bias understates the true per-lesion error and a per-component
     term is worth adding; if the two agree, per-component work buys nothing.

Nothing here touches inference.py / instance_opt.py: it reloads the saved
half-resolution field and replays exactly the warps the scorer performs.
"""

import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hd95_official
import my_data
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
from scipy import ndimage

# --- variables (define here, no argparse; mirrors inference.py) ------------
VAL_IMAGE_DIR = Path("/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/imagesTs")
# evaluation-quality segmentations, same ones the scorer uses for HD95 / dice
SEG_DIR = Path("/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/labelsTs")
# fast segmentations: where the PET lesion masks live
SEG_DIR_FAST = Path("/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/labelsTs")

CT_LABEL_TEMPLATE = "PSMARegPSMA_{case_id}_0000_{tp}"
PET_LABEL_TEMPLATE = "PSMARegPSMA_{case_id}_0001_{tp}"

IMG_SHAPE: Tuple[int, int, int] = (192, 192, 288)
LABEL_IDS = range(1, 118)

# per-label HD95 of the *unwarped* moving labels, for reference. Doubles the
# surface-distance work, which is the slow part of this script.
COMPUTE_BEFORE: bool = True

# 6-connectivity for lesion connected components. ndimage's default; stated
# explicitly because 26-connectivity merges lesions that touch only at corners
# and would change the component count this whole diagnostic is about.
CC_STRUCTURE: Optional[np.ndarray] = None


class SpatialTransformer(torch.nn.Module):
    """Copy of inference.SpatialTransformer (voxelmorph).

    Duplicated rather than imported: importing inference pulls in
    totalsegmentator and the autopet repo, which this script does not need.
    """

    def __init__(self, size: Tuple[int, int, int], mode: str = "bilinear") -> None:
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
        new_locs = new_locs.permute(0, 2, 3, 4, 1)
        new_locs = new_locs[..., [2, 1, 0]]
        return F.grid_sample(src, new_locs, align_corners=False, mode=self.mode)


def unpack_submission(zip_path: Path, work_dir: Path) -> Dict[str, Path]:
    """Extract the zip and map case_id -> displacement file.

    Files are named disp_{case}_00_{case}_01.nii.gz (see inference.save_disp);
    a nested top-level folder inside the zip is tolerated.
    """
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(work_dir)

    cases: Dict[str, Path] = {}
    for path in sorted(work_dir.rglob("disp_*.nii*")):
        stem = path.name.split(".")[0]
        parts = stem.split("_")
        if len(parts) < 2:
            continue
        cases[parts[1]] = path
    return cases


def load_disp_their(disp_path: Path, device: torch.device) -> torch.Tensor:
    """Load a submitted field and return it in SpatialTransformer convention.

    inference.save_disp writes disp_np[::-1] (channel order reversed) of the
    half-resolution field, and the evaluation path applies total_voxel.flip(1) —
    the same reversal. So the array on disk is already `disp_their`: no further
    flip is needed here.

    The stored magnitudes are full-resolution voxels on a half-resolution grid,
    and upsampling a unit flow then rescaling equals interpolating the voxel
    field directly (the scale is a per-channel constant that commutes with
    linear interpolation), so a plain trilinear x2 reproduces the field the
    scorer sees.
    """
    arr = nib.load(str(disp_path)).get_fdata().astype(np.float32)
    assert arr.shape[0] == 3, f"expected (3, X, Y, Z), got {arr.shape}"
    disp = torch.from_numpy(arr)[None].to(device)

    spatial = tuple(disp.shape[2:])
    half_shape = tuple(s // 2 for s in IMG_SHAPE)
    if spatial == half_shape:
        disp = F.interpolate(
            disp, size=IMG_SHAPE, mode="trilinear", align_corners=False
        )
    elif spatial != IMG_SHAPE:
        raise ValueError(f"{disp_path.name}: unexpected shape {spatial}")
    return disp


def resolve(directory: Path, stem: str) -> Path:
    path = directory / f"{stem}.nii.gz"
    if not path.exists():
        path = directory / f"{stem}.nii"
    return path


def load_seg(case_id: str, tp: str, device: torch.device) -> torch.Tensor:
    stem = CT_LABEL_TEMPLATE.format(case_id=case_id, tp=tp)
    arr = nib.load(str(resolve(SEG_DIR, stem))).get_fdata().astype(np.int16)
    return torch.from_numpy(arr)[None, None].to(device).float()


def load_pet_mask(case_id: str, tp: str, device: torch.device) -> torch.Tensor:
    stem = PET_LABEL_TEMPLATE.format(case_id=case_id, tp=tp)
    arr = nib.load(str(resolve(SEG_DIR_FAST, stem))).get_fdata()
    return (torch.from_numpy(arr)[None, None].to(device).float() > 0).float()


def load_pet_image(case_id: str, tp: str, device: torch.device) -> torch.Tensor:
    """Moving PET exactly as inference.load_val_pair prepares it: body-masked
    with the CT's own body mask, then SUV-normalized. TLG bias is a ratio so the
    normalization cancels, but the body mask does not."""
    ct_path = VAL_IMAGE_DIR / f"PSMARegPSMA_{case_id}_0000_{tp}.nii.gz"
    pet_path = VAL_IMAGE_DIR / f"PSMARegPSMA_{case_id}_0001_{tp}.nii.gz"
    ct_raw = nib.load(str(ct_path)).get_fdata().astype(np.float32)
    pet_raw = nib.load(str(pet_path)).get_fdata().astype(np.float32)
    mask = my_data.get_body_mask(ct_raw)
    pet = my_data.apply_body_mask(pet_raw, mask, fill_value=0.0)
    pet = my_data.norm_pet(pet)
    return torch.from_numpy(pet)[None, None].to(device).float()


def per_label_hd95(
    fixed: np.ndarray,
    moving: np.ndarray,
    warped: np.ndarray,
    spacing_mm: Tuple[float, float, float],
    case_id: str,
) -> List[Dict[str, float]]:
    """Per-label HD95 with the official scorer's own gates.

    A label is scored only when present in both fixed and *unwarped* moving
    (hd95_official.compute_average_ct_label_hd95), and non-finite scores are
    dropped from the mean — an emptied warped label silently leaves the average,
    so n_warped is recorded to make that visible.
    """
    rows: List[Dict[str, float]] = []
    for label_id in LABEL_IDS:
        fixed_mask = fixed == label_id
        moving_mask = moving == label_id
        if not fixed_mask.any() or not moving_mask.any():
            continue
        warped_mask = warped == label_id

        hd95 = float("inf")
        if warped_mask.any():
            sd = hd95_official.compute_surface_distances(
                fixed_mask, warped_mask, spacing_mm
            )
            hd95 = float(hd95_official.compute_robust_hausdorff(sd, 95.0))

        hd95_before = float("nan")
        if COMPUTE_BEFORE:
            sd_before = hd95_official.compute_surface_distances(
                fixed_mask, moving_mask, spacing_mm
            )
            hd95_before = float(hd95_official.compute_robust_hausdorff(sd_before, 95.0))

        volume_sum = int(warped_mask.sum()) + int(fixed_mask.sum())
        dice = (
            float("nan")
            if volume_sum == 0
            else 2.0 * float((warped_mask & fixed_mask).sum()) / volume_sum
        )

        rows.append(
            {
                "case": case_id,
                "label": int(label_id),
                "hd95": hd95,
                "hd95_before": hd95_before,
                "dice": dice,
                "n_fixed": int(fixed_mask.sum()),
                "n_moving": int(moving_mask.sum()),
                "n_warped": int(warped_mask.sum()),
                "scored": bool(np.isfinite(hd95)),
            }
        )
    return rows


def per_cc_mtv_tlg(
    moving_mask: torch.Tensor,
    pet_image: torch.Tensor,
    warped_pet_image: torch.Tensor,
    disp_their: torch.Tensor,
    st_nearest: SpatialTransformer,
    case_id: str,
) -> List[Dict[str, float]]:
    """Decompose the global MTV / TLG bias over lesion connected components.

    The component id map is warped once with nearest interpolation; because the
    lesion mask is the union of its components, counting per id gives exactly
    the decomposition of the warped mask the scorer builds.
    """
    moving_np = moving_mask[0, 0].cpu().numpy() > 0.5
    cc_np, n_cc = ndimage.label(moving_np, structure=CC_STRUCTURE)
    if n_cc == 0:
        return []

    cc = torch.from_numpy(cc_np.astype(np.float32))[None, None].to(moving_mask.device)
    warped_cc = st_nearest(cc, disp_their)[0, 0].round().long().cpu().numpy()

    pet_moving_np = pet_image[0, 0].cpu().numpy()
    pet_warped_np = warped_pet_image[0, 0].cpu().numpy()

    rows: List[Dict[str, float]] = []
    for cc_id in range(1, n_cc + 1):
        m_sel = cc_np == cc_id
        w_sel = warped_cc == cc_id
        n_moving = int(m_sel.sum())
        n_warped = int(w_sel.sum())
        tlg_moving = float(pet_moving_np[m_sel].sum())
        tlg_warped = float(pet_warped_np[w_sel].sum())
        rows.append(
            {
                "case": case_id,
                "cc": cc_id,
                "n_moving": n_moving,
                "n_warped": n_warped,
                "mtv_bias": (n_warped - n_moving) / max(n_moving, 1),
                "tlg_moving": tlg_moving,
                "tlg_warped": tlg_warped,
                # signed on purpose: the point of this diagnostic is whether the
                # signs cancel when summed into the global metric
                "tlg_bias": (tlg_warped - tlg_moving) / max(tlg_moving, 1e-5),
            }
        )
    return rows


def analyse_case(
    case_id: str,
    disp_path: Path,
    device: torch.device,
    st_nearest: SpatialTransformer,
    st_linear: SpatialTransformer,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]], Dict[str, float]]:
    disp_their = load_disp_their(disp_path, device)

    seg_moving = load_seg(case_id, "01", device)
    seg_fixed = load_seg(case_id, "00", device)
    seg_warped = st_nearest(seg_moving, disp_their)

    fixed_stem = CT_LABEL_TEMPLATE.format(case_id=case_id, tp="00")
    spacing_mm = tuple(
        float(z)
        for z in nib.load(str(resolve(SEG_DIR, fixed_stem))).header.get_zooms()[:3]
    )

    fixed_np = seg_fixed[0, 0].round().long().cpu().numpy().astype(np.int16)
    moving_np = seg_moving[0, 0].round().long().cpu().numpy().astype(np.int16)
    warped_np = seg_warped[0, 0].round().long().cpu().numpy().astype(np.int16)

    label_rows = per_label_hd95(fixed_np, moving_np, warped_np, spacing_mm, case_id)

    moving_pet_mask = load_pet_mask(case_id, "01", device)
    moving_pet_image = load_pet_image(case_id, "01", device)
    warped_pet_mask = st_nearest(moving_pet_mask, disp_their)
    warped_pet_image = st_linear(moving_pet_image, disp_their)

    cc_rows = per_cc_mtv_tlg(
        moving_pet_mask,
        moving_pet_image,
        warped_pet_image,
        disp_their,
        st_nearest,
        case_id,
    )

    # global metrics, computed the way the scorer does (nearest mask, bilinear
    # image). Reproduce the numbers already in the results CSVs — a mismatch
    # means this script's field reconstruction is wrong, so check it first.
    n_moving = float(moving_pet_mask.sum().item())
    n_warped = float((warped_pet_mask > 0.5).float().sum().item())
    tlg_moving = float((moving_pet_image * moving_pet_mask).sum().item())
    tlg_warped = float(
        (warped_pet_image * (warped_pet_mask > 0.5).float()).sum().item()
    )
    mtv_global = abs(n_warped - n_moving) / max(n_moving, 1.0)
    tlg_global = abs(tlg_warped - tlg_moving) / max(tlg_moving, 1e-5)

    scored = [r["hd95"] for r in label_rows if r["scored"]]
    summary = {
        "case": case_id,
        "hd95_mean": float(np.mean(scored)) if scored else float("nan"),
        "n_labels_scored": len(scored),
        "n_labels_dropped": sum(1 for r in label_rows if not r["scored"]),
        "n_cc": len(cc_rows),
        "mtv_global": mtv_global,
        "tlg_global": tlg_global,
    }
    if cc_rows:
        # |sum| is what the metric charges; sum|.| is the error actually present.
        # Their ratio is the cancellation currently working in our favour.
        mtv_signed = [r["mtv_bias"] * r["n_moving"] for r in cc_rows]
        tlg_signed = [r["tlg_bias"] * r["tlg_moving"] for r in cc_rows]
        total_vox = sum(r["n_moving"] for r in cc_rows)
        total_tlg = sum(r["tlg_moving"] for r in cc_rows)
        summary["mtv_abs_sum"] = abs(sum(mtv_signed)) / max(total_vox, 1)
        summary["mtv_sum_abs"] = sum(abs(v) for v in mtv_signed) / max(total_vox, 1)
        summary["tlg_abs_sum"] = abs(sum(tlg_signed)) / max(total_tlg, 1e-5)
        summary["tlg_sum_abs"] = sum(abs(v) for v in tlg_signed) / max(total_tlg, 1e-5)
    return label_rows, cc_rows, summary


def report(
    label_df: pd.DataFrame, cc_df: pd.DataFrame, summary_df: pd.DataFrame
) -> None:
    print("\n=== HD95 ===")
    scored = label_df[label_df["scored"]]
    if not scored.empty:
        pcts = np.percentile(scored["hd95"], [50, 75, 90, 95, 99])
        print(
            "per-label hd95 mm: median {:.2f}  p75 {:.2f}  p90 {:.2f}  "
            "p95 {:.2f}  p99 {:.2f}  max {:.2f}".format(*pcts, scored["hd95"].max())
        )
        # how top-heavy is the mean? if the worst decile carries most of it, the
        # problem is a tail (capture range), not a bulk (boundary refinement)
        ordered = np.sort(scored["hd95"].to_numpy())[::-1]
        k = max(1, len(ordered) // 10)
        print(
            f"worst 10% of (case,label) entries carry "
            f"{100 * ordered[:k].sum() / ordered.sum():.1f}% of the hd95 total"
        )
        print("\ntop 15 labels by mean hd95:")
        by_label = (
            scored.groupby("label")["hd95"]
            .agg(["mean", "max", "count"])
            .sort_values("mean", ascending=False)
            .head(15)
        )
        print(by_label.to_string())
    dropped = int((~label_df["scored"]).sum())
    if dropped:
        print(f"\n{dropped} label instances vanished after warping (dropped from mean)")

    print("\n=== MTV / TLG cancellation across lesions ===")
    print(f"cases with >1 lesion: {(summary_df['n_cc'] > 1).sum()} / {len(summary_df)}")
    for metric in ("mtv", "tlg"):
        abs_sum = summary_df.get(f"{metric}_abs_sum")
        sum_abs = summary_df.get(f"{metric}_sum_abs")
        if abs_sum is None or sum_abs is None:
            continue
        ratio = sum_abs / abs_sum.replace(0.0, np.nan)
        print(
            f"{metric}: mean |sum_c| = {abs_sum.mean():.4f}   "
            f"mean sum_c|.| = {sum_abs.mean():.4f}   "
            f"median ratio = {ratio.median():.2f}"
        )
    print(
        "\nratio ~1 -> nothing cancels, per-component terms buy nothing.\n"
        "ratio >>1 -> the global metric understates the true per-lesion error."
    )


def main() -> None:
    zip_path = Path(
        "/home/iml/fryderyk.koegl/code/LapIRN-koegl/submission_results/submission_polite-snake-38577202_IO_lr1.0e-02_it90_wNCC6.00_wDiceCT6.00_wJac10.00_wSmooth0.00_wBoneRigid0.00_wMTV200.00_wMTVmean0.00_wJactum20.00_wTLG20.00_wMTVcc80.00_wMTVavgcc150.00_wTLGcc20.00.zip"
    )
    out_dir = Path(
        "/home/iml/fryderyk.koegl/code/LapIRN-koegl/submission_results/csvs/analysis"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st_nearest = SpatialTransformer(size=IMG_SHAPE, mode="nearest").to(device)
    st_linear = SpatialTransformer(size=IMG_SHAPE, mode="bilinear").to(device)

    work_dir = Path(tempfile.mkdtemp(prefix="disp_analysis_"))
    try:
        cases = unpack_submission(zip_path, work_dir)
        print(f"{len(cases)} cases in {zip_path.name}")

        label_rows: List[Dict[str, float]] = []
        cc_rows: List[Dict[str, float]] = []
        summaries: List[Dict[str, float]] = []
        for case_id, disp_path in tqdm.tqdm(sorted(cases.items()), desc="cases"):
            labels, ccs, summary = analyse_case(
                case_id, disp_path, device, st_nearest, st_linear
            )
            label_rows.extend(labels)
            cc_rows.extend(ccs)
            summaries.append(summary)
            tqdm.tqdm.write(
                f"{case_id}: hd95={summary['hd95_mean']:.3f} "
                f"mtv={summary['mtv_global']:.4f} tlg={summary['tlg_global']:.4f} "
                f"n_cc={summary['n_cc']}"
            )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    label_df = pd.DataFrame(label_rows)
    cc_df = pd.DataFrame(cc_rows)
    summary_df = pd.DataFrame(summaries)

    label_df.to_csv(out_dir / "per_label_hd95.csv", index=False)
    cc_df.to_csv(out_dir / "per_cc_mtv_tlg.csv", index=False)
    summary_df.to_csv(out_dir / "per_case_summary.csv", index=False)
    print(f"\nwrote 3 csvs to {out_dir}")

    report(label_df, cc_df, summary_df)


if __name__ == "__main__":
    main()
