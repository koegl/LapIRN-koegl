"""Qualitative baseline comparison on one test pair -> PNG + LaTeX.

Columns: fixed scan, then the moving scan as aligned by Initial, NiftyReg,
ConvexAdam and Ours + IO. Rows:
    1. CT, coronal slice through the spine, with the fixed scan's organ-label
       boundaries drawn on top (misalignment = image edges off the lines)
    2. PET coronal maximum-intensity projection (SUV), warped lesion mask
       outlined (fixed column: the fixed scan and its lesions)
    3. Jacobian determinant of the transform on the same slice (folds black)

The pair (CASE) defaults to the test pair with >= MIN_LESIONS lesions whose
Ours + IO CT Dice is closest to the median: representative, not cherry-picked.
Fields are cached in cache/ (Ours + IO reruns IO, ~3 min on the GPU).

    ../../.venv/bin/python figure_qualitative.py
        ->  out/figures/qualitative.{png,tex}
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import common  # noqa: E402
import evaluate  # noqa: E402
import jacobian  # noqa: E402
from config import TrainingConfig  # noqa: E402

# None = automatic choice; else (case_id, tp_moving, tp_fixed)
CASE: Optional[Tuple[str, str, str]] = None
MIN_LESIONS = 5
# column -> (title, evaluation run)
METHODS: Dict[str, Tuple[str, evaluate.EvalRun]] = {
    "B1": ("Initial", evaluate.EVAL_RUNS["B1"]),
    "B3": ("NiftyReg", evaluate.EVAL_RUNS["B3"]),
    "B4": ("ConvexAdam", evaluate.EVAL_RUNS["B4"]),
    "B6": ("Ours + IO", evaluate.EVAL_RUNS["I4"]),
}
CT_WINDOW = (-400.0, 800.0)  # HU
SUV_MAX = 8.0
JAC_RANGE = (0.0, 2.0)
SPINE_LABELS = tuple(range(26, 51))  # TotalSegmentator vertebrae (incl. sacrum)
LABEL_LINE = "#eda100"
LESION_LINE = "#e34948"
CACHE_DIR = Path(__file__).resolve().parent / "cache"
DATA_DIR = evaluate.DATA_PATH / "PSMAReg/PSMAReg_dataset"


def choose_case() -> Tuple[str, str, str]:
    df = common.load_pairs("B6")
    df = df[df["n_lesions"] >= MIN_LESIONS]
    row = df.loc[(df["dice_ct"] - df["dice_ct"].median()).abs().idxmin()]
    return row["case_id"], f"{int(row['tp_moving']):02d}", f"{int(row['tp_fixed']):02d}"


def load(directory: str, case_id: str, channel: str, tp: str) -> np.ndarray:
    path = DATA_DIR / directory / f"PSMARegPSMA_{case_id}_{channel}_{tp}.nii.gz"
    return np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)


def disp_fields(case: Tuple[str, str, str], device) -> Dict[str, torch.Tensor]:
    """method -> scorer-convention voxel displacement (1, 3, H, W, D)."""
    case_id, tp_x, tp_y = case
    cfg = TrainingConfig()
    cfg.model_save_dir = evaluate.MODEL_DIR
    transform = evaluate.miccai2020_model_stage.SpatialTransform_unit().to(device)
    transform_nearest = evaluate.miccai2020_model_stage.SpatialTransformNearest_unit().to(device)
    grid_full = torch.from_numpy(evaluate.Functions.generate_grid_unit(cfg.img_shape)[None]).to(device).float()
    out = {}
    for m, (_, run) in METHODS.items():
        cache = CACHE_DIR / f"{case_id}_{tp_x}_{tp_y}_{m}.pt"
        if cache.exists():
            out[m] = torch.load(cache, map_location=device)
            continue
        model = None
        if run.network is not None:
            model = evaluate.create_model(cfg, evaluate.model_path_for(cfg, run), device)
        unit, _, _ = evaluate.register_pair(
            model, run, cfg, case_id, tp_x, tp_y, transform, transform_nearest,
            grid_full, device, history_csv=CACHE_DIR / f"io_history_{case_id}_{tp_x}.csv",
        )
        voxel = evaluate.Functions.transform_unit_flow_to_flow_cuda(
            unit.permute(0, 2, 3, 4, 1).clone()
        ).permute(0, 4, 1, 2, 3)
        out[m] = voxel.flip(1).detach()  # as in evaluate.evaluate_pair
        CACHE_DIR.mkdir(exist_ok=True)
        torch.save(out[m], cache)
    return out


def coronal(volume: np.ndarray, y: Optional[int] = None) -> np.ndarray:
    """(R, A, S) volume -> coronal image, superior up, patient right on the
    left (radiological); y = None -> maximum-intensity projection."""
    return display(volume.max(axis=1) if y is None else volume[:, y, :])


def display(plane: np.ndarray) -> np.ndarray:
    """(R, S) plane -> image rows S (superior up), columns R reversed."""
    return plane.T[::-1, ::-1]


def boundaries(labels: np.ndarray) -> np.ndarray:
    edge = np.zeros(labels.shape, bool)
    edge[:-1] |= labels[:-1] != labels[1:]
    edge[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    return edge & (labels > 0)


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    case = CASE or choose_case()
    case_id, tp_x, tp_y = case
    print(f"pair {case_id} {tp_x}->{tp_y}")
    fields = disp_fields(case, device)

    ct = {"moving": load("imagesTr", case_id, "0000", tp_x), "fixed": load("imagesTr", case_id, "0000", tp_y)}
    pet = {"moving": load("imagesTr", case_id, "0001", tp_x), "fixed": load("imagesTr", case_id, "0001", tp_y)}
    lesions = {"moving": load("labelsTr", case_id, "0001", tp_x) == 1, "fixed": load("labelsTr", case_id, "0001", tp_y) == 1}
    fixed_labels = load("labelsTr", case_id, "0000", tp_y).round().astype(np.int16)

    shape = tuple(ct["fixed"].shape)
    warp = {
        "bilinear": evaluate.SpatialTransformer(shape, mode="bilinear").to(device),
        "nearest": evaluate.SpatialTransformer(shape, mode="nearest").to(device),
    }

    def warped(volume: np.ndarray, m: str, mode: str) -> np.ndarray:
        if not METHODS[m][1].warp:  # Initial: not resampled
            return volume
        src = torch.from_numpy(volume.astype(np.float32))[None, None].to(device)
        return warp[mode](src, fields[m])[0, 0].cpu().numpy()

    # coronal slice through the fixed spine; crop to the body
    y = int(np.round(np.nonzero(np.isin(fixed_labels, SPINE_LABELS))[1].mean()))
    body = coronal(ct["fixed"] > -500.0, None)
    rows_any, cols_any = np.nonzero(body.any(axis=1))[0], np.nonzero(body.any(axis=0))[0]
    crop = (slice(rows_any[0], rows_any[-1] + 1), slice(cols_any[0], cols_any[-1] + 1))
    zooms = nib.load(str(DATA_DIR / "imagesTr" / f"PSMARegPSMA_{case_id}_0000_{tp_y}.nii.gz")).header.get_zooms()
    aspect = float(zooms[2] / zooms[0])
    fixed_edges = display(boundaries(fixed_labels[:, y, :]))[crop]

    columns = ["fixed"] + list(METHODS)
    titles = ["Fixed"] + [METHODS[m][0] for m in METHODS]
    fig, axes = plt.subplots(3, len(columns), figsize=(7.0, 6.4))
    for j, col in enumerate(columns):
        if col == "fixed":
            ct_img, pet_img, les_img, jac = ct["fixed"], pet["fixed"], lesions["fixed"], None
        else:
            ct_img = warped(ct["moving"], col, "bilinear")
            pet_img = warped(pet["moving"], col, "bilinear")
            les_img = warped(lesions["moving"].astype(np.float32), col, "nearest") > 0.5
            det, _ = jacobian.jacobian_matrix(fields[col].permute(0, 2, 3, 4, 1))
            jac = det[0, 0].cpu().numpy()

        ax = axes[0, j]
        ax.imshow(coronal(ct_img, y)[crop], cmap="gray", vmin=CT_WINDOW[0], vmax=CT_WINDOW[1], aspect=aspect)
        overlay = np.ma.masked_where(~fixed_edges, fixed_edges)
        ax.imshow(overlay, cmap=matplotlib.colors.ListedColormap([LABEL_LINE]), aspect=aspect, alpha=0.8, interpolation="nearest")
        ax.set_title(titles[j], fontsize=8)

        ax = axes[1, j]
        ax.imshow(coronal(pet_img)[crop], cmap="gray_r", vmin=0, vmax=SUV_MAX, aspect=aspect)
        ax.contour(coronal(les_img.astype(np.float32))[crop], levels=[0.5], colors=LESION_LINE, linewidths=0.6)

        ax = axes[2, j]
        if jac is None:
            ax.axis("off")
            continue
        image = ax.imshow(coronal(jac, y)[crop], cmap="RdBu_r", vmin=JAC_RANGE[0], vmax=JAC_RANGE[1], aspect=aspect)
        folds = coronal(jac, y)[crop] <= 0
        ax.imshow(np.ma.masked_where(~folds, folds), cmap=matplotlib.colors.ListedColormap(["black"]), aspect=aspect, interpolation="nearest")
        print(f"{col}: folding voxels on slice {int(folds.sum())}, in volume {int((jac <= 0).sum())}")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    for i, label in enumerate(["CT", "PET MIP", r"$\det J$"]):
        axes[i, 0].set_ylabel(label, fontsize=8)
    cax = axes[2, 0].inset_axes([0.45, 0.1, 0.1, 0.8])
    fig.colorbar(image, cax=cax, ticks=[0, 1, 2])
    cax.tick_params(labelsize=7)
    cax.set_title(r"$\det J$", fontsize=7)
    fig.tight_layout(pad=0.3, w_pad=0.2, h_pad=0.4)

    out_dir = common.OUT_DIR / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "qualitative.png", dpi=300, bbox_inches="tight")
    tex = rf"""\begin{{figure}}[t]
\centering
\includegraphics[width=\textwidth]{{figures/qualitative.png}}
\caption{{Test pair (patient {case_id}, follow-up {tp_x} $\rightarrow$ baseline {tp_y}), chosen as the pair with at least {MIN_LESIONS} lesions whose CT Dice for Ours + IO is closest to the median. Top: CT, coronal slice through the spine, with the fixed scan's organ-label boundaries (yellow). Middle: PET maximum-intensity projection (SUV 0--{SUV_MAX:g}) with the warped lesion mask (red); left: fixed scan and its lesions. Bottom: Jacobian determinant on the same slice (folds in black). Initial: the moving scan, not resampled.}}
\label{{fig:qualitative}}
\end{{figure}}
"""
    (out_dir / "qualitative.tex").write_text(tex)
    print(tex)


if __name__ == "__main__":
    main()
