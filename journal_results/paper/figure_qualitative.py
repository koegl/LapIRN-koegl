"""Qualitative example: one test pair registered by our method -> PNG + LaTeX.

The pair (CASE) has a large change in tumor volume between the scans. Panels,
one coronal slice index (fixed frame; the moving scan shown at the same index
of its own frame):
    row 1: fixed (CT with PET), moving (CT with PET), CT overlay of fixed and
           moving, CT overlay of fixed and warped moving (fixed green, other
           magenta: gray where they agree)
    row 2: label residual (fixed minus moving / warped moving label ID, black
           where they agree); Jacobian determinant of the transform; warped
           moving (CT with PET)
Ground-truth lesion masks are drawn as red contours on the CT + PET panels.
The slice maximizes the smaller of the fixed and warped lesion voxel counts on
it, so lesions are visible in both.

The field is the one saved by the evaluation (evaluate.FIELD_DIR, official
scorer convention), copied from lustre to cache/ as
{case}_{tp_moving}_{tp_fixed}_{RUN}_official.npz; without it, it is computed
here (runs with IO take ~3 min on the GPU; set CUDA_VISIBLE_DEVICES= to compute
on the CPU while the GPU is busy).

    ../../.venv/bin/python figure_qualitative.py  ->  out/figures/qualitative.{png,tex}
"""

from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

# isort: off
import common  # noqa: E402  (puts Code/ on the path)
import evaluate  # noqa: E402
import jacobian  # noqa: E402
import synthetic  # noqa: E402
from config import TrainingConfig  # noqa: E402
# isort: on

# (case_id, tp_moving, tp_fixed): tumor volume 2400 -> 24191 voxels (x10)
CASE: Tuple[str, str, str] = ("0146", "01", "00")
# evaluation run of the method shown: T4 = network trained with PET + rigidity
# losses (no IO); I4 = the same followed by IO with both losses; I7 = network
# trained without losses followed by IO with both losses ("Ours")
RUN = "I7"
RUN_LABEL = {
    "T4": "our network (PET and rigidity losses in training, no IO)",
    "I4": "our network with PET and rigidity losses in training and IO",
    "I7": "our method (PET and rigidity losses only in IO)",
}
CT_WINDOW = (-400.0, 800.0)  # HU
SUV_RANGE = (0.0, 8.0)  # PET, magma, no threshold
PET_ALPHA = 0.35
LESION_COLOR = "#e8000b"
# label residual: shade per label group (bone full, organ medium, muscle dark)
MUSCLE_LABELS = tuple(range(80, 90))  # gluteus, autochthon, iliopsoas
GROUP_SHADE = np.full(118, 0.6)  # organs (incl. vessels)
SACRUM = 25  # bone, but not in the rigidity label set
GROUP_SHADE[list(synthetic.BONE_LABEL_VALUES) + [SACRUM]] = 1.0
GROUP_SHADE[list(MUSCLE_LABELS)] = 0.35
JAC_RANGE = (0.0, 2.0)
FIG_WIDTH = 6.5  # in, text width
GAP_W, GAP_H, TITLE_H = 0.05, 0.05, 0.18  # in
CACHE_DIR = Path(__file__).resolve().parent / "cache"
if not torch.cuda.is_available():
    # the model code calls .cuda() directly; without a GPU, keep everything on
    # the CPU (only used here, to avoid competing with GPU jobs)
    torch.nn.Module.cuda = lambda self, *args, **kwargs: self
    torch.Tensor.cuda = lambda self, *args, **kwargs: self
DATA_DIR = evaluate.DATA_PATH / "PSMAReg/PSMAReg_dataset"


def load(directory: str, case_id: str, channel: str, tp: str) -> np.ndarray:
    path = DATA_DIR / directory / f"PSMARegPSMA_{case_id}_{channel}_{tp}.nii.gz"
    return np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)


def disp_field(device) -> torch.Tensor:
    """Scorer-convention voxel displacement (1, 3, H, W, D) of RUN on CASE."""
    case_id, tp_x, tp_y = CASE
    saved = CACHE_DIR / f"{case_id}_{tp_x}_{tp_y}_{RUN}_official.npz"
    if saved.exists():
        return torch.from_numpy(np.load(saved)["disp"])[None].to(device)
    cache = CACHE_DIR / f"{case_id}_{tp_x}_{tp_y}_{RUN}_official.pt"
    if cache.exists():
        return torch.load(cache, map_location=device)
    cfg = TrainingConfig()
    cfg.model_save_dir = evaluate.MODEL_DIR
    run = evaluate.EVAL_RUNS[RUN]
    transform = evaluate.miccai2020_model_stage.SpatialTransform_unit().to(device)
    transform_nearest = (
        evaluate.miccai2020_model_stage.SpatialTransformNearest_unit().to(device)
    )
    grid_full = (
        torch.from_numpy(evaluate.Functions.generate_grid_unit(cfg.img_shape)[None])
        .to(device)
        .float()
    )
    model = evaluate.create_model(cfg, evaluate.model_path_for(cfg, run), device)
    unit, _, _ = evaluate.register_pair(
        model,
        run,
        cfg,
        case_id,
        tp_x,
        tp_y,
        transform,
        transform_nearest,
        grid_full,
        device,
        history_csv=CACHE_DIR / f"io_history_{case_id}_{tp_x}.csv",
    )
    field = evaluate.unit_flow_to_eval_disp(unit).detach()  # as in the evaluation
    CACHE_DIR.mkdir(exist_ok=True)
    torch.save(field, cache)
    return field


def display(plane: np.ndarray) -> np.ndarray:
    """(R, S) plane -> image rows S (superior up), patient right on the left."""
    return plane.T[::-1, ::-1]


def ct_gray(ct: np.ndarray) -> np.ndarray:
    return np.clip((ct - CT_WINDOW[0]) / (CT_WINDOW[1] - CT_WINDOW[0]), 0, 1)


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    case_id, tp_x, tp_y = CASE
    field = disp_field(device)
    shape = tuple(field.shape[2:])
    warpers = {
        m: evaluate.SpatialTransformer(shape, mode=m).to(device)
        for m in ("bilinear", "nearest")
    }

    def warp(volume: np.ndarray, mode: str) -> np.ndarray:
        src = torch.from_numpy(volume.astype(np.float32))[None, None].to(device)
        with torch.no_grad():
            return warpers[mode](src, field)[0, 0].cpu().numpy()

    ct = {
        "fixed": load("imagesTr", case_id, "0000", tp_y),
        "moving": load("imagesTr", case_id, "0000", tp_x),
    }
    pet = {
        "fixed": load("imagesTr", case_id, "0001", tp_y),
        "moving": load("imagesTr", case_id, "0001", tp_x),
    }
    organs = {
        "fixed": load("labelsTr", case_id, "0000", tp_y).round(),
        "moving": load("labelsTr", case_id, "0000", tp_x).round(),
    }
    lesions = {
        "fixed": load("labelsTr", case_id, "0001", tp_y) == 1,
        "moving": load("labelsTr", case_id, "0001", tp_x) == 1,
    }
    ct["warped"] = warp(ct["moving"], "bilinear")
    pet["warped"] = warp(pet["moving"], "bilinear")
    organs["warped"] = warp(organs["moving"], "nearest").round()
    lesions["warped"] = warp(lesions["moving"].astype(np.float32), "nearest") > 0.5
    with torch.no_grad():
        det, _ = jacobian.jacobian_matrix(field.permute(0, 2, 3, 4, 1))
    jac = det[0, 0].cpu().numpy()

    per_slice = np.minimum(
        lesions["fixed"].sum(axis=(0, 2)), lesions["warped"].sum(axis=(0, 2))
    )
    y = int(np.argmax(per_slice))
    planes = {
        name: {k: display(v[:, y, :]) for k, v in d.items()}
        for name, d in (
            ("ct", ct),
            ("pet", pet),
            ("organs", organs),
            ("lesions", lesions),
        )
    }
    jac_plane = display(jac[:, y, :])
    body = planes["ct"]["fixed"] > -500
    rows_any, cols_any = (
        np.nonzero(body.any(axis=1))[0],
        np.nonzero(body.any(axis=0))[0],
    )
    crop = (slice(rows_any[0], rows_any[-1] + 1), slice(cols_any[0], cols_any[-1] + 1))
    zooms = nib.load(
        str(DATA_DIR / "imagesTr" / f"PSMARegPSMA_{case_id}_0000_{tp_y}.nii.gz")
    ).header.get_zooms()
    aspect = float(zooms[2] / zooms[0])

    plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 8})
    # size the figure to the panels: full text width, minimal gaps
    panel_w = (FIG_WIDTH - 3 * GAP_W) / 4
    panel_h = (
        panel_w
        * (crop[0].stop - crop[0].start)
        * aspect
        / (crop[1].stop - crop[1].start)
    )
    fig_h = 2 * panel_h + 2 * TITLE_H + GAP_H
    fig, axes = plt.subplots(2, 4, figsize=(FIG_WIDTH, fig_h))
    fig.subplots_adjust(
        left=0,
        right=1,
        bottom=0,
        top=1 - TITLE_H / fig_h,
        wspace=GAP_W / panel_w,
        hspace=(TITLE_H + GAP_H) / panel_h,
    )

    def show_ct_pet(ax, key):
        ax.imshow(
            planes["ct"][key][crop],
            cmap="gray",
            vmin=CT_WINDOW[0],
            vmax=CT_WINDOW[1],
            aspect=aspect,
        )
        suv = planes["pet"][key][crop]
        ax.imshow(
            suv,
            cmap="magma",
            vmin=SUV_RANGE[0],
            vmax=SUV_RANGE[1],
            alpha=PET_ALPHA,
            aspect=aspect,
        )
        ax.contour(
            planes["lesions"][key][crop].astype(float),
            levels=[0.5],
            colors=LESION_COLOR,
            linewidths=0.5,
        )

    def show_overlay(ax, other):
        f, o = ct_gray(planes["ct"]["fixed"][crop]), ct_gray(planes["ct"][other][crop])
        ax.imshow(
            np.stack([o, f, o], axis=-1), aspect=aspect
        )  # fixed green, other magenta

    def show_labels(ax, other) -> float:
        """Label residual, colored as the CT overlay: green where the fixed
        scan's organ label is not matched by the other scan (unlabelled there
        or a different organ), magenta where only the other scan is labeled;
        black where the labels agree. The shade gives the group of the shown
        label: bone (full), organ (medium), muscle (dark)."""
        a, b = (
            planes["organs"]["fixed"][crop].astype(int),
            planes["organs"][other][crop].astype(int),
        )
        fixed_unmatched = (a > 0) & (a != b)
        other_only = (a == 0) & (b > 0)
        rgb = np.zeros(a.shape + (3,))
        for mask, label, color in (
            (fixed_unmatched, a, (0.0, 1.0, 0.0)),
            (other_only, b, (1.0, 0.0, 1.0)),
        ):
            rgb[mask] = np.array(color) * GROUP_SHADE[label[mask]][:, None]
        ax.imshow(rgb, aspect=aspect, interpolation="nearest")
        labeled = (a > 0) | (b > 0)
        return 100 * (fixed_unmatched | other_only).sum() / max(labeled.sum(), 1)

    show_ct_pet(axes[0, 0], "fixed")
    show_ct_pet(axes[0, 1], "moving")
    show_overlay(axes[0, 2], "moving")
    show_overlay(axes[0, 3], "warped")
    res_before = show_labels(axes[1, 0], "moving")
    res_after = show_labels(axes[1, 1], "warped")
    image = axes[1, 2].imshow(
        jac_plane[crop],
        cmap="RdBu_r",
        vmin=JAC_RANGE[0],
        vmax=JAC_RANGE[1],
        aspect=aspect,
    )
    show_ct_pet(axes[1, 3], "warped")

    titles = [
        ["Fixed (baseline)", "Moving (follow-up)", "Fixed / moving", "Fixed / warped"],
        ["Label residual before", "Label residual after", r"$\det J$", "Warped moving"],
    ]
    for i in range(2):
        for j in range(4):
            axes[i, j].set_title(
                f"({'abcdefgh'[4 * i + j]}) {titles[i][j]}", fontsize=8
            )
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
    # color bar inside the det J panel, at the bottom
    cax = axes[1, 2].inset_axes([0.15, 0.04, 0.7, 0.025])
    fig.colorbar(image, cax=cax, ticks=[0, 1, 2], orientation="horizontal")
    cax.tick_params(labelsize=6, pad=1, length=2)

    out_dir = common.OUT_DIR / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "qualitative.png", dpi=300, bbox_inches="tight", pad_inches=0)
    print(
        f"run {RUN}, slice y={y} (lesion voxels on slice: fixed {lesions['fixed'][:, y].sum()}, "
        f"warped {lesions['warped'][:, y].sum()}); labels differ on slice {res_before:.1f} % -> {res_after:.1f} %; "
        f"det J <= 0 voxels {(jac <= 0).sum()}"
    )
    tex = rf"""\begin{{figure}}[p]
\centering
\includegraphics[width=\textwidth]{{figures/qualitative.png}}
\caption{{Anatomical alignment example from the test set, registered by {RUN_LABEL[RUN]}; one coronal slice.
Patient {case_id} was selected for its large between-image change in lesion volume ({lesions["fixed"].sum()} voxels at baseline and {lesions["moving"].sum()} at follow-up), without considering the registration result.
The example illustrates anatomical alignment and deformation.
(a)~Fixed and (b)~moving CT with PET (SUV {SUV_RANGE[0]:g}--{SUV_RANGE[1]:g}, {100 * PET_ALPHA:.0f}\,\% opacity) and provided lesion contours (red).
(c,~d)~CT of the fixed image (green) overlaid with the moving image (magenta) before~(c) and after~(d) registration; gray where they agree.
(e,~f)~Organ labels that differ before~(e) and after~(f) registration: green where the fixed image has a label that the moving image lacks there, magenta where only the moving image is labeled; bright, medium, and dark shades for bones, organs, and muscles.
(g)~Jacobian determinant of the fixed-to-moving sampling transform: white is volume-preserving (1), blue is below 1 and red is above 1; changes in the displayed warped anatomy have the opposite direction.
(h)~Warped moving CT with PET and warped lesion contours.}}
\label{{fig:qualitative}}
\end{{figure}}
"""
    (out_dir / "qualitative.tex").write_text(tex)


if __name__ == "__main__":
    main()
