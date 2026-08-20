"""Coronal visualisation of a DVF: fixed / moving / warped moving as greyscale
PNGs plus the deformed grid as a green overlay on a transparent background.
The warped moving image is also written as a .nii.gz.

All PNGs share the same field of view and pixel size, so they can be overlaid
directly.

Frames
------
io_on_train.py saves the IO field in the RESIDUAL frame: it maps the
*affine-preregistered* moving image onto the fixed one, not the raw moving
image. So the chain here is

    moving --(cached ANTs affine DVF)--> moving_prereg --(disp_unit)--> warped

Skipping the affine step is why a raw-moving warp does not line up with the
fixed image at all.

"disp_unit" is (3, H, W, D) in grid_sample convention: channel 0 displaces
along the last axis, channel 1 along the second, channel 2 along the first, in
normalised [-1, 1] units with align_corners=False.

The deformed grid is drawn from the TOTAL field (affine composed with the
residual), not from the residual alone: total(x) = d(x) + a(x + d(x)).

As a sanity check the script prints the multilabel Dice (same definition as
instance_opt.multilabel_dice, which produced the io_summary.csv numbers)
before and after the deformable step.
"""

from pathlib import Path

import matplotlib
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DATA = Path("/home/iml/fryderyk.koegl/data/PSMAReg")
CASE_ID = "0053"
TP_MOVING = "01"  # tp_x
TP_FIXED = "00"  # tp_y

DISP_PATH = DATA / f"io_train_dvfs/dvf_{CASE_ID}_{TP_MOVING}_{TP_FIXED}.pt"
DATASET_DIR = DATA / "PSMAReg_dataset"
FIXED_PATH = DATASET_DIR / f"imagesTr/PSMARegPSMA_{CASE_ID}_0000_{TP_FIXED}.nii.gz"
MOVING_PATH = DATASET_DIR / f"imagesTr/PSMARegPSMA_{CASE_ID}_0000_{TP_MOVING}.nii.gz"
FIXED_LABEL_PATH = (
    DATASET_DIR / f"labelsTr/PSMARegPSMA_{CASE_ID}_0000_{TP_FIXED}.nii.gz"
)
MOVING_LABEL_PATH = (
    DATASET_DIR / f"labelsTr/PSMARegPSMA_{CASE_ID}_0000_{TP_MOVING}.nii.gz"
)
# PET of the same timepoints (channel 0001), on the same grid as the CT
FIXED_PET_PATH = DATASET_DIR / f"imagesTr/PSMARegPSMA_{CASE_ID}_0001_{TP_FIXED}.nii.gz"
MOVING_PET_PATH = (
    DATASET_DIR / f"imagesTr/PSMARegPSMA_{CASE_ID}_0001_{TP_MOVING}.nii.gz"
)
# voxel-displacement affine DVF cached by affine_reg.get_affine_dvf
AFFINE_DVF_PATH = DATA / f"affine_cache/affine_{CASE_ID}_{TP_MOVING}_{TP_FIXED}.npy"

OUT_DIR = Path("/home/iml/fryderyk.koegl/code/LapIRN-koegl/paper_results/dvf_vis")

APPLY_AFFINE_PREREG = True  # False -> warp the raw moving image (will not align)
COMPUTE_DICE = True
LABEL_IDS = range(1, 118)  # same fixed label set as the official scorer

SLICE = None  # coronal slice index (anterior-posterior axis); None -> middle
GRID_SPACING = 8  # grid line spacing [vox]
GRID_COLOR = "#00bd81"
GRID_LW = 2.0
DPI = 200
WINDOW = None  # (min, max) intensity window; None -> 1/99 percentile of fixed


def load_nifti(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    """Raw array + image, exactly as the training pipeline loads it (no
    reorientation), so voxel indices match the displacement fields."""
    img = nib.load(str(path))
    return np.asanyarray(img.dataobj, dtype=np.float32), img


def check_ras(img: nib.Nifti1Image) -> None:
    """The coronal view below assumes axis0 -> right, axis1 -> anterior,
    axis2 -> superior (diagonal, positive affine)."""
    aff = img.affine[:3, :3]
    if not (np.allclose(aff, np.diag(np.diag(aff))) and np.all(np.diag(aff) > 0)):
        print(f"warning: affine is not diagonal-positive RAS:\n{img.affine}")


def load_disp(path: Path, shape: tuple[int, int, int]) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    disp = payload["disp_unit"] if isinstance(payload, dict) else payload
    disp = disp.float()
    if disp.dim() == 5:
        disp = disp[0]
    if tuple(disp.shape[1:]) != tuple(shape):
        raise ValueError(f"disp shape {tuple(disp.shape)} does not match image {shape}")
    return disp


def load_affine_flow(path: Path, shape: tuple[int, int, int]) -> torch.Tensor:
    """Cached affine DVF (H, W, D, 3) in voxel displacements -> unit flow
    (3, H, W, D), mirroring affine_reg.create_affine_flow."""
    if not path.exists():
        raise FileNotFoundError(
            f"no cached affine DVF at {path} - run the training/IO pipeline for "
            "this pair once, or set APPLY_AFFINE_PREREG = False"
        )
    dvf = np.load(str(path)).astype(np.float32)
    if tuple(dvf.shape[:3]) != tuple(shape):
        raise ValueError(f"affine dvf shape {dvf.shape} does not match image {shape}")
    h, w, d = shape
    dvf_t = torch.from_numpy(dvf).permute(3, 0, 1, 2)  # (3, H, W, D), voxels
    return torch.stack(
        (dvf_t[2] / (d / 2.0), dvf_t[1] / (w / 2.0), dvf_t[0] / (h / 2.0))
    )


def compose(disp: torch.Tensor, affine_flow: torch.Tensor) -> torch.Tensor:
    """Total field of the two-step warp moving -> prereg -> fixed frame.

    warped(x) = moving_prereg(x + d(x)) and moving_prereg(y) = moving(y + a(y)),
    so the total displacement is total(x) = d(x) + a(x + d(x)): the affine flow
    resampled at the residually displaced positions, plus the residual.
    All in unit coordinates."""
    grid = unit_grid(tuple(disp.shape[1:])) + disp.permute(1, 2, 3, 0)[None]
    a_at_d = F.grid_sample(
        affine_flow[None],
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )[0]
    return disp + a_at_d


def unit_grid(shape: tuple[int, int, int]) -> torch.Tensor:
    """Sampling grid in grid_sample convention, align_corners=False."""
    axes = [(torch.arange(s, dtype=torch.float32) + 0.5) / s * 2 - 1 for s in shape]
    gh, gw, gd = torch.meshgrid(*axes, indexing="ij")
    # last channel is (x, y, z) = (last axis, middle axis, first axis)
    return torch.stack((gd, gw, gh), dim=-1)[None]


def warp(volume: np.ndarray, disp: torch.Tensor, mode: str = "bilinear") -> np.ndarray:
    src = torch.from_numpy(volume.astype(np.float32))[None, None]
    grid = unit_grid(tuple(volume.shape)) + disp.permute(1, 2, 3, 0)[None]
    out = F.grid_sample(
        src, grid, mode=mode, padding_mode="border", align_corners=False
    )
    return out[0, 0].numpy()


def multilabel_dice(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean Dice over a fixed label set, matching instance_opt.multilabel_dice."""
    pred = np.rint(pred).astype(np.int32)
    target = np.rint(target).astype(np.int32)
    dices = []
    for lbl in LABEL_IDS:
        p = pred == lbl
        t = target == lbl
        volume_sum = p.sum() + t.sum()
        dices.append(0.0 if volume_sum == 0 else 2.0 * (p & t).sum() / volume_sum)
    return float(np.mean(dices)) if dices else float("nan")


def new_canvas(
    extent_x: tuple[float, float], extent_y: tuple[float, float]
) -> tuple[plt.Figure, plt.Axes]:
    """Axes filling the whole figure, sized in mm. The view direction is set by
    finalize_canvas() *after* the content is drawn."""
    width_mm = abs(extent_x[1] - extent_x[0])
    height_mm = abs(extent_y[1] - extent_y[0])
    scale = 2.0  # px per mm
    fig = plt.figure(figsize=(width_mm * scale / DPI, height_mm * scale / DPI), dpi=DPI)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    ax.set_aspect("equal")
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    return fig, ax


def finalize_canvas(
    ax: plt.Axes, extent_x: tuple[float, float], extent_y: tuple[float, float]
) -> None:
    """Radiological-style coronal view, as 3D Slicer shows it: patient right on
    the left of the image, superior up.

    Must run AFTER everything is drawn: imshow() re-applies its extent to the
    axis limits while autoscaling is on, which silently undoes an x-flip set
    beforehand - ax.plot() does not, so the grid and the images would end up
    mirrored against each other."""
    ax.set_xlim(max(extent_x), min(extent_x))
    ax.set_ylim(min(extent_y), max(extent_y))
    ax.set_autoscale_on(False)


def save_image(
    volume: np.ndarray,
    y_index: int,
    zooms: np.ndarray,
    window: tuple[float, float],
    out_path: Path,
) -> None:
    sl = volume[:, y_index, :]  # (R, S)
    ex_x = (0.0, volume.shape[0] * zooms[0])
    ex_y = (0.0, volume.shape[2] * zooms[2])
    fig, ax = new_canvas(ex_x, ex_y)
    ax.imshow(
        sl.T,
        cmap="gray",
        origin="lower",
        extent=(ex_x[0], ex_x[1], ex_y[0], ex_y[1]),
        vmin=window[0],
        vmax=window[1],
        interpolation="bilinear",
    )
    finalize_canvas(ax, ex_x, ex_y)
    fig.savefig(out_path, dpi=DPI, transparent=True)
    plt.close(fig)
    print(f"wrote {out_path}")


def save_grid(
    disp: torch.Tensor,
    shape: tuple[int, int, int],
    y_index: int,
    zooms: np.ndarray,
    out_path: Path,
) -> None:
    nx, _, nz = shape
    # unit -> voxel displacement (align_corners=False): d_vox = d_unit * n / 2
    d_r = disp[2, :, y_index, :].numpy() * nx / 2  # along axis 0 (right)
    d_s = disp[0, :, y_index, :].numpy() * nz / 2  # along axis 2 (superior)

    ii, kk = np.meshgrid(np.arange(nx), np.arange(nz), indexing="ij")
    # deformed positions of the source grid, in mm. The +0.5 puts the nodes on
    # voxel centres, matching imshow's extent (which spans voxel edges).
    x = (ii + 0.5 + d_r) * zooms[0]
    y = (kk + 0.5 + d_s) * zooms[2]

    ex_x = (0.0, nx * zooms[0])
    ex_y = (0.0, nz * zooms[2])
    fig, ax = new_canvas(ex_x, ex_y)

    for i in range(0, nx, GRID_SPACING):
        ax.plot(
            x[i, :],
            y[i, :],
            color=GRID_COLOR,
            linewidth=GRID_LW,
            solid_joinstyle="round",
        )
    for k in range(0, nz, GRID_SPACING):
        ax.plot(
            x[:, k],
            y[:, k],
            color=GRID_COLOR,
            linewidth=GRID_LW,
            solid_joinstyle="round",
        )

    finalize_canvas(ax, ex_x, ex_y)
    fig.savefig(out_path, dpi=DPI, transparent=True)
    plt.close(fig)
    print(f"wrote {out_path}")


def save_nifti(volume: np.ndarray, reference: nib.Nifti1Image, out_path: Path) -> None:
    img = nib.Nifti1Image(volume.astype(np.float32), reference.affine, reference.header)
    img.set_data_dtype(np.float32)
    nib.save(img, str(out_path))
    print(f"wrote {out_path}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fixed, fixed_img = load_nifti(FIXED_PATH)
    moving, moving_img = load_nifti(MOVING_PATH)
    y_index = SLICE if SLICE is not None else fixed.shape[1] // 2

    check_ras(fixed_img)
    if fixed.shape != moving.shape:
        raise ValueError(f"shape mismatch: fixed {fixed.shape}, moving {moving.shape}")
    zooms = np.asarray(fixed_img.header.get_zooms()[:3], dtype=np.float64)

    disp = load_disp(DISP_PATH, fixed.shape)

    if APPLY_AFFINE_PREREG:
        affine_flow = load_affine_flow(AFFINE_DVF_PATH, fixed.shape)
        moving_prereg = warp(moving, affine_flow)
    else:
        print("APPLY_AFFINE_PREREG = False: warping the raw moving image")
        affine_flow = None
        moving_prereg = moving

    warped = warp(moving_prereg, disp)

    # the PET shares the CT grid, so it goes through the very same chain
    moving_pet, moving_pet_img = load_nifti(MOVING_PET_PATH)
    if moving_pet.shape != fixed.shape:
        raise ValueError(
            f"PET shape {moving_pet.shape} does not match CT {fixed.shape}"
        )
    warped_pet = warp(
        warp(moving_pet, affine_flow) if affine_flow is not None else moving_pet, disp
    )
    # the grid shows the total transform the moving image undergoes
    total = compose(disp, affine_flow) if affine_flow is not None else disp

    if COMPUTE_DICE:
        moving_lbl, _ = load_nifti(MOVING_LABEL_PATH)
        fixed_lbl, _ = load_nifti(FIXED_LABEL_PATH)
        if affine_flow is not None:
            moving_lbl = warp(moving_lbl, affine_flow, mode="nearest")
        warped_lbl = warp(moving_lbl, disp, mode="nearest")
        dice_before = multilabel_dice(moving_lbl, fixed_lbl)
        dice_after = multilabel_dice(warped_lbl, fixed_lbl)
        print(f"dice before deformable warp (affine only): {dice_before:.4f}")
        print(f"dice after  deformable warp:                {dice_after:.4f}")

    window = WINDOW or (float(np.percentile(fixed, 1)), float(np.percentile(fixed, 99)))
    print(f"coronal slice {y_index}/{fixed.shape[1]}, window {window}")

    save_image(fixed, y_index, zooms, window, OUT_DIR / "fixed.png")
    save_image(moving, y_index, zooms, window, OUT_DIR / "moving.png")
    save_image(moving_prereg, y_index, zooms, window, OUT_DIR / "moving_prereg.png")
    save_image(warped, y_index, zooms, window, OUT_DIR / "warped.png")
    save_grid(total, fixed.shape, y_index, zooms, OUT_DIR / "grid.png")
    save_nifti(warped, fixed_img, OUT_DIR / "warped.nii.gz")
    save_nifti(warped_pet, moving_pet_img, OUT_DIR / "warped_pet.nii.gz")


if __name__ == "__main__":
    main()
