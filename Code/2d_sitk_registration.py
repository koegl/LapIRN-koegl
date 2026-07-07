from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from matplotlib.colors import TwoSlopeNorm
from tqdm import tqdm


def load_frames(path: Path, i: int) -> Tuple[sitk.Image, sitk.Image]:
    volume = sitk.ReadImage(str(path))
    # Cine assumed as 2D + time (3D volume, first axis = time).
    fixed = sitk.Cast(volume[i, :, :], sitk.sitkFloat32)
    moving = sitk.Cast(volume[i + 1, :, :], sitk.sitkFloat32)

    # normalise to 0-1 range
    fixed = sitk.RescaleIntensity(fixed, 0.0, 1.0)
    moving = sitk.RescaleIntensity(moving, 0.0, 1.0)
    return fixed, moving


def register_demons(
    fixed: sitk.Image,
    moving: sitk.Image,
    iterations: int,
    smoothing: float,
) -> sitk.Image:
    matched = sitk.HistogramMatching(
        moving,
        fixed,
        numberOfHistogramLevels=1024,
        numberOfMatchPoints=7,
        thresholdAtMeanIntensity=True,
    )
    demons = sitk.DiffeomorphicDemonsRegistrationFilter()
    demons.SetNumberOfIterations(iterations)
    demons.SetStandardDeviations(smoothing)  # Gaussian smoothing of the field

    pbar = tqdm(total=iterations, desc="Demons", leave=True)
    demons.AddCommand(sitk.sitkIterationEvent, lambda: pbar.update(1))
    field = demons.Execute(fixed, matched)
    pbar.close()
    tqdm.write(
        f"elapsed_iters={demons.GetElapsedIterations()} "
        f"final_metric={demons.GetMetric():.4f}"
    )
    return field


def warp_image(
    moving: sitk.Image, field: sitk.Image, reference: sitk.Image
) -> sitk.Image:
    transform = sitk.DisplacementFieldTransform(sitk.Image(field))
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(transform)
    warped = resampler.Execute(moving)
    return warped


def invert_field(field: sitk.Image) -> sitk.Image:
    inverse_field = sitk.InvertDisplacementField(
        field,
        maximumNumberOfIterations=20,
        maxErrorToleranceThreshold=0.01,
        meanErrorToleranceThreshold=5e-4,
        enforceBoundaryCondition=True,
    )
    return inverse_field


def generate_synthetic_data(
    fixed: sitk.Image, field: sitk.Image, inverse_field: sitk.Image
) -> Tuple[sitk.Image, sitk.Image]:
    # Apply the inverse deformation to the fixed to synthesise a moving.
    # `field` then maps fixed -> fake_moving exactly (ground-truth correspondence),
    # so warping fake_moving back with `field` recovers the fixed.
    fake_moving = warp_image(fixed, inverse_field, fixed)
    warped_fake = warp_image(fake_moving, field, fixed)
    return fake_moving, warped_fake


def orient_2d(array: np.ndarray) -> np.ndarray:
    # Acquisition geometry: rotate 180 deg, then flip about the vertical axis.
    oriented = np.fliplr(np.rot90(array, 2))
    return oriented


def orient_points(
    points: List[Tuple[float, float]], n_rows: int
) -> Tuple[List[float], List[float]]:
    # Match orient_2d (net vertical flip): column unchanged, row -> n_rows-1-row.
    xs = [p[0] for p in points]
    ys = [n_rows - 1 - p[1] for p in points]
    return xs, ys


def warp_points(
    points: List[Tuple[float, float]],
    inverse_field: sitk.Image,
    moving: sitk.Image,
    fixed: sitk.Image,
) -> List[Tuple[float, float]]:
    # Points are (x, y) in moving index space; map them into the fixed frame.
    # This needs the inverse, since `field` maps fixed -> moving.
    transform = sitk.DisplacementFieldTransform(sitk.Image(inverse_field))
    warped = []
    for x, y in points:
        phys = moving.TransformContinuousIndexToPhysicalPoint((float(x), float(y)))
        moved = transform.TransformPoint(phys)
        idx = fixed.TransformPhysicalPointToContinuousIndex(moved)
        warped.append((idx[0], idx[1]))
    return warped


def show_results(
    fixed: sitk.Image,
    moving: sitk.Image,
    warped: sitk.Image,
    field: sitk.Image,
    jac: sitk.Image,
    grid_step: int,
    title: str,
    inverse_field: Optional[sitk.Image] = None,
    points: Optional[List[Tuple[float, float]]] = None,
) -> None:
    f = orient_2d(sitk.GetArrayFromImage(fixed))
    m = orient_2d(sitk.GetArrayFromImage(moving))
    w = orient_2d(sitk.GetArrayFromImage(warped))
    j = orient_2d(sitk.GetArrayFromImage(jac))
    disp = sitk.GetArrayFromImage(field)
    diff = f - w

    spacing = fixed.GetSpacing()
    ny, nx = f.shape
    grid_x, grid_y = np.meshgrid(np.arange(nx), np.arange(ny))
    # Orient the field too; the vertical mirror flips the y-component's sign.
    dx = orient_2d(disp[:, :, 0]) / spacing[0]
    dy = -orient_2d(disp[:, :, 1]) / spacing[1]
    def_x = grid_x + dx
    def_y = grid_y + dy

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)

    axes[0, 0].imshow(f, cmap="gray")
    axes[0, 0].set_title("Fixed (frame 0)")

    axes[0, 1].imshow(m, cmap="gray")
    axes[0, 1].set_title("Moving (frame 1)")

    axes[0, 2].imshow(w, cmap="gray")
    axes[0, 2].set_title("Warped moving")

    if points is not None and inverse_field is not None:
        warped_points = warp_points(points, inverse_field, moving, fixed)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(points)))
        px, py = orient_points(points, ny)
        wx, wy = orient_points(warped_points, ny)
        # axes[0, 1].scatter(px, py, c=colors, s=25, edgecolors="k", linewidths=0.5)
        axes[0, 1].scatter(px, py, c="red", s=30)
        axes[0, 2].scatter(wx, wy, c="red", s=30)
        axes[0, 0].scatter(wx, wy, c="red", s=30)

    dmax = np.abs(diff).max()
    im_d = axes[1, 0].imshow(diff, cmap="gray", vmin=-dmax, vmax=dmax)
    axes[1, 0].set_title("Diff (fixed - warped)")
    fig.colorbar(im_d, ax=axes[1, 0], fraction=0.046, pad=0.04)

    ax_g = axes[1, 1]
    for i in range(0, ny, grid_step):
        ax_g.plot(def_x[i, :], def_y[i, :], color="tab:blue", lw=0.5)
    for jx in range(0, nx, grid_step):
        ax_g.plot(def_x[:, jx], def_y[:, jx], color="tab:blue", lw=0.5)
    ax_g.set_aspect("equal")
    ax_g.set_xlim(0, nx)
    ax_g.set_ylim(0, ny)
    ax_g.invert_yaxis()
    ax_g.set_title("Displacement field (warped grid)")

    jmin = min(j.min(), 1.0 - 1e-3)
    jmax = max(j.max(), 1.0 + 1e-3)
    norm = TwoSlopeNorm(vmin=jmin, vcenter=1.0, vmax=jmax)
    im_j = axes[1, 2].imshow(j, cmap="RdBu", norm=norm)
    axes[1, 2].set_title("Jacobian determinant")
    fig.colorbar(im_j, ax=axes[1, 2], fraction=0.046, pad=0.04)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()

    save_path = f"demons_{title.replace(' ', '_')}.png"
    fig.savefig(save_path, dpi=300)
    plt.show()


def main() -> None:
    mha_path = Path("/home/iml/fryderyk.koegl/data/daniel_cine_mri/A_143_frames.mha")
    iterations = 200
    smoothing = 2.0  # std of Gaussian field regularization (voxels) (higher = smoother)
    grid_step = 8  # step for plotting the warped grid (pixels)

    fixed, moving = load_frames(mha_path, i=54)
    field = register_demons(fixed, moving, iterations, smoothing)
    inverse_field = invert_field(field)
    jac = sitk.DisplacementFieldJacobianDeterminant(field)
    warped = warp_image(moving, field, fixed)

    # show_results(
    #     fixed,
    #     moving,
    #     warped,
    #     field,
    #     jac,
    #     grid_step,
    #     "real dataset",
    #     inverse_field,
    #     points,
    # )

    points = [(235.0, 9.0)]
    fake_moving, warped_fake = generate_synthetic_data(fixed, field, inverse_field)
    show_results(
        fixed,
        fake_moving,
        warped_fake,
        field,
        jac,
        grid_step,
        "synthetic dataset",
        inverse_field,
        points,
    )
    x = 0


if __name__ == "__main__":
    main()
