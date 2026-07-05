from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from scipy import ndimage

BONE_LABEL_VALUES: Sequence[int] = (
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    69,
    70,
    71,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    91,
    92,
    93,
    94,
    95,
    96,
    97,
    98,
    99,
    100,
    101,
    102,
    103,
    104,
    105,
    106,
    107,
    108,
    109,
    110,
    111,
    112,
    113,
    114,
    115,
    116,
)


def save_frozen_pair(frozen: Dict[str, torch.Tensor], path: Path) -> None:
    cpu_frozen = {key: value.detach().cpu() for key, value in frozen.items()}
    torch.save(cpu_frozen, path)


def load_frozen_pair(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    cpu_frozen = torch.load(path, map_location="cpu")
    frozen = {key: value.to(device) for key, value in cpu_frozen.items()}
    return frozen


def build_identity_grid(
    shape: Tuple[int, int, int], device: torch.device
) -> torch.Tensor:
    """Voxel-coordinate identity grid (1, 3, D0, D1, D2), component order (d0, d1, d2)."""
    d0, d1, d2 = shape
    g0, g1, g2 = torch.meshgrid(
        torch.arange(d0, device=device),
        torch.arange(d1, device=device),
        torch.arange(d2, device=device),
        indexing="ij",
    )
    grid = torch.stack([g0, g1, g2], dim=0).unsqueeze(0).float()
    return grid


def voxel_coords_to_sample_grid(
    coords_vox: torch.Tensor, shape: Tuple[int, int, int]
) -> torch.Tensor:
    """Voxel coords (1, 3, D0, D1, D2) -> grid_sample grid (1, D0, D1, D2, 3),
    normalized [-1, 1], align_corners=True, last channel ordered (d2, d1, d0)."""
    d0, d1, d2 = shape
    c0 = coords_vox[:, 0]
    c1 = coords_vox[:, 1]
    c2 = coords_vox[:, 2]
    n0 = 2.0 * c0 / (d0 - 1) - 1.0
    n1 = 2.0 * c1 / (d1 - 1) - 1.0
    n2 = 2.0 * c2 / (d2 - 1) - 1.0
    grid = torch.stack([n2, n1, n0], dim=-1)
    return grid


def warp(
    volume: torch.Tensor,
    disp: torch.Tensor,
    identity_vox: torch.Tensor,
    mode: str = "bilinear",
) -> torch.Tensor:
    shape = (volume.shape[2], volume.shape[3], volume.shape[4])
    coords = identity_vox + disp
    grid = voxel_coords_to_sample_grid(coords, shape)
    warped = F.grid_sample(
        volume, grid, mode=mode, padding_mode="border", align_corners=True
    )
    return warped


def integrate_svf(
    velocity: torch.Tensor, identity_vox: torch.Tensor, num_steps: int
) -> torch.Tensor:
    """Scaling-and-squaring integration of a stationary velocity field to a displacement."""
    disp = velocity / (2**num_steps)
    for _ in range(num_steps):
        warped = warp(disp, disp, identity_vox)
        disp = disp + warped
    return disp


def gaussian_blur_3d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable 3D Gaussian blur on (1, C, D0, D1, D2)."""
    if sigma <= 0.0:
        return x
    radius = int(3.0 * sigma)
    coords = torch.arange(-radius, radius + 1, device=x.device).float()
    kernel_1d = torch.exp(-(coords**2) / (2.0 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()

    n_channels = x.shape[1]
    out = x
    for dim in range(2, 5):
        view_shape = [1, 1, 1, 1, 1]
        view_shape[dim] = kernel_1d.numel()
        kernel = kernel_1d.view(view_shape).repeat(n_channels, 1, 1, 1, 1)
        pad = [0, 0, 0, 0, 0, 0]
        pad_index = (4 - dim) * 2
        pad[pad_index] = radius
        pad[pad_index + 1] = radius
        padded = F.pad(out, pad, mode="replicate")
        out = F.conv3d(padded, kernel, groups=n_channels)
    return out


def get_bone_components(
    ct_hu: np.ndarray,
    label_path: Optional[Path],
    bone_label_values: Sequence[int],
    hu_fallback_threshold: float,
    min_component_voxels: int,
) -> List[np.ndarray]:
    """Return a list of boolean masks, one per rigid bone body.

    Prefers the organizer label map (each listed bone label = one body). Falls
    back to an HU threshold + connected components when the label file is
    absent or contains none of the requested bone labels.
    """
    bodies: List[np.ndarray] = []

    if label_path is not None and label_path.exists():
        label = nib.load(label_path.as_posix()).get_fdata().astype(np.int32)
        for value in bone_label_values:
            body = label == value
            if int(body.sum()) >= min_component_voxels:
                bodies.append(body)

    if len(bodies) == 0:
        # Fallback: threshold HU, split into connected components.
        bone = ct_hu > hu_fallback_threshold
        bone = ndimage.binary_opening(bone, structure=np.ones((3, 3, 3), dtype=bool))
        labels, n = ndimage.label(bone)
        for idx in range(1, n + 1):
            body = labels == idx
            if int(body.sum()) >= min_component_voxels:
                bodies.append(body)

    return bodies


def build_rigid_velocity(
    bodies: List[np.ndarray],
    identity_vox: torch.Tensor,
    device: torch.device,
    bone_translation: float,
    bone_rotation_deg: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a velocity field that is a small rigid motion inside each bone body.

    For each body: disp(x) = t + w x (x - centroid), with random small
    translation t and rotation vector w. Returns (v_rigid, bone_weight), where
    bone_weight is 1 inside any body, 0 elsewhere.
    """
    v_rigid = torch.zeros_like(identity_vox)
    bone_weight = torch.zeros(
        (1, 1, *identity_vox.shape[2:]), device=device, dtype=torch.float32
    )

    for body in bodies:
        idx = np.argwhere(body)  # (N, 3) in (d0, d1, d2)
        if idx.shape[0] == 0:
            continue
        centroid = idx.mean(axis=0)
        coords = torch.from_numpy(idx).to(device).float()  # (N, 3)
        rel = coords - torch.from_numpy(centroid).to(device).float()

        t = torch.randn(3, device=device)
        t = t / (t.norm() + 1e-8) * (bone_translation * torch.rand(1, device=device))

        axis = torch.randn(3, device=device)
        axis = axis / (axis.norm() + 1e-8)
        angle = np.deg2rad(bone_rotation_deg) * float(torch.rand(1).item())
        w = axis * angle

        disp = t[None, :] + torch.cross(w[None, :].expand_as(rel), rel, dim=1)  # (N, 3)

        i0 = torch.from_numpy(idx[:, 0]).to(device).long()
        i1 = torch.from_numpy(idx[:, 1]).to(device).long()
        i2 = torch.from_numpy(idx[:, 2]).to(device).long()
        for c in range(3):
            v_rigid[0, c, i0, i1, i2] = disp[:, c]
        bone_weight[0, 0, i0, i1, i2] = 1.0

    return v_rigid, bone_weight


def generate_synthetic_pair(
    ct_path: Path,
    out_dir: Path,
    label_path: Optional[Path] = None,
    bone_label_values: Sequence[int] = (),
    max_displacement: float = 15.0,
    coarse_downsample: int = 32,
    smoothing_sigma: float = 5.0,
    num_integration_steps: int = 7,
    move_bones: bool = True,
    bone_translation: float = 4.0,
    bone_rotation_deg: float = 3.0,
    bone_transition_sigma: float = 6.0,
    hu_fallback_threshold: float = 200.0,
    min_component_voxels: int = 200,
    device: torch.device = torch.device("cuda"),
    seed: Optional[int] = None,
) -> None:
    """Generate one synthetic moving image from a fixed CT and save for inspection.

    Bones (from organizer labels, else HU fallback) move as small independent
    rigid bodies; soft tissue deforms smoothly; both are composed in velocity
    space and integrated so the result stays diffeomorphic.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    out_dir.mkdir(parents=True, exist_ok=True)

    ct_nib = nib.load(ct_path.as_posix())
    ct_hu = ct_nib.get_fdata().astype(np.float32)
    affine = ct_nib.affine
    shape = ct_hu.shape

    fixed = torch.from_numpy(ct_hu).to(device)[None, None].float()
    identity_vox = build_identity_grid(shape, device)

    # --- smooth soft-tissue velocity -------------------------------------------
    coarse_shape = tuple(max(2, s // coarse_downsample) for s in shape)
    v_coarse = torch.randn(1, 3, *coarse_shape, device=device)
    v_soft = F.interpolate(v_coarse, size=shape, mode="trilinear", align_corners=True)
    v_soft = gaussian_blur_3d(v_soft, smoothing_sigma)

    # --- bones: rigid bodies, feathered mask -----------------------------------
    if move_bones:
        bodies = get_bone_components(
            ct_hu,
            label_path,
            bone_label_values,
            hu_fallback_threshold,
            min_component_voxels,
        )
        v_rigid, bone_mask = build_rigid_velocity(
            bodies, identity_vox, device, bone_translation, bone_rotation_deg
        )

        # normalized smoothing: spreads rigid motion smoothly, keeps magnitude
        v_rigid_s = gaussian_blur_3d(v_rigid * bone_mask, bone_transition_sigma)
        w = gaussian_blur_3d(bone_mask, bone_transition_sigma)
        v_rigid_s = v_rigid_s / (w + 1e-6)
        bone_weight = w.clamp(0.0, 1.0)
        soft_weight = 1.0 - bone_weight

        norm = torch.sqrt((v_soft**2).sum(dim=1, keepdim=True))
        max_norm = norm.max()
        if max_norm > 0:
            v_soft = v_soft * (max_displacement / max_norm)

        velocity = soft_weight * v_soft + bone_weight * v_rigid_s
    else:
        norm = torch.sqrt((v_soft**2).sum(dim=1, keepdim=True))
        max_norm = norm.max()
        if max_norm > 0:
            v_soft = v_soft * (max_displacement / max_norm)
        velocity = v_soft
        bone_weight = torch.zeros((1, 1, *shape), device=device, dtype=torch.float32)

    # --- integrate and warp ----------------------------------------------------
    disp = integrate_svf(velocity, identity_vox, num_integration_steps)
    moving = warp(fixed, disp, identity_vox)
    disp_mag = torch.sqrt((disp**2).sum(dim=1, keepdim=True))

    disp_inv = integrate_svf(-velocity, identity_vox, num_integration_steps)
    fixed_recon = warp(moving, disp_inv, identity_vox)
    recon_err = (fixed_recon - fixed).abs()
    print(
        f"[inv-check] {ct_path.name}: recon L1 "
        f"mean={recon_err.mean().item():.4f} max={recon_err.max().item():.4f}"
    )

    stem = ct_path.name.removesuffix(".nii.gz")
    nib.save(
        nib.Nifti1Image(fixed_recon.squeeze().cpu().numpy(), affine),
        (out_dir / f"{stem}_fixedrecon.nii.gz").as_posix(),
    )
    nib.save(
        nib.Nifti1Image(fixed.squeeze().cpu().numpy(), affine),
        (out_dir / f"{stem}_fixed.nii.gz").as_posix(),
    )
    nib.save(
        nib.Nifti1Image(moving.squeeze().cpu().numpy(), affine),
        (out_dir / f"{stem}_moving.nii.gz").as_posix(),
    )
    nib.save(
        nib.Nifti1Image(disp_mag.squeeze().cpu().numpy(), affine),
        (out_dir / f"{stem}_dispmag.nii.gz").as_posix(),
    )
    nib.save(
        nib.Nifti1Image(bone_weight.squeeze().cpu().numpy(), affine),
        (out_dir / f"{stem}_boneweight.nii.gz").as_posix(),
    )

    return fixed, moving, disp_inv, shape


def bodies_from_label(
    label_ct: torch.Tensor, bone_label_values: Sequence[int], min_voxels: int
) -> List[np.ndarray]:
    """Boolean bone-body masks from an organizer CT label tensor (1,1,D0,D1,D2)."""
    lbl = label_ct.squeeze().cpu().numpy().astype(np.int32)
    bodies: List[np.ndarray] = []
    for value in bone_label_values:
        body = lbl == value
        if int(body.sum()) >= min_voxels:
            bodies.append(body)
    return bodies


def decorrelate_intensity(
    img: torch.Tensor, bias_strength: float, noise_std: float, device: torch.device
) -> torch.Tensor:
    """Break the perfect intensity match: smooth multiplicative bias + noise, clip [0,1]."""
    shape = (img.shape[2], img.shape[3], img.shape[4])
    coarse = torch.randn(1, 1, 4, 4, 4, device=device)
    bias = F.interpolate(coarse, size=shape, mode="trilinear", align_corners=True)
    bias = 1.0 + bias_strength * bias
    out = img * bias + noise_std * torch.randn_like(img)
    return out.clamp(0.0, 1.0)


def unit_flow_from_voxel_disp(
    disp_vox: torch.Tensor, shape: Tuple[int, int, int]
) -> torch.Tensor:
    """Voxel disp (1, 3, D0, D1, D2), order (d0, d1, d2) -> unit-flow."""
    d0, d1, d2 = shape
    unit = torch.empty_like(disp_vox)
    unit[:, 0] = disp_vox[:, 0] * 2.0 / (d0 - 1)
    unit[:, 1] = disp_vox[:, 1] * 2.0 / (d1 - 1)
    unit[:, 2] = disp_vox[:, 2] * 2.0 / (d2 - 1)
    return unit


def generate_synthetic_moving(
    source: torch.Tensor,
    source_label_ct: torch.Tensor,
    source_label_pet: torch.Tensor,
    bone_label_values: Sequence[int],
    max_displacement: float = 15.0,
    coarse_downsample: int = 32,
    smoothing_sigma: float = 5.0,
    num_integration_steps: int = 7,
    move_bones: bool = True,
    bone_translation: float = 4.0,
    bone_rotation_deg: float = 3.0,
    bone_transition_sigma: float = 6.0,
    min_component_voxels: int = 200,
    bias_strength: float = 0.08,
    noise_std: float = 0.005,
    device: torch.device = torch.device("cuda"),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a synthetic moving pair from one source (the fixed image).

    Returns:
        moving        (1, 2, D0,D1,D2) warped + intensity-decorrelated
        moving_lbl_ct (1, 1, ...)
        moving_lbl_pet(1, 1, ...)
        gt_unit       (1, 3, ...) unit-flow DVF for X->Y, matching F_X_Y
    """
    shape = (source.shape[2], source.shape[3], source.shape[4])
    identity_vox = build_identity_grid(shape, device)

    coarse_shape = tuple(max(2, s // coarse_downsample) for s in shape)
    v_coarse = torch.randn(1, 3, *coarse_shape, device=device)
    v_soft = F.interpolate(v_coarse, size=shape, mode="trilinear", align_corners=True)
    v_soft = gaussian_blur_3d(v_soft, smoothing_sigma)

    if move_bones:
        bodies = bodies_from_label(
            source_label_ct, bone_label_values, min_component_voxels
        )
        v_rigid, bone_mask = build_rigid_velocity(
            bodies, identity_vox, device, bone_translation, bone_rotation_deg
        )
        v_rigid_s = gaussian_blur_3d(v_rigid * bone_mask, bone_transition_sigma)
        w = gaussian_blur_3d(bone_mask, bone_transition_sigma)
        v_rigid_s = v_rigid_s / (w + 1e-6)
        bone_weight = w.clamp(0.0, 1.0)
        soft_weight = 1.0 - bone_weight
        norm = torch.sqrt((v_soft**2).sum(dim=1, keepdim=True))
        max_norm = norm.max()
        if max_norm > 0:
            v_soft = v_soft * (max_displacement / max_norm)
        velocity = soft_weight * v_soft + bone_weight * v_rigid_s
    else:
        norm = torch.sqrt((v_soft**2).sum(dim=1, keepdim=True))
        max_norm = norm.max()
        if max_norm > 0:
            v_soft = v_soft * (max_displacement / max_norm)
        velocity = v_soft

    disp = integrate_svf(velocity, identity_vox, num_integration_steps)
    disp_inv = integrate_svf(-velocity, identity_vox, num_integration_steps)

    moving = warp(source, disp, identity_vox, mode="bilinear")
    moving = decorrelate_intensity(moving, bias_strength, noise_std, device)
    moving_lbl_ct = warp(source_label_ct.float(), disp, identity_vox, mode="nearest")
    moving_lbl_pet = warp(source_label_pet.float(), disp, identity_vox, mode="nearest")

    gt_unit = unit_flow_from_voxel_disp(disp_inv, shape).flip(1)

    result = (moving, moving_lbl_ct, moving_lbl_pet, gt_unit)
    return result


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    out_dir = Path("tmp/synthetic_check")

    # (ct_path, label_path) — label_path may be None to force the HU fallback
    cases: List[Tuple[Path, Optional[Path]]] = [
        (
            Path(
                "/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/imagesTr/PSMARegPSMA_0002_0000_00.nii.gz"
            ),
            Path(
                "/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/labelsTr/PSMARegPSMA_0002_0000_00.nii.gz"
            ),
        ),
    ]

    # TODO: fill with the integer bone labels in the organizer CT label map
    bone_label_values: Sequence[int] = (
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        69,
        70,
        71,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        91,
        92,
        93,
        94,
        95,
        96,
        97,
        98,
        99,
        100,
        101,
        102,
        103,
        104,
        105,
        106,
        107,
        108,
        109,
        110,
        111,
        112,
        113,
        114,
        115,
        116,
    )

    for ct_path, label_path in tqdm.tqdm(cases, desc="synthetic pairs"):
        generate_synthetic_pair(
            ct_path=ct_path,
            out_dir=out_dir,
            label_path=label_path,
            bone_label_values=bone_label_values,
            max_displacement=15.0,
            move_bones=True,
            device=device,
            seed=0,
        )


if __name__ == "__main__":
    main()
