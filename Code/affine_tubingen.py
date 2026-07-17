"""
Affine registration of the preprocessed Tubingen dataset.

For every fixed/moving pair in imagesTr_tubingen (``*_00`` = fixed,
``*_01`` = moving) this computes the ANTs affine DVF using the existing
affine registration code, stores it in the affine cache, and writes the
affine-warped moving image to imagesTr_tubingen_affine.
"""

import re
import sys
from pathlib import Path
from typing import Dict, Tuple

import nibabel as nib
import numpy as np
import torch
import tqdm

# The Code/ modules import each other by top-level name, so put it on the path.
CODE_DIR = Path(__file__).resolve().parent / "Code"
sys.path.insert(0, str(CODE_DIR))

import affine_reg  # noqa: E402
import config  # noqa: E402
from Functions import generate_grid_unit  # noqa: E402
from miccai2020_model_stage import SpatialTransform_unit  # noqa: E402

IMAGES_DIR = Path(
    "/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/imagesTr_tubingen"
)
OUT_DIR = Path(
    "/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/imagesTr_tubingen_affine"
)

FIXED_TP = "00"
MOVING_TP = "01"


def collect_case_ids(images_dir: Path) -> list[str]:
    """Return sorted case ids that have both a fixed (_00) and moving (_01)."""
    pattern = re.compile(r"^PSMARegPSMA_(?P<cid>\d+)_0000_(?P<tp>\d+)\.nii\.gz$")
    tps: Dict[str, set] = {}
    for path in sorted(images_dir.glob("PSMARegPSMA_*_0000_*.nii.gz")):
        match = pattern.match(path.name)
        if match is None:
            continue
        tps.setdefault(match.group("cid"), set()).add(match.group("tp"))

    return sorted(
        cid for cid, seen in tps.items() if FIXED_TP in seen and MOVING_TP in seen
    )


def dvf_to_flow(
    dvf: np.ndarray,
    img_shape: Tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor:
    """Convert a voxel-displacement DVF (H, W, D, 3) to a grid_sample flow.

    Mirrors ``affine_reg.create_affine_flow`` without any augmentation.
    """
    dvf_tensor = affine_reg.dvf_to_tensor(dvf, device)  # (1, 3, H, W, D)

    H, W, D = img_shape
    d_h = dvf_tensor[:, 0] / (H / 2.0)
    d_w = dvf_tensor[:, 1] / (W / 2.0)
    d_d = dvf_tensor[:, 2] / (D / 2.0)
    flow_affine = torch.stack([d_d, d_w, d_h], dim=1)  # (1, 3, H, W, D)

    return flow_affine.permute(0, 2, 3, 4, 1)  # (1, H, W, D, 3)


def main() -> None:
    cfg = config.TrainingConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)

    transform = SpatialTransform_unit().to(device)
    for param in transform.parameters():
        param.requires_grad = False

    grid_full = generate_grid_unit(cfg.img_shape)
    grid_full = (
        torch.from_numpy(np.reshape(grid_full, (1,) + grid_full.shape))
        .to(device)
        .float()
    )

    case_ids = collect_case_ids(IMAGES_DIR)
    print(f"Found {len(case_ids)} pairs to register.")

    for idx, case_id in enumerate(tqdm.tqdm(case_ids)):
        fixed_path = IMAGES_DIR / f"PSMARegPSMA_{case_id}_0000_{FIXED_TP}.nii.gz"
        moving_path = IMAGES_DIR / f"PSMARegPSMA_{case_id}_0000_{MOVING_TP}.nii.gz"
        out_path = OUT_DIR / f"PSMARegPSMA_{case_id}_0000_{MOVING_TP}_warped.nii.gz"

        if out_path.exists():
            continue

        # tp_x = moving, tp_y = fixed (matches affine_reg conventions)
        dvf = affine_reg.get_affine_dvf(
            case_id=case_id,
            tp_x=MOVING_TP,
            tp_y=FIXED_TP,
            fixed_ct_path=fixed_path,
            moving_ct_path=moving_path,
            make_lowres_ants_image_fn=affine_reg.make_lowres_ants_image,
            preprocess_ct_fn=affine_reg.preprocess_ct,
            ants_affine_to_fullres_voxel_disp_fn=(
                affine_reg.ants_affine_to_fullres_voxel_disp
            ),
        )

        flow = dvf_to_flow(dvf, cfg.img_shape, device)

        moving_np = nib.load(str(moving_path)).get_fdata().astype(np.float32)
        moving_tensor = (
            torch.from_numpy(moving_np).unsqueeze(0).unsqueeze(0).to(device).float()
        )

        with torch.no_grad():
            warped = transform(moving_tensor, flow, grid_full)

        # warped moving lives in the fixed image's grid -> use the fixed affine
        affine = nib.load(str(fixed_path)).affine
        warped_np = warped.squeeze().cpu().numpy().astype(np.float32)
        nib.save(nib.Nifti1Image(warped_np, affine), str(out_path))

        x = 0

    print("Done.")


if __name__ == "__main__":
    main()
