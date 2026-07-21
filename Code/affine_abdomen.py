"""
Affine registration of the preprocessed Abdomen dataset.

Unlike the longitudinal datasets (which register timepoints of the same
patient), the abdomen dataset registers *inter-subject* pairs. This script
pre-computes and caches the ANTs affine DVF for exactly the pairs the
``AbdomenDataset`` produces (both the train and val splits) so that no
on-the-fly affine registration is needed at train time.

The pairs are obtained by building the datasets the same way ``utils`` does
and reading ``AbdomenDataset.pairs`` directly, so the ring + seeded-random
pairing (sized by ``cfg.use_abdomen``) matches training exactly.

Each pair is ``(case_id, tp_x, tp_y)`` with ``case_id = f"{a}_{b}"``; at
lookup time training splits it into ``case_id_x = a``, ``case_id_y = b`` with
moving = a @ tp_x and fixed = b @ tp_y (see ``affine_reg.get_affine_dvf``).
The DVF is cached under ``affine_{a}_{b}_{tp_x}_{tp_y}.npy``.

DVF convention: (H, W, D, 3), float32, voxel displacements at full resolution.
"""

import sys
from pathlib import Path
from typing import List, Tuple

import nibabel as nib
import numpy as np
import torch
import tqdm

# The Code/ modules import each other by top-level name, so put it on the path.
CODE_DIR = Path(__file__).resolve().parent / "Code"
sys.path.insert(0, str(CODE_DIR))

import affine_reg  # noqa: E402
import config  # noqa: E402
import my_data  # noqa: E402
from Functions import generate_grid_unit  # noqa: E402
from miccai2020_model_stage import SpatialTransform_unit  # noqa: E402

# Write the affine-warped moving volumes to disk in addition to caching the
# DVFs. The DVF cache is what training actually consumes.
SAVE_WARPED = True
OUT_DIR = Path(
    "/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/imagesTr_abdomen_affine"
)


def collect_abdomen_pairs(cfg: config.TrainingConfig) -> List[Tuple[str, str, str]]:
    """Return the (case_id, tp_x, tp_y) pairs exactly as AbdomenDataset builds
    them, for both the train and val abdomen splits (deduplicated)."""
    train_ids, val_ids = my_data.get_train_val_split(
        data_dir=cfg.data_dir,
        split_path=cfg.split_path,
        val_fraction=cfg.val_fraction,
        tubingen=False,
        nlst=False,
        abdomen=True,
    )

    pairs: List[Tuple[str, str, str]] = []
    seen = set()
    for split_ids in (train_ids, val_ids):
        dataset = my_data.AbdomenDataset(
            cfg=cfg,
            case_ids=split_ids,
            augment=False,
            use_cache=False,
        )
        for pair in dataset.pairs:
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
    return pairs


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
    # overfit truncates AbdomenDataset.pairs to a single element; we want the
    # full set of pairs the dataset would use across a real run.
    cfg.overfit = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_dir = cfg.data_dir / "imagesTr"
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    if SAVE_WARPED:
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    transform = SpatialTransform_unit().to(device)
    for param in transform.parameters():
        param.requires_grad = False

    grid_full = generate_grid_unit(cfg.img_shape)
    grid_full = (
        torch.from_numpy(np.reshape(grid_full, (1,) + grid_full.shape))
        .to(device)
        .float()
    )

    pairs = collect_abdomen_pairs(cfg)
    print(f"Found {len(pairs)} abdomen pairs to register.")

    for case_id, tp_x, tp_y in tqdm.tqdm(pairs):
        # case_id is "a_b"; split matches how the training loop looks it up
        case_id_x, case_id_y = case_id.split("_")

        # moving = case_id_x @ tp_x, fixed = case_id_y @ tp_y (see get_affine_dvf)
        moving_path = image_dir / f"PSMARegPSMA_{case_id_x}_0000_{tp_x}.nii.gz"
        fixed_path = image_dir / f"PSMARegPSMA_{case_id_y}_0000_{tp_y}.nii.gz"

        # get_affine_dvf handles the on-disk cache; this both computes (once)
        # and returns the DVF so we can optionally save the warped moving.
        dvf = affine_reg.get_affine_dvf(
            case_id_x=case_id_x,
            case_id_y=case_id_y,
            tp_x=tp_x,
            tp_y=tp_y,
            fixed_ct_path=fixed_path,
            moving_ct_path=moving_path,
            make_lowres_ants_image_fn=affine_reg.make_lowres_ants_image,
            preprocess_ct_fn=affine_reg.preprocess_ct,
            ants_affine_to_fullres_voxel_disp_fn=(
                affine_reg.ants_affine_to_fullres_voxel_disp
            ),
        )

        if not SAVE_WARPED:
            continue

        out_path = (
            OUT_DIR
            / f"PSMARegPSMA_{case_id_x}_{case_id_y}_0000_{tp_x}_{tp_y}_warped.nii.gz"
        )
        if out_path.exists():
            continue

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

    print("Done.")


if __name__ == "__main__":
    main()
