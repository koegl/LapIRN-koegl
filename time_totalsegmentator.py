import argparse
import os
import time
from typing import Tuple

import nibabel as nib
import numpy as np
from totalsegmentator import python_api


def crop_axial(img: nib.Nifti1Image, n_crop: int) -> Tuple[nib.Nifti1Image, int, int]:
    data = img.get_fdata()
    z_start = n_crop
    z_end = data.shape[2] - n_crop

    cropped = data[:, :, z_start:z_end]

    affine = img.affine.copy()
    origin = affine @ np.array([0.0, 0.0, float(z_start), 1.0])
    affine[:3, 3] = origin[:3]

    out_img = nib.Nifti1Image(cropped, affine, img.header)
    return out_img, z_start, z_end


def restore_full(
    seg: np.ndarray, full_shape: Tuple[int, int, int], z_start: int, z_end: int
) -> np.ndarray:
    restored = np.zeros(full_shape, dtype=seg.dtype)
    restored[:, :, z_start:z_end] = seg
    return restored


def segment_one(img_path: str, tag: str, n_crop: int, out_dir: str) -> None:
    img = nib.load(img_path)
    full_shape = img.shape

    cropped_img, z_start, z_end = crop_axial(img, n_crop)
    z_size = z_end - z_start

    crop_path = os.path.join(out_dir, f"{tag}_img_crop{n_crop:03d}.nii.gz")
    seg_path = os.path.join(out_dir, f"{tag}_seg_crop{n_crop:03d}.nii.gz")
    full_path = os.path.join(out_dir, f"{tag}_seg_full_crop{n_crop:03d}.nii.gz")

    nib.save(cropped_img, crop_path)

    python_api.totalsegmentator(
        crop_path, seg_path, ml=True, task="total", fast=True, body_seg=True
    )

    seg_img = nib.load(seg_path)
    seg_data = np.asarray(seg_img.dataobj)
    restored = restore_full(seg_data, full_shape, z_start, z_end)
    nib.save(nib.Nifti1Image(restored, img.affine, seg_img.header), full_path)

    print(f"{tag}: z={z_size} restored -> {full_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop", type=int, required=True)
    parser.add_argument(
        "--fixed",
        type=str,
        default="/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/imagesVal/PSMARegPSMA_0001_0000_00.nii.gz",
    )
    parser.add_argument(
        "--moving",
        type=str,
        default="/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/imagesVal/PSMARegPSMA_0001_0000_01.nii.gz",
    )
    parser.add_argument("--out_dir", type=str, default="crop_sweep")
    args = parser.parse_args()

    t_start = time.perf_counter()

    os.makedirs(args.out_dir, exist_ok=True)

    t1 = time.perf_counter()
    segment_one(args.fixed, "fixed", args.crop, args.out_dir)
    t2 = time.perf_counter()
    segment_one(args.moving, "moving", args.crop, args.out_dir)

    print(f"first inferecne: {t2 - t1:.2f}s", flush=True)
    print(f"second inference: {time.perf_counter() - t2:.2f}s", flush=True)

    elapsed = time.perf_counter() - t_start
    z_size = 288 - 2 * args.crop

    time_path = os.path.join(args.out_dir, f"time_crop{args.crop:03d}.txt")
    with open(time_path, "w") as f:
        f.write(f"{args.crop}\t{z_size}\t{elapsed:.2f}\n")

    print(f"crop={args.crop} z={z_size} total={elapsed:.2f}s", flush=True)


if __name__ == "__main__":
    main()
