from pathlib import Path
from typing import List

import nibabel as nib
import torch
import totalsegmentator.python_api as totalseg
import tqdm

val_image_dir = Path("/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/imagesVal")
out_dir = Path("/home/iml/fryderyk.koegl/data/PSMAReg/io_labels_ct")
val_subjects: List[str] = [
    "0001",
    "0003",
    "0005",
    "0007",
    "0008",
    "0009",
    "0013",
    "0021",
    "0024",
    "0029",
    "0031",
    "0033",
    "0034",
    "0035",
    "0036",
    "0038",
    "0039",
    "0042",
    "0047",
    "0048",
]
timepoints: List[str] = ["00", "01"]


def predict_one(ct_path: Path, out_path: Path, device_str: str) -> None:
    ct_img = nib.load(str(ct_path))
    seg_img = totalseg.totalsegmentator(
        ct_img, task="total", fast=False, device=device_str, quiet=True
    )
    nib.save(seg_img, str(out_path))


def main() -> None:
    device_str = "gpu" if torch.cuda.is_available() else "cpu"
    out_dir.mkdir(parents=True, exist_ok=True)

    for case_id in tqdm.tqdm(val_subjects, desc="CT labels"):
        for tp in timepoints:
            ct_path = val_image_dir / f"PSMARegPSMA_{case_id}_0000_{tp}.nii.gz"
            out_path = out_dir / f"ct_{case_id}_{tp}.nii.gz"
            if out_path.exists():
                continue
            predict_one(ct_path, out_path, device_str)
            tqdm.tqdm.write(f"{case_id}_{tp} done")


if __name__ == "__main__":
    main()
