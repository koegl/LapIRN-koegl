"""Average per-label volume over the validation case pairs.

Every validation case is a pair of timepoints ("00" and "01"). For each label
the volume (in mm^3) is computed in both segmentations of a pair, averaged
within the pair and then averaged over all pairs. Labels that are absent in a
segmentation contribute a volume of 0, so the result is a mean over all pairs,
not only over the pairs where the label occurs.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import tqdm
from config import DATA_PATH

# --- variables (define here, no argparse) ---
n_labels: int = 118  # totalsegmentator "total" task, labels 0..117
timepoints: List[str] = ["00", "01"]

# official validation cases (same list as in inference.py)
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

# which validation set to use: "official" or "my_val"
eval_set: str = "official"

official_seg_dir = DATA_PATH / "PSMAReg/PSMAReg_dataset/segmentations_fast"
official_seg_template = "{case_id}_{tp}"

my_val_seg_dir = DATA_PATH / "PSMAReg/PSMAReg_dataset/labelsTr"
my_val_seg_template = "PSMARegPSMA_{case_id}_0000_{tp}"

split_path = DATA_PATH / "PSMAReg/PSMAReg_dataset/split.json"

out_csv = Path(
    "/home/iml/fryderyk.koegl/code/LapIRN-koegl/submission_results/csvs/label_volumes.csv"
)


def load_split(split_path: Path) -> Tuple[List[str], List[str]]:
    with open(split_path, "r") as f:
        split = json.load(f)
    train = split["train"]
    val = split["val"]
    return train, val


def resolve_seg_path(seg_dir: Path, seg_template: str, case_id: str, tp: str) -> Path:
    stem = seg_template.format(case_id=case_id, tp=tp)
    path = seg_dir / f"{stem}.nii.gz"
    if not path.exists():
        path = seg_dir / f"{stem}.nii"
    return path


def label_volumes_mm3(path: Path, n_labels: int) -> np.ndarray:
    """Volume in mm^3 of every label 0..n_labels-1 in one segmentation."""
    img = nib.load(path.as_posix())
    data = np.asarray(img.dataobj).astype(np.int16)
    counts = np.bincount(data.ravel(), minlength=n_labels)[:n_labels]
    voxel_vol = float(np.abs(np.linalg.det(img.affine[:3, :3])))
    volumes = counts.astype(np.float64) * voxel_vol
    return volumes


def average_label_volumes(
    subjects: List[str],
    seg_dir: Path,
    seg_template: str,
    n_labels: int,
) -> np.ndarray:
    """Mean over case pairs of the within-pair mean volume of each label."""
    pair_volumes: List[np.ndarray] = []
    for case_id in tqdm.tqdm(subjects, desc="cases"):
        tp_volumes: List[np.ndarray] = []
        for tp in timepoints:
            path = resolve_seg_path(seg_dir, seg_template, case_id, tp)
            if not path.exists():
                tqdm.tqdm.write(f"missing segmentation, skipping: {path}")
                continue
            tp_volumes.append(label_volumes_mm3(path, n_labels))

        if len(tp_volumes) != len(timepoints):
            tqdm.tqdm.write(f"incomplete pair, skipping case {case_id}")
            continue

        pair_volumes.append(np.mean(np.stack(tp_volumes), axis=0))

    if not pair_volumes:
        raise RuntimeError("no complete validation case pairs found")

    tqdm.tqdm.write(f"averaged over {len(pair_volumes)} case pairs")
    volumes = np.mean(np.stack(pair_volumes), axis=0)
    return volumes


def main() -> None:
    if eval_set == "official":
        subjects = val_subjects
        seg_dir = official_seg_dir
        seg_template = official_seg_template
    elif eval_set == "my_val":
        _, subjects = load_split(split_path)
        seg_dir = my_val_seg_dir
        seg_template = my_val_seg_template
    else:
        raise ValueError(f"unknown eval_set: {eval_set}")

    volumes = average_label_volumes(subjects, seg_dir, seg_template, n_labels)

    rows: Dict[str, object] = {
        "label": np.arange(1, n_labels, dtype=int),
        "volume_mm3": volumes[1:],
    }
    df = pd.DataFrame(rows)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
