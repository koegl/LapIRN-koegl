"""Analyse where PET tumour labels sit with respect to the CT TotalSegmentator labels.

Two analyses are produced:

1. Per image: overlap (DSC) of the PET tumour mask with the union of all bone
   labels (``synthetic.BONE_LABEL_VALUES``).
2. Over all images: overlap of the PET tumour mask with every single
   TotalSegmentator CT label, averaged over the images, reporting how much of
   each CT structure is covered by tumour.

Only images whose name starts with ``--prefix`` and that have both a CT label
(``_0000_``) and a PET label (``_0001_``) in ``labelsTr`` are used.
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1] / "Code"))

import synthetic  # noqa: E402

IMAGES_DIR = Path("/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/imagesTr")
LABELS_DIR = Path("/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/labelsTr")
LABEL_NAMES_CSV = Path(
    "/home/iml/fryderyk.koegl/code/LapIRN-koegl/total_segmentator_labels.csv"
)
OUTPUT_DIR = Path("/home/iml/fryderyk.koegl/code/LapIRN-koegl/tumour_analysis")
PREFIX = "PSMARegPSMA_0"

CT_MODALITY = "0000"
PET_MODALITY = "0001"


def load_label_names(path: Path) -> Dict[int, str]:
    """Read the id,name TotalSegmentator table."""
    names: Dict[int, str] = {}
    with open(path, newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 2 or not row[0].strip().isdigit():
                continue
            names[int(row[0])] = row[1].strip()
    return names


def find_cases(prefix: str) -> List[Tuple[str, Path, Path]]:
    """Return (case_id, ct_label_path, pet_label_path) for all usable images."""
    cases = []
    for image in sorted(IMAGES_DIR.glob(f"{prefix}*_{CT_MODALITY}_*.nii.gz")):
        ct_label = LABELS_DIR / image.name
        pet_label = LABELS_DIR / image.name.replace(
            f"_{CT_MODALITY}_", f"_{PET_MODALITY}_"
        )
        pet_image = IMAGES_DIR / pet_label.name
        if not (ct_label.is_file() and pet_label.is_file() and pet_image.is_file()):
            continue
        cases.append((image.name.replace(".nii.gz", ""), ct_label, pet_label))
    return cases


def dice(intersection: int, size_a: int, size_b: int) -> float:
    """Dice similarity coefficient, nan if both masks are empty."""
    denominator = size_a + size_b
    if denominator == 0:
        return float("nan")
    return 2.0 * intersection / denominator


def drop_nan_rows(rows: List[Dict], description: str) -> List[Dict]:
    """Remove rows that contain a nan in any column."""
    kept = [
        row
        for row in rows
        if not any(
            isinstance(value, float) and np.isnan(value) for value in row.values()
        )
    ]
    if len(kept) < len(rows):
        print(f"dropped {len(rows) - len(kept)} {description} rows containing nan")
    if not kept:
        raise SystemExit(f"all {description} rows contained nan - nothing to write")
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default=PREFIX, help="image filename prefix")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    label_names = load_label_names(LABEL_NAMES_CSV)
    bone_values = np.asarray(sorted(synthetic.BONE_LABEL_VALUES), dtype=np.int64)

    cases = find_cases(args.prefix)
    if not cases:
        raise SystemExit(f"no images found for prefix '{args.prefix}'")
    print(f"found {len(cases)} images with both CT and PET labels")

    bone_rows = []
    # per CT label accumulators, keyed by label id
    per_label_dice: Dict[int, List[float]] = {i: [] for i in label_names}
    per_label_covered: Dict[int, List[float]] = {i: [] for i in label_names}
    per_label_of_tumour: Dict[int, List[float]] = {i: [] for i in label_names}
    total_intersection: Dict[int, int] = {i: 0 for i in label_names}
    total_label_voxels: Dict[int, int] = {i: 0 for i in label_names}
    n_images_with_label: Dict[int, int] = {i: 0 for i in label_names}
    total_tumour_voxels = 0
    # long format (one row per image and CT label), needed for the by-session analysis
    per_image_label_rows = []

    pbar = tqdm.tqdm(cases, desc="analysing tumour locations", unit="image", ncols=100)
    for case_id, ct_path, pet_path in pbar:
        ct = nib.load(str(ct_path))
        ct_label = np.asanyarray(ct.dataobj).astype(np.int64)
        tumour = np.asanyarray(nib.load(str(pet_path)).dataobj) > 0

        if ct_label.shape != tumour.shape:
            raise ValueError(
                f"{case_id}: CT label shape {ct_label.shape} != PET label shape "
                f"{tumour.shape}"
            )

        voxel_volume_ml = float(np.prod(ct.header.get_zooms()[:3])) / 1000.0
        n_tumour = int(tumour.sum())
        total_tumour_voxels += n_tumour

        # --- task 1: tumour vs. union of all bone labels ---
        bone = np.isin(ct_label, bone_values)
        n_bone = int(bone.sum())
        n_bone_tumour = int((bone & tumour).sum())
        bone_rows.append(
            {
                "case_id": case_id,
                "tumour_voxels": n_tumour,
                "tumour_volume_ml": round(n_tumour * voxel_volume_ml, 4),
                "bone_voxels": n_bone,
                "bone_volume_ml": round(n_bone * voxel_volume_ml, 4),
                "intersection_voxels": n_bone_tumour,
                "intersection_volume_ml": round(n_bone_tumour * voxel_volume_ml, 4),
                "dice_tumour_bone": round(dice(n_bone_tumour, n_tumour, n_bone), 6),
                "frac_tumour_in_bone": (
                    round(n_bone_tumour / n_tumour, 6) if n_tumour else float("nan")
                ),
                "frac_bone_covered_by_tumour": (
                    round(n_bone_tumour / n_bone, 8) if n_bone else float("nan")
                ),
            }
        )

        # --- task 2: tumour vs. each individual CT label ---
        # counts of every CT label, and of the CT labels under the tumour mask
        label_counts = np.bincount(ct_label.ravel(), minlength=max(label_names) + 1)
        tumour_counts = np.bincount(
            ct_label[tumour].ravel(), minlength=max(label_names) + 1
        )
        for label_id in label_names:
            n_label = int(label_counts[label_id])
            n_inter = int(tumour_counts[label_id])
            total_intersection[label_id] += n_inter
            total_label_voxels[label_id] += n_label
            if n_label == 0:
                continue
            n_images_with_label[label_id] += 1
            per_label_dice[label_id].append(dice(n_inter, n_tumour, n_label))
            per_label_covered[label_id].append(n_inter / n_label)
            if n_tumour:
                per_label_of_tumour[label_id].append(n_inter / n_tumour)
            per_image_label_rows.append(
                {
                    "case_id": case_id,
                    "label_id": label_id,
                    "label_name": label_names[label_id],
                    "tumour_voxels": n_tumour,
                    "label_voxels": n_label,
                    "intersection_voxels": n_inter,
                    "dice_tumour_label": round(dice(n_inter, n_tumour, n_label), 6),
                    "frac_label_covered_by_tumour": round(n_inter / n_label, 8),
                    "frac_tumour_in_label": (
                        round(n_inter / n_tumour, 8) if n_tumour else float("nan")
                    ),
                    "is_bone_label": int(label_id in set(bone_values.tolist())),
                }
            )

        tqdm.tqdm.write(
            f"{case_id}: tumour={n_tumour} vox, in bone={n_bone_tumour} vox, "
            f"dice(tumour,bone)={bone_rows[-1]['dice_tumour_bone']}"
        )

    bone_csv = args.output_dir / "tumour_bone_overlap_per_image.csv"
    with open(bone_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(bone_rows[0]))
        writer.writeheader()
        writer.writerows(bone_rows)
    print(f"wrote {bone_csv}")

    def mean(values: List[float]) -> float:
        return float(np.nanmean(values)) if values else float("nan")

    label_rows = []
    for label_id, name in sorted(label_names.items()):
        label_rows.append(
            {
                "label_id": label_id,
                "label_name": name,
                "n_images_with_label": n_images_with_label[label_id],
                "mean_dice_tumour_label": round(mean(per_label_dice[label_id]), 6),
                "mean_frac_label_covered_by_tumour": round(
                    mean(per_label_covered[label_id]), 6
                ),
                "mean_frac_tumour_in_label": round(
                    mean(per_label_of_tumour[label_id]), 6
                ),
                "total_intersection_voxels": total_intersection[label_id],
                "total_label_voxels": total_label_voxels[label_id],
                "pooled_frac_label_covered_by_tumour": (
                    round(
                        total_intersection[label_id] / total_label_voxels[label_id], 8
                    )
                    if total_label_voxels[label_id]
                    else float("nan")
                ),
                "pooled_frac_tumour_in_label": (
                    round(total_intersection[label_id] / total_tumour_voxels, 8)
                    if total_tumour_voxels
                    else float("nan")
                ),
                "is_bone_label": int(label_id in set(bone_values.tolist())),
            }
        )

    # labels that never occur in any image only produce nans - drop those rows
    n_before = len(label_rows)
    label_rows = [
        row
        for row in label_rows
        if not any(
            isinstance(value, float) and np.isnan(value) for value in row.values()
        )
    ]
    if len(label_rows) < n_before:
        print(f"dropped {n_before - len(label_rows)} label rows containing nan")
    if not label_rows:
        raise SystemExit("all label rows contained nan - nothing to write")

    label_rows.sort(key=lambda row: -row["mean_frac_label_covered_by_tumour"])
    label_csv = args.output_dir / "tumour_ct_label_overlap.csv"
    with open(label_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(label_rows[0]))
        writer.writeheader()
        writer.writerows(label_rows)
    print(f"wrote {label_csv}")

    long_csv = args.output_dir / "tumour_ct_label_overlap_per_image.csv"
    with open(long_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_image_label_rows[0]))
        writer.writeheader()
        writer.writerows(per_image_label_rows)
    print(f"wrote {long_csv}")

    print("\ntop 10 CT labels by mean fraction covered by tumour:")
    for row in label_rows[:10]:
        print(
            f"  {row['label_name']:<25} covered={row['mean_frac_label_covered_by_tumour']:.5f} "
            f"dice={row['mean_dice_tumour_label']:.5f}"
        )


if __name__ == "__main__":
    main()
