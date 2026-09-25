"""Per image: tumour volume and how much of the tumour lies in bone / organs.

For every tumour label (PSMARegPSMA_<case>_0001_<tp>) in the label directory,
reads the matching organ label (..._0000_<tp>) and counts the tumour voxels
(value 1) that fall on a bone label (synthetic.BONE_LABEL_VALUES) and on any
other (non-bone, non-zero) label, and those on background (label 0). The
three counts partition the tumour. Only cases with tumour labels for at least
two time points are considered (single-scan cases cannot form a pair).

Writes one row per image to a CSV and prints a short summary.

Usage:
    python tumour_in_bone.py
    python tumour_in_bone.py --label-dir <.../labelsTs> --out tumour_in_bone_ts.csv
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "Code"))

from synthetic import BONE_LABEL_VALUES  # noqa: E402

LABEL_DIR = Path("/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/labelsTr")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label-dir", type=Path, default=LABEL_DIR)
    p.add_argument("--out", type=Path, default=REPO / "saved/tumour_in_bone.csv")
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()


def analyse(tumour_path: Path) -> dict:
    case_id, _, tp = tumour_path.name.removesuffix(".nii.gz").split("_")[1:4]
    organ_path = tumour_path.with_name(f"PSMARegPSMA_{case_id}_0000_{tp}.nii.gz")

    tumour_nii = nib.load(str(tumour_path))
    tumour = np.asarray(tumour_nii.dataobj) == 1
    organs = np.asarray(nib.load(str(organ_path)).dataobj)
    assert organs.shape == tumour.shape, f"shape mismatch for {tumour_path.name}"

    voxel_ml = float(np.prod(tumour_nii.header.get_zooms()[:3])) / 1000.0
    n_tumour = int(tumour.sum())
    bone = np.isin(organs, BONE_LABEL_VALUES)
    n_in_bone = int((tumour & bone).sum())
    n_in_organs = int((tumour & (organs != 0) & ~bone).sum())
    n_in_background = int((tumour & (organs == 0)).sum())
    assert n_in_bone + n_in_organs + n_in_background == n_tumour
    return {
        "case_id": case_id,
        "tp": tp,
        "tumour_voxels": n_tumour,
        "tumour_ml": n_tumour * voxel_ml,
        "tumour_in_bone_voxels": n_in_bone,
        "tumour_in_bone_ml": n_in_bone * voxel_ml,
        "fraction_in_bone": n_in_bone / n_tumour if n_tumour else float("nan"),
        "tumour_in_organs_voxels": n_in_organs,
        "tumour_in_organs_ml": n_in_organs * voxel_ml,
        "fraction_in_organs": n_in_organs / n_tumour if n_tumour else float("nan"),
        "tumour_in_background_voxels": n_in_background,
        "tumour_in_background_ml": n_in_background * voxel_ml,
        "fraction_in_background": (
            n_in_background / n_tumour if n_tumour else float("nan")
        ),
    }


def main() -> None:
    args = parse_args()
    paths = sorted(args.label_dir.glob("PSMARegPSMA_*_0001_*.nii.gz"))
    by_case: dict[str, list[Path]] = {}
    for path in paths:
        by_case.setdefault(path.name.split("_")[1], []).append(path)
    n_cases = len(by_case)
    by_case = {case: ps for case, ps in by_case.items() if len(ps) >= 2}
    paths = [p for ps in by_case.values() for p in ps]
    print(
        f"{len(paths)} tumour labels from {len(by_case)} of {n_cases} cases "
        f"with >= 2 time points in {args.label_dir}"
    )

    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(analyse, paths, chunksize=4))
    df = pd.DataFrame(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    with_tumour = df[df.tumour_voxels > 0]
    n_total = with_tumour.tumour_voxels.sum()
    bone_pooled = with_tumour.tumour_in_bone_voxels.sum() / n_total
    organs_pooled = with_tumour.tumour_in_organs_voxels.sum() / n_total
    background_pooled = with_tumour.tumour_in_background_voxels.sum() / n_total
    print(
        f"{len(with_tumour)} of {len(df)} images have tumour\n"
        f"tumour volume [ml]: median {with_tumour.tumour_ml.median():.1f}, "
        f"mean {with_tumour.tumour_ml.mean():.1f}, max {with_tumour.tumour_ml.max():.1f}\n"
        f"fraction in bone (per image): median {with_tumour.fraction_in_bone.median():.3f}, "
        f"mean {with_tumour.fraction_in_bone.mean():.3f}\n"
        f"fraction in bone (all tumour voxels pooled): {bone_pooled:.3f}\n"
        f"fraction in organs (per image): median {with_tumour.fraction_in_organs.median():.3f}, "
        f"mean {with_tumour.fraction_in_organs.mean():.3f}\n"
        f"fraction in organs (all tumour voxels pooled): {organs_pooled:.3f}\n"
        f"fraction in background (per image): median {with_tumour.fraction_in_background.median():.3f}, "
        f"mean {with_tumour.fraction_in_background.mean():.3f}\n"
        f"fraction in background (all tumour voxels pooled): {background_pooled:.3f}\n"
        f"wrote {args.out}"
    )


if __name__ == "__main__":
    main()
