"""Staring stencil leakage and erosion retention on every labelled challenge scan.

Both are properties of the bone labels alone (no registration involved), so
they are computed on all scans of all challenge patients (train, val, test and
single-session patients), using the ground-truth CT labels.

Per scan:
    leak_full        share of Staring stencil reads outside the centre voxel's
                     own bone, on the full bone mask
    leak_eroded      the same on the mask eroded per bone by the stencil
                     footprint (must be 0)
    voxels_retained  bone voxels kept by the erosion
    bones_retained   bones (>= BONE_MIN_VOXELS) keeping >= INTERIOR_MIN_VOXELS

Writes erosion_leakage/per_scan.csv, per_bone.csv and summary.txt next to this
script. Run: ../.venv/bin/python erosion_leakage.py
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))

import bone_metrics  # noqa: E402
import config  # noqa: E402
import utils  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "erosion_leakage"
RIB_LABELS = set(range(92, 116))
VERTEBRA_LABELS = set(range(26, 51))


def bone_group(label: int) -> str:
    if label in RIB_LABELS:
        return "ribs"
    if label in VERTEBRA_LABELS:
        return "vertebrae"
    return "other"


def main() -> None:
    cfg = config.TrainingConfig()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open(cfg.split_path_journal) as f:
        split = json.load(f)
    split_of = {c: name for name in ("train", "val", "test") for c in split[name]}

    paths = sorted((cfg.data_dir / "labelsTr").glob("PSMARegPSMA_0*_0000_*.nii.gz"))
    bone_values = torch.tensor(bone_metrics.BONE_LABELS, device=device)

    def load(path: Path):
        return path, np.asarray(nib.load(str(path)).dataobj).astype(np.int16)

    scan_rows, bone_rows = [], []
    with ThreadPoolExecutor(8) as pool:
        for i, (path, array) in enumerate(pool.map(load, paths)):
            case_id, _, tp = path.name.removesuffix(".nii.gz").split("_")[1:4]
            labels = torch.from_numpy(array).to(device).long()
            bone_map = torch.where(
                torch.isin(labels, bone_values), labels, torch.zeros_like(labels)
            ).float()
            full = bone_map > 0
            eroded = utils.erode_label_map(bone_map[None, None])[0, 0] > 0

            n_present = n_kept = 0
            for label in bone_metrics.BONE_LABELS:
                bone = labels == label
                n_voxels = int(bone.sum())
                if n_voxels < bone_metrics.BONE_MIN_VOXELS:
                    continue
                n_interior = int((bone & eroded).sum())
                vanished = n_interior < bone_metrics.INTERIOR_MIN_VOXELS
                n_present += 1
                n_kept += not vanished
                bone_rows.append(
                    {
                        "case_id": case_id,
                        "tp": tp,
                        "label": label,
                        "group": bone_group(label),
                        "voxels": n_voxels,
                        "voxels_retained": n_interior / n_voxels,
                        "vanished": vanished,
                    }
                )

            scan_rows.append(
                {
                    "split": split_of.get(case_id, "single_session"),
                    "case_id": case_id,
                    "tp": tp,
                    "n_bones": n_present,
                    "leak_full": bone_metrics.stencil_leakage(bone_map, full),
                    "leak_eroded": bone_metrics.stencil_leakage(bone_map, eroded),
                    "voxels_retained": float(eroded.sum() / full.sum()),
                    "bones_retained": n_kept / n_present if n_present else float("nan"),
                }
            )
            print(f"[{i + 1}/{len(paths)}] {case_id}_{tp}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scans = pd.DataFrame(scan_rows)
    bones = pd.DataFrame(bone_rows)
    scans.to_csv(OUT_DIR / "per_scan.csv", index=False)
    bones.to_csv(OUT_DIR / "per_bone.csv", index=False)

    metrics = ["leak_full", "leak_eroded", "voxels_retained", "bones_retained"]
    by_split = scans.groupby("split").agg(
        patients=("case_id", "nunique"),
        scans=("tp", "size"),
        **{f"{m}_mean": (m, "mean") for m in metrics},
        **{f"{m}_std": (m, "std") for m in metrics},
    )
    overall = scans[metrics].agg(["mean", "std", "min", "max"])
    by_group = bones.groupby("group").agg(
        n=("voxels_retained", "size"),
        voxels_retained=("voxels_retained", "mean"),
        vanished=("vanished", "mean"),
    )

    with pd.option_context("display.width", 200, "display.max_columns", 50):
        summary = (
            f"{len(scans)} scans of {scans.case_id.nunique()} patients\n\n"
            f"by split:\n{by_split.round(3).to_string()}\n\n"
            f"all scans:\n{overall.round(3).to_string()}\n\n"
            f"by bone group (all scans):\n{by_group.round(3).to_string()}\n"
        )
    (OUT_DIR / "summary.txt").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
