"""Rank healthy organs by PET uptake across the PSMA dataset.

Walks imagesTr/labelsTr directly (every PSMARegPSMA_0* session, all
timepoints), measures the mean SUV inside each TotalSegmentator organ with the
PET lesion mask -- dilated, because the SUV halo around a lesion spills into
whatever it sits in -- removed, and aggregates over the dataset.

PET is read raw: no min-max, no norm_pet SUV clip, otherwise the numbers are
neither comparable across scans nor interpretable as SUV.
"""

import argparse
import csv
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import config
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import tqdm
from scipy import ndimage

# TotalSegmentator organ ids run 1..117; 0 is background.
N_LABELS = 118

# per-session organ statistics are dropped below this many non-tumour voxels
MIN_ORGAN_VOXELS = 50

# reference organ for the per-session normalized column. SUV scale drifts
# between scans and scanners, so the ratio ranking is usually more trustworthy
# than the absolute one, and liver is the standard PET reference.
REFERENCE_ORGAN = "liver"


def load_label_names(csv_path: Path) -> Dict[int, str]:
    names: Dict[int, str] = {}
    with open(csv_path, "r") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            names[int(row[0])] = row[1].strip()
    return names


def list_sessions(data_dir: Path) -> List[Tuple[str, str]]:
    """(case_id, tp) for every PSMARegPSMA_0* session found on disk.

    Discovery is on the CT channel; the PET and label files are checked in the
    worker so a missing one is reported per session instead of silently
    dropping the case here.
    """
    image_dir = data_dir / "imagesTr"
    sessions: List[Tuple[str, str]] = []
    for path in image_dir.glob("PSMARegPSMA_0*_0000_*.nii.gz"):
        case_id, _, tp = path.name.removesuffix(".nii.gz").split("_")[1:4]
        sessions.append((case_id, tp))
    return sorted(set(sessions))


def session_paths(data_dir: Path, case_id: str, tp: str) -> Dict[str, Path]:
    return {
        "pet": data_dir / "imagesTr" / f"PSMARegPSMA_{case_id}_0001_{tp}.nii.gz",
        "label_ct": data_dir / "labelsTr" / f"PSMARegPSMA_{case_id}_0000_{tp}.nii.gz",
        "label_pet": data_dir / "labelsTr" / f"PSMARegPSMA_{case_id}_0001_{tp}.nii.gz",
    }


def measure_session(
    args: Tuple[Path, str, str, int, bool],
) -> Tuple[str, str, Optional[pd.DataFrame], Optional[str]]:
    """Per-organ PET statistics for one session, tumour voxels excluded."""
    data_dir, case_id, tp, dilation_vox, with_percentiles = args
    paths = session_paths(data_dir, case_id, tp)

    missing = [key for key, path in paths.items() if not path.exists()]
    if missing:
        return case_id, tp, None, f"missing {', '.join(missing)}"

    pet_img = nib.load(paths["pet"].as_posix())
    pet = np.asarray(pet_img.dataobj, dtype=np.float32)
    organs = np.asarray(nib.load(paths["label_ct"].as_posix()).dataobj).astype(np.int32)
    lesion = np.asarray(nib.load(paths["label_pet"].as_posix()).dataobj) > 0

    if not (pet.shape == organs.shape == lesion.shape):
        return (
            case_id,
            tp,
            None,
            f"shape mismatch pet={pet.shape} ct_label={organs.shape} "
            f"pet_label={lesion.shape}",
        )

    if dilation_vox > 0 and lesion.any():
        lesion = ndimage.binary_dilation(
            lesion,
            structure=ndimage.generate_binary_structure(3, 1),
            iterations=dilation_vox,
        )

    voxel_vol_mm3 = float(np.abs(np.linalg.det(pet_img.affine[:3, :3])))

    healthy = ~lesion
    labels_flat = organs[healthy].ravel()
    pet_flat = pet[healthy].ravel()

    counts = np.bincount(labels_flat, minlength=N_LABELS)[:N_LABELS]
    sums = np.bincount(labels_flat, weights=pet_flat, minlength=N_LABELS)[:N_LABELS]
    total = np.bincount(organs.ravel(), minlength=N_LABELS)[:N_LABELS]

    label_ids = np.arange(N_LABELS)
    keep = (label_ids > 0) & (counts >= MIN_ORGAN_VOXELS)
    if not keep.any():
        return case_id, tp, None, "no organ above the voxel threshold"

    rows = {
        "case_id": case_id,
        "tp": tp,
        "label": label_ids[keep],
        "mean_suv": sums[keep] / counts[keep],
        "n_voxels": counts[keep],
        "n_voxels_total": total[keep],
        "tumour_voxels_removed": total[keep] - counts[keep],
        "volume_mm3": counts[keep] * voxel_vol_mm3,
    }

    if with_percentiles:
        medians = np.full(keep.sum(), np.nan, dtype=np.float64)
        p90s = np.full(keep.sum(), np.nan, dtype=np.float64)
        order = np.argsort(labels_flat, kind="stable")
        sorted_labels = labels_flat[order]
        sorted_pet = pet_flat[order]
        starts = np.searchsorted(sorted_labels, label_ids[keep], side="left")
        ends = np.searchsorted(sorted_labels, label_ids[keep], side="right")
        for i, (start, end) in enumerate(zip(starts, ends)):
            values = sorted_pet[start:end]
            medians[i] = np.median(values)
            p90s[i] = np.percentile(values, 90)
        rows["median_suv"] = medians
        rows["p90_suv"] = p90s

    return case_id, tp, pd.DataFrame(rows), None


def aggregate(per_session: pd.DataFrame, label_names: Dict[int, str]) -> pd.DataFrame:
    """Dataset-level table: one row per organ, ranked by mean SUV."""
    grouped = per_session.groupby("label")
    summary = pd.DataFrame(
        {
            # each scan weighted equally -- the headline number
            "mean_suv": grouped["mean_suv"].mean(),
            "std_suv": grouped["mean_suv"].std(),
            "median_of_session_means": grouped["mean_suv"].median(),
            "n_sessions": grouped["mean_suv"].size(),
            "mean_volume_mm3": grouped["volume_mm3"].mean(),
            "tumour_voxels_removed": grouped["tumour_voxels_removed"].sum(),
        }
    )
    # big organs dominate this one; kept as a cross-check on the headline
    weighted = per_session.assign(w=per_session["mean_suv"] * per_session["n_voxels"])
    summary["voxel_weighted_mean_suv"] = weighted.groupby("label")["w"].sum() / (
        per_session.groupby("label")["n_voxels"].sum()
    )

    if "ratio_to_reference" in per_session.columns:
        ref = per_session.groupby("label")["ratio_to_reference"]
        summary[f"mean_ratio_to_{REFERENCE_ORGAN}"] = ref.mean()
        summary[f"std_ratio_to_{REFERENCE_ORGAN}"] = ref.std()

    if "median_suv" in per_session.columns:
        summary["mean_median_suv"] = grouped["median_suv"].mean()
        summary["mean_p90_suv"] = grouped["p90_suv"].mean()

    summary = summary.reset_index()
    summary.insert(1, "organ", summary["label"].map(label_names).fillna("unknown"))
    return summary.sort_values("mean_suv", ascending=False).reset_index(drop=True)


def add_reference_ratio(
    per_session: pd.DataFrame, label_names: Dict[int, str]
) -> pd.DataFrame:
    """Per-session organ mean divided by that session's reference organ mean."""
    reference_ids = [
        lid for lid, name in label_names.items() if name == REFERENCE_ORGAN
    ]
    if not reference_ids:
        return per_session

    reference = per_session[per_session["label"] == reference_ids[0]]
    reference = reference.set_index(["case_id", "tp"])["mean_suv"]
    keys = pd.MultiIndex.from_frame(per_session[["case_id", "tp"]])
    denominator = reference.reindex(keys).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        per_session["ratio_to_reference"] = per_session["mean_suv"] / denominator
    return per_session


def plot_summary(summary: pd.DataFrame, out_path: Path, top_n: int) -> None:
    top = summary.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(4.0, 0.28 * len(top))))
    ax.barh(top["organ"], top["mean_suv"], xerr=top["std_suv"], color="#4c72b0")
    ax.set_xlabel("mean SUV (tumour excluded), averaged over sessions")
    ax.set_title(f"Healthy-organ PET uptake, top {len(top)}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = config.TrainingConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=cfg.data_dir)
    parser.add_argument(
        "--label-csv", type=Path, default=cfg.repo_dir / "total_segmentator_labels.csv"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=cfg.save_dir / "healthy_organ_activity"
    )
    parser.add_argument(
        "--tumour-dilation",
        type=int,
        default=2,
        help="voxels the lesion mask is dilated by before exclusion",
    )
    parser.add_argument("--workers", type=int, default=cfg.num_workers)
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument(
        "--percentiles",
        action="store_true",
        help="also compute per-organ median / p90 SUV (slower)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="debug: first N sessions"
    )
    args = parser.parse_args()

    label_names = load_label_names(args.label_csv)
    sessions = list_sessions(args.data_dir)
    if args.limit is not None:
        sessions = sessions[: args.limit]
    print(f"found {len(sessions)} PSMARegPSMA_0* sessions in {args.data_dir}/imagesTr")

    jobs = [
        (args.data_dir, case_id, tp, args.tumour_dilation, args.percentiles)
        for case_id, tp in sessions
    ]

    frames: List[pd.DataFrame] = []
    skipped: List[Tuple[str, str, str]] = []
    with Pool(processes=args.workers) as pool:
        for case_id, tp, frame, error in tqdm.tqdm(
            pool.imap_unordered(measure_session, jobs), total=len(jobs)
        ):
            if frame is None:
                skipped.append((case_id, tp, error or "unknown"))
            else:
                frames.append(frame)

    if skipped:
        print(f"skipped {len(skipped)} sessions:")
        for case_id, tp, error in skipped[:20]:
            print(f"  {case_id}_{tp}: {error}")
        if len(skipped) > 20:
            print(f"  ... and {len(skipped) - 20} more")

    if not frames:
        raise SystemExit("no session could be measured")

    per_session = pd.concat(frames, ignore_index=True)
    per_session = add_reference_ratio(per_session, label_names)
    per_session.insert(
        3, "organ", per_session["label"].map(label_names).fillna("unknown")
    )

    # sanity check that the PET really is SUV-scaled and not pre-normalized
    percentiles = np.percentile(per_session["mean_suv"], [50, 90, 99, 100])
    print(
        "per-session organ mean SUV percentiles "
        f"(p50/p90/p99/max): {percentiles[0]:.3f} / {percentiles[1]:.3f} / "
        f"{percentiles[2]:.3f} / {percentiles[3]:.3f}"
    )

    summary = aggregate(per_session, label_names)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_session_path = args.out_dir / "per_session_organ_suv.csv"
    summary_path = args.out_dir / "organ_suv_summary.csv"
    per_session.to_csv(per_session_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_summary(summary, args.out_dir / "organ_suv_top.png", args.top_n)

    columns = ["organ", "mean_suv", "std_suv", "n_sessions", "voxel_weighted_mean_suv"]
    ratio_column = f"mean_ratio_to_{REFERENCE_ORGAN}"
    if ratio_column in summary.columns:
        columns.append(ratio_column)
    print(f"\ntop {args.top_n} organs by mean SUV (tumour excluded):")
    print(summary[columns].head(args.top_n).to_string(index=False, float_format="%.3f"))
    print(f"\nwrote {per_session_path}\n      {summary_path}")


if __name__ == "__main__":
    main()
