"""Split the HD95 tail into 'we created it' vs 'it was already there'.

Reads per_label_hd95.csv from per_label_and_cc_analysis.py. The per-label dump
showed the HD95 mean is entirely a tail phenomenon (median == p75 == one voxel),
so the only question left is whether that tail is inherent to the pairs or
inflicted by our field:

  created  : good before registration, bad after -> the field itself moved the
             label away. Fixable by regularisation (the IO objective currently
             runs with no smoothness term at all).
  inherent : bad before and bad after -> no correspondence / out of capture
             range. No loss weighting fixes this.
  improved : bad before, better after -> working as intended.
  floor    : good before and after -> already at the discretisation floor.

The number that matters is each bucket's share of the total HD95 mass: that is
the size of the prize for fixing it.
"""

import csv
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

# --- variables (define here, no argparse) ---------------------------------
ANALYSIS_DIR = Path(
    "/home/iml/fryderyk.koegl/code/LapIRN-koegl/submission_results/csvs/analysis"
)
PER_LABEL_CSV = ANALYSIS_DIR / "per_label_hd95.csv"
LABEL_NAMES_CSV = Path(
    "/home/iml/fryderyk.koegl/code/LapIRN-koegl/total_segmentator_labels.csv"
)

# "good" is anchored on the in-plane spacing (2.7344 mm): one voxel is the floor
# a nearest-warped label map can reach, so anything at or near it is solved.
GOOD_MM: float = 5.0
BAD_MM: float = 15.0


def load_label_names(path: Path) -> Dict[int, str]:
    """total_segmentator_labels.csv has no header: rows are `id,name`."""
    names: Dict[int, str] = {}
    if not path.exists():
        return names
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            try:
                names[int(row[0])] = row[1]
            except ValueError:
                continue
    return names


def classify(row: pd.Series) -> str:
    before, after = row["hd95_before"], row["hd95"]
    if not np.isfinite(before):
        return "no_baseline"
    if after <= GOOD_MM and before <= GOOD_MM:
        return "floor"
    if before <= GOOD_MM and after > BAD_MM:
        return "created"
    if before > BAD_MM and after > BAD_MM:
        return "inherent"
    if after < before:
        return "improved"
    return "worsened_mild"


def main() -> None:
    df = pd.read_csv(PER_LABEL_CSV)
    names = load_label_names(LABEL_NAMES_CSV)

    # non-finite hd95 (label vanished after warping) is dropped by the official
    # scorer, so it contributes no mass and is excluded here too
    df = df[np.isfinite(df["hd95"])].copy()
    if "hd95_before" not in df.columns or not np.isfinite(df["hd95_before"]).any():
        raise SystemExit("no hd95_before column: rerun the analysis with COMPUTE_BEFORE=True")

    df["bucket"] = df.apply(classify, axis=1)
    df["name"] = df["label"].map(lambda i: names.get(int(i), str(i)))
    total = df["hd95"].sum()

    print(f"thresholds: good <= {GOOD_MM} mm, bad > {BAD_MM} mm")
    print(f"{len(df)} scored (case, label) entries, hd95 sum = {total:.1f} mm\n")

    print("=== bucket shares of the total hd95 mass ===")
    grouped = (
        df.groupby("bucket")
        .agg(n=("hd95", "size"), mass=("hd95", "sum"), mean=("hd95", "mean"))
        .sort_values("mass", ascending=False)
    )
    grouped["share_%"] = 100 * grouped["mass"] / total
    print(grouped.to_string(float_format=lambda v: f"{v:.2f}"))

    created_share = grouped.loc["created", "share_%"] if "created" in grouped.index else 0.0
    print(
        f"\n-> 'created' is {created_share:.1f}% of the hd95 mass. "
        "That is the ceiling for what regularisation (IO smoothness) can win back."
    )

    for bucket in ("created", "inherent"):
        sub = df[df["bucket"] == bucket]
        if sub.empty:
            continue
        print(f"\n=== {bucket}: worst 15 entries ===")
        cols = ["case", "label", "name", "hd95_before", "hd95", "n_moving", "n_warped"]
        print(
            sub.nlargest(15, "hd95")[cols].to_string(
                index=False, float_format=lambda v: f"{v:.2f}"
            )
        )
        print(f"\n=== {bucket}: labels by mass ===")
        by_label = (
            sub.groupby(["label", "name"])
            .agg(n=("hd95", "size"), mass=("hd95", "sum"), mean=("hd95", "mean"))
            .sort_values("mass", ascending=False)
            .head(12)
        )
        print(by_label.to_string(float_format=lambda v: f"{v:.2f}"))

    # a label that grows a lot after warping is speckle/tearing, which is what
    # blows up a surface percentile; a label that shrinks was squashed
    created = df[df["bucket"] == "created"]
    if not created.empty:
        ratio = created["n_warped"] / created["n_moving"].clip(lower=1)
        print(
            f"\ncreated-bucket warped/moving voxel ratio: "
            f"median {ratio.median():.2f}, min {ratio.min():.2f}, max {ratio.max():.2f}"
        )

    out_path = ANALYSIS_DIR / "per_label_hd95_buckets.csv"
    df.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
