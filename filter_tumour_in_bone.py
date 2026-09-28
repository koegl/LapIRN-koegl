"""Filter the per-image tumour location table written by tumour_in_bone.py.

Keeps the images whose value is >= the minimum for every column below. Edit
the constants and re-run. With all minima at 0 every image is kept; images
without tumour have NaN fractions, which count as 0 here.
"""

from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent
IN_CSV = REPO / "saved/tumour_in_bone.csv"
OUT_CSV = REPO / "saved/tumour_in_bone_filtered.csv"

# minima, inclusive
MIN_TUMOUR_ML = 800.0

MIN_TUMOUR_IN_BONE_VOXELS = 0
MIN_TUMOUR_IN_BONE_ML = 0.0
MIN_FRACTION_IN_BONE = 0.0

MIN_TUMOUR_IN_ORGANS_VOXELS = 0
MIN_TUMOUR_IN_ORGANS_ML = 0.0
MIN_FRACTION_IN_ORGANS = 0.3

MIN_TUMOUR_IN_BACKGROUND_VOXELS = 0
MIN_TUMOUR_IN_BACKGROUND_ML = 0.0
MIN_FRACTION_IN_BACKGROUND = 0.0

MINIMA = {
    "tumour_ml": MIN_TUMOUR_ML,
    "tumour_in_bone_voxels": MIN_TUMOUR_IN_BONE_VOXELS,
    "tumour_in_bone_ml": MIN_TUMOUR_IN_BONE_ML,
    "fraction_in_bone": MIN_FRACTION_IN_BONE,
    "tumour_in_organs_voxels": MIN_TUMOUR_IN_ORGANS_VOXELS,
    "tumour_in_organs_ml": MIN_TUMOUR_IN_ORGANS_ML,
    "fraction_in_organs": MIN_FRACTION_IN_ORGANS,
    "tumour_in_background_voxels": MIN_TUMOUR_IN_BACKGROUND_VOXELS,
    "tumour_in_background_ml": MIN_TUMOUR_IN_BACKGROUND_ML,
    "fraction_in_background": MIN_FRACTION_IN_BACKGROUND,
}


def main() -> None:
    df = pd.read_csv(IN_CSV, dtype={"case_id": str, "tp": str})

    keep = pd.Series(True, index=df.index)
    for column, minimum in MINIMA.items():
        passes = df[column].fillna(0) >= minimum
        if minimum > 0:
            print(f"{column} >= {minimum}: {int(passes.sum())} of {len(df)}")
        keep &= passes

    filtered = df[keep]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(OUT_CSV, index=False)

    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(filtered.round(3).to_string(index=False))
    print(f"\n{len(filtered)} of {len(df)} images kept, wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
