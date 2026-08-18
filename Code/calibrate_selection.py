"""Calibrate the sel_ref_* / sel_scale_* checkpoint-selection constants.

Reads the lvl3 validation metric history of a finished run and prints a config
block to paste. ref = median over validation rounds, scale = robust spread
(IQR / 1.349), so a metric one spread better than typical scores q ~ 0.73.

SOURCE = "wandb" pulls straight from the API (no download needed). SOURCE =
"csv" reads chart exports from CSV_DIR instead: any .csv there is scanned and
columns are matched to metrics by name, so one combined export or one file per
chart both work.

sel_scale_ndv is not calibrated: %NDV sits at 0 for a healthy run, so its spread
carries no information. It is a judgement call -- the amount of folding, in %
above the equivalence threshold, that should visibly cost regularity.
"""

import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import utils

SOURCE = "wandb"  # "wandb" or "csv"

# --- wandb source ---
WANDB_PROJECT = "PSMAReg_LapIRN"
WANDB_ENTITY: Optional[str] = None  # None -> your default entity
RUN_NAME: Optional[str] = (
    "angry-mare-39459564"  # None -> most recent run in the project
)

# --- csv source ---
CSV_DIR = Path("~/Downloads").expanduser()

# (config field stem, logged metric key)
METRICS: List[Tuple[str, str]] = [
    ("dice_ct", "valid_lvl3/val_dice_ct"),
    ("hd95", "valid_lvl3/val_hd95"),
    ("mtv", "valid_lvl3/val_mtv_bias"),
    ("tlg", "valid_lvl3/val_tlg_bias"),
]
NDV_KEY = "valid_lvl3/val_ndv"


def clean(values: List[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def from_wandb() -> Dict[str, np.ndarray]:
    import wandb

    api = wandb.Api()
    path = f"{WANDB_ENTITY}/{WANDB_PROJECT}" if WANDB_ENTITY else WANDB_PROJECT
    if RUN_NAME is None:
        runs = api.runs(path, order="-created_at")
        run = runs[0]
    else:
        matches = api.runs(path, filters={"display_name": RUN_NAME})
        if len(matches) == 0:
            raise SystemExit(f"no run named {RUN_NAME!r} in {path}")
        run = matches[0]
    print(f"run: {run.name}  ({run.url})\n")

    keys = [key for _, key in METRICS] + [NDV_KEY]
    # scan_history is exact; run.history() downsamples to ~500 points
    rows = list(run.scan_history(keys=keys))
    out = {}
    for key in keys:
        out[key] = clean([r[key] for r in rows if r.get(key) is not None])
    return out


def from_csv() -> Dict[str, np.ndarray]:
    import csv

    files = sorted(glob.glob(str(CSV_DIR / "*.csv")))
    if not files:
        raise SystemExit(f"no .csv files in {CSV_DIR}")
    print(f"reading {len(files)} csv file(s) from {CSV_DIR}\n")

    keys = [key for _, key in METRICS] + [NDV_KEY]
    collected: Dict[str, List[float]] = {key: [] for key in keys}
    for path in files:
        with open(path, newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            continue
        for key in keys:
            # wandb names columns "<run> - <metric>"; skip the __MIN/__MAX bands
            columns = [
                c
                for c in rows[0]
                if c and key in c and not c.endswith(("__MIN", "__MAX"))
            ]
            for column in columns:
                for row in rows:
                    try:
                        collected[key].append(float(row[column]))
                    except (TypeError, ValueError):
                        pass
    return {key: clean(values) for key, values in collected.items()}


def main() -> None:
    data = from_wandb() if SOURCE == "wandb" else from_csv()

    print(f"{'metric':<10}{'n':>5}{'median':>12}{'spread':>12}{'min':>12}{'max':>12}")
    refs: Dict[str, float] = {}
    scales: Dict[str, float] = {}
    warnings: List[str] = []

    for stem, key in METRICS:
        values = data.get(key, np.array([]))
        if values.size == 0:
            print(f"{stem:<10}{0:>5}   no data -- is it being logged?")
            warnings.append(f"{stem}: no data, keeping the existing config values")
            continue
        median = float(np.median(values))
        q25, q75 = np.percentile(values, [25, 75])
        spread = float((q75 - q25) / 1.349)
        print(
            f"{stem:<10}{values.size:>5}{median:>12.5f}{spread:>12.5f}"
            f"{values.min():>12.5f}{values.max():>12.5f}"
        )
        if values.size < 10:
            warnings.append(f"{stem}: only {values.size} rounds, spread is noisy")
        refs[stem] = median
        if spread > 0:
            scales[stem] = spread
        else:
            warnings.append(f"{stem}: zero spread, keeping the existing scale")

    ndv = data.get(NDV_KEY, np.array([]))
    if ndv.size:
        above = float(np.mean(ndv - utils.SEL_NDV_EQUIVALENCE > 0) * 100)
        print(
            f"\n%NDV: median {np.median(ndv):.6f}, max {ndv.max():.6f}, "
            f"{above:.1f}% of rounds above the "
            f"{utils.SEL_NDV_EQUIVALENCE}% equivalence threshold"
        )

    for warning in warnings:
        print(f"[WARN] {warning}")

    print("\n--- paste into config.TrainingConfig ---")
    for stem, _ in METRICS:
        if stem in refs:
            print(f"    sel_ref_{stem}: float = {refs[stem]:.5g}")
    for stem, _ in METRICS:
        if stem in scales:
            print(f"    sel_scale_{stem}: float = {scales[stem]:.5g}")


if __name__ == "__main__":
    main()
