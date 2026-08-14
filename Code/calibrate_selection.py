"""Calibrate the sel_ref_* / sel_scale_* checkpoint-selection constants.

Pulls the lvl3 validation metric history of a finished run from MLflow and
prints a config block to paste. ref = median over validation rounds, scale =
robust spread (IQR / 1.349), so a metric one spread better than typical scores
q ~ 0.73.

sel_scale_ndv is not calibrated: %NDV sits at 0 for a healthy run, so its spread
carries no information. It is a judgement call -- the amount of folding, in %
above the equivalence threshold, that should visibly cost regularity.
"""

import os
from typing import Dict, List, Optional, Tuple

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import mlflow
import numpy as np
import utils
from mlflow.tracking import MlflowClient

TRACKING_URI = "file:///home/iml/fryderyk.koegl/mlflow_hpc/mlruns"
EXPERIMENT_NAME = "PSMAReg_LapIRN"
RUN_NAME: Optional[str] = None  # None -> latest run

# (config field stem, mlflow metric key)
METRICS: List[Tuple[str, str]] = [
    ("dice_ct", "valid_lvl3/val_dice_ct"),
    ("hd95", "valid_lvl3/val_hd95"),
    ("mtv", "valid_lvl3/val_mtv_bias"),
    ("tlg", "valid_lvl3/val_tlg_bias"),
]


def get_run_id(client: MlflowClient, run_name: Optional[str]) -> str:
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if run_name is None:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
    else:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.runName = '{run_name}'",
            max_results=1,
        )
    return runs[0].info.run_id


def history(client: MlflowClient, run_id: str, key: str) -> np.ndarray:
    values = [m.value for m in client.get_metric_history(run_id, key)]
    return np.array([v for v in values if np.isfinite(v)], dtype=float)


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)
    run_id = get_run_id(client, RUN_NAME)
    print(f"run_id: {run_id}\n")

    header = f"{'metric':<10}{'n':>5}{'median':>12}{'spread':>12}{'min':>12}{'max':>12}"
    print(header)

    refs: Dict[str, float] = {}
    scales: Dict[str, float] = {}
    warnings: List[str] = []

    for stem, key in METRICS:
        values = history(client, run_id, key)
        if values.size == 0:
            print(f"{stem:<10}{0:>5}   no data -- is it being logged?")
            warnings.append(f"{stem}: no data, keeping the existing config value")
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

    ndv = history(client, run_id, "valid_lvl3/val_ndv")
    if ndv.size:
        above = np.mean(ndv - utils.SEL_NDV_EQUIVALENCE > 0) * 100
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
