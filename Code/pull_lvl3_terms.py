"""Print epoch-averaged, weight-multiplied level-3 loss terms from MLflow.

Confirms (or refutes) the regularizer-domination hypothesis: if smooth /
masked_jac / tlg are much larger than dice_ct after weighting, the level-3
head is being pushed toward the near-identity (frozen level-2) field.

Note: tlg is only added for real pairs and dvf only for synthetic pairs, so
the epoch average mixes both; treat these two as approximate.
"""

import os
from typing import List, Optional, Tuple

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import config
import mlflow
from mlflow.tracking import MlflowClient


def get_latest_run_id(client: MlflowClient, experiment_name: str) -> str:
    experiment = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    run_id = runs[0].info.run_id
    return run_id


def get_run_id_by_name(
    client: MlflowClient, experiment_name: str, run_name: str
) -> str:
    experiment = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        max_results=1,
    )
    run_id = runs[0].info.run_id
    return run_id


def last_metric_value(client: MlflowClient, run_id: str, key: str) -> float:
    history = client.get_metric_history(run_id, key)
    if len(history) == 0:
        return float("nan")
    value = history[-1].value
    return value


def main() -> None:
    tracking_uri = "file:///home/iml/fryderyk.koegl/mlflow_hpc/mlruns"
    experiment_name = "PSMAReg_LapIRN"
    run_name: Optional[str] = (
        "legendary-elk-625"  # set to a run name to override "latest"
    )

    cfg = config.TrainingConfig()
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    if run_name is None:
        run_id = get_latest_run_id(client, experiment_name)
    else:
        run_id = get_run_id_by_name(client, experiment_name, run_name)

    # (label, mlflow metric key, weight)
    terms: List[Tuple[str, str, float]] = [
        ("ncc_ct", "train_lvl3/ncc_ct_epoch", cfg.w_ct),
        ("dice_ct", "train_lvl3/dice_ct_epoch", cfg.w_dice_ct),
        ("smooth", "train_lvl3/smooth_epoch", cfg.w_smooth),
        ("jacobian", "train_lvl3/jacob_epoch", cfg.w_jacobian),
        ("masked_jac", "train_lvl3/masked_jac_epoch", cfg.w_masked_jac),
        ("tlg", "train_lvl3/tlg_bias_epoch", cfg.w_tlg),
        ("dvf", "train_lvl3/dvf_epoch", cfg.w_dvf),
    ]

    rows: List[Tuple[str, float, float, float]] = []
    for name, key, weight in terms:
        raw = last_metric_value(client, run_id, key)
        weighted = raw * weight
        rows.append((name, raw, weight, weighted))

    rows.sort(key=lambda r: abs(r[3]), reverse=True)

    print(f"run_id: {run_id}")
    print(f"{'term':<14}{'raw':>14}{'weight':>10}{'weighted':>14}")
    for name, raw, weight, weighted in rows:
        print(f"{name:<14}{raw:>14.6f}{weight:>10.2f}{weighted:>14.4f}")


if __name__ == "__main__":
    main()
