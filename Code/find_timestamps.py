import datetime
import os

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import mlflow

mlflow.set_tracking_uri("file:////home/iml/fryderyk.koegl/mlflow_hpc/mlruns")
client = mlflow.MlflowClient()

runs = mlflow.search_runs(
    experiment_names=["PSMAReg_LapIRN"],
    filter_string="tags.mlflow.runName = 'unleashed-auk-838'",
)
run_id = runs.iloc[0]["run_id"]

history = client.get_metric_history(run_id, "valid_lvl2/val_dice_ct")

for metric in history:
    readable_time = datetime.datetime.fromtimestamp(metric.timestamp / 1000)
    print(metric.step, metric.value, readable_time)
