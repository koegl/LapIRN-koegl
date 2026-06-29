import json
import os

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import mlflow
import my_data
import torch
from config import TrainingConfig


def main() -> None:

    from Functions import generate_grid, generate_grid_unit

    g1 = generate_grid((4, 4, 4))
    g2 = generate_grid_unit((4, 4, 4))
    print("generate_grid range:", g1.min(), g1.max())
    print("generate_grid_unit range:", g2.min(), g2.max())

    return

    config = TrainingConfig()

    mlflow.set_tracking_uri("file:///home/iml/fryderyk.koegl/code/mlruns")
    mlflow.set_experiment("PSMAReg_LapIRN")
    with mlflow.start_run():
        train_ids, val_ids = my_data.get_train_val_split(
            data_dir=config.data_dir,
            split_path=config.split_path,
            val_fraction=config.val_fraction,
        )
        config_to_log = config.to_mlflow_params()
        config_to_log["train_indices"] = train_ids
        config_to_log["val_indices"] = val_ids
        mlflow.log_params(config.to_mlflow_params())
        mlflow.log_text(
            json.dumps(config.to_mlflow_params(), indent=2),
            artifact_file="config.json",
        )
    ds_no_flip = my_data.PSMARegDataset(case_ids=train_ids, cfg=config, augment=False)

    ds_flip = my_data.PSMARegDataset(case_ids=train_ids, cfg=config, augment=True)

    batch_no_flip = ds_no_flip[0]
    batch_flip = ds_flip[0]

    x_no_flip = batch_no_flip["x"]
    x_flip = batch_flip["x"]

    print("max diff:", (x_no_flip - x_flip).abs().max().item())
    print("are identical:", torch.allclose(x_no_flip, x_flip))

    # Check if flipping axis 0 of x_no_flip matches x_flip
    for axis in [0, 1, 2, 3]:
        flipped = torch.flip(x_no_flip, dims=[axis])
        print(
            f"matches flip on dim {axis}:", torch.allclose(flipped, x_flip, atol=1e-5)
        )


if __name__ == "__main__":
    main()
