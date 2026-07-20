import json
import os

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"


import os
from pathlib import Path

import level1
import level2
import level3
import mlflow
import torch
import utils
from config import TrainingConfig
from torch.utils import data as torch_data


def main() -> None:
    # input shapes are fixed → cuDNN picks optimal 3D-conv kernels. Nearly free, often 10–30%
    torch.backends.cudnn.benchmark = False

    config = TrainingConfig()

    mlflow.set_tracking_uri("file:///home/iml/fryderyk.koegl/code/mlruns")
    mlflow.set_experiment("PSMAReg_LapIRN")
    with mlflow.start_run():
        utils.add_jobid_to_mlflow_run()

        (
            train_combined,
            train_dataset,
            train_dataset_synthetic,
            train_dataset_tubingen,
            train_dataset_nlst,
            val_dataset,
            val_dataset_tubingen,
            val_dataset_nlst,
            config_to_log,
        ) = utils.create_datasets(
            config,
        )

        mlflow.log_params(config_to_log)
        mlflow.log_text(
            json.dumps(config_to_log, indent=2),
            artifact_file="config.json",
        )

        train_generator = torch_data.DataLoader(
            train_combined,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
        )
        valid_generator = torch_data.DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
        )
        valid_tubingen_generator = (
            None
            if val_dataset_tubingen is None
            else torch_data.DataLoader(
                val_dataset_tubingen,
                batch_size=config.batch_size,
                shuffle=False,
            )
        )
        valid_nlst_generator = (
            None
            if val_dataset_nlst is None
            else torch_data.DataLoader(
                val_dataset_nlst,
                batch_size=config.batch_size,
                shuffle=False,
            )
        )

        with utils.track_peak_memory("training"):
            Path()
            level1.train_lvl1
            level2.train_lvl2
            level3.train_lvl3

            if config.overfit:
                print(
                    "Warning: Overfitting mode is enabled. This is for debugging purposes only."
                )
                train_generator = torch_data.DataLoader(
                    train_dataset_nlst,
                    batch_size=config.batch_size,
                    shuffle=config.shuffle,
                )

                paths_model_level1 = level1.train_lvl1(
                    config, train_generator, valid_generator
                )
                # path_model_level_1 = Path(
                #     "/home/iml/fryderyk.koegl/data/PSMAReg/models/PSMAReg_LapIRN_sedate-fish_local_stagelvl1_150.pth"
                # )
                path_model_level_1 = paths_model_level1["final"]

                paths_model_level2 = level2.train_lvl2(
                    config,
                    path_model_level_1,
                    train_generator,
                    valid_generator,
                )
                path_model_level_2 = paths_model_level2["final"]

                return
                path_model_level3 = level3.train_lvl3(
                    config,
                    path_model_level_2,
                    train_generator,
                    valid_generator,
                )
            else:
                paths_model_level1 = level1.train_lvl1(
                    config,
                    train_generator,
                    valid_generator,
                    valid_tubingen_generator,
                    valid_nlst_generator,
                )
                path_model_level_1 = paths_model_level1["best"]
                # print("skipping level 1, already trained")
                # path_model_level_1 = Path(
                #     "/lustre/groups/iml/data/PSMAReg/models/PSMAReg_LapIRN_stylish-kite-38590187_stagelvl1_best.pth"
                # )
                # print("skipping level 2, already trained")
                paths_model_level2 = level2.train_lvl2(
                    config,
                    path_model_level_1,
                    train_generator,
                    valid_generator,
                    valid_tubingen_generator,
                    valid_nlst_generator,
                )
                path_model_level_2 = paths_model_level2["best"]
                # path_model_level_2 = Path(
                #     "/home/iml/fryderyk.koegl/data/PSMAReg/models/PSMAReg_LapIRN_nosy-shrike-707_stagelvl2_best.pth"
                # )
                # path_model_level_2 = Path(
                #     "/lustre/groups/iml/data/PSMAReg/models/PSMAReg_LapIRN_adventurous-wren-38715925_stagelvl2_best.pth"
                # )
                path_model_level3 = level3.train_lvl3(
                    config,
                    path_model_level_2,
                    train_generator,
                    valid_generator,
                    valid_tubingen_generator,
                    valid_nlst_generator,
                )

        # print(f"Final model path: {path_model_level3}")


if __name__ == "__main__":
    main()
