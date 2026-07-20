import json
import os

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"


import os
from pathlib import Path

import level1
import level2
import level3
import mlflow
import my_data
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

        train_ids, val_ids = my_data.get_train_val_split(
            data_dir=config.data_dir,
            split_path=config.split_path,
            val_fraction=config.val_fraction,
            tubingen=False,
        )
        synth_ids = [
            c
            for c, _ in my_data.list_single_session_sources(
                config.data_dir, exclude_case_ids=train_ids + val_ids
            )
        ]
        train_ids_tubingen, val_ids_tubingen = my_data.get_train_val_split(
            data_dir=config.data_dir,
            split_path=config.split_path,
            val_fraction=config.val_fraction,
            tubingen=True,
        )
        config_to_log = config.to_mlflow_params()
        config_to_log["train_indices"] = train_ids
        config_to_log["train_indices_tubingen"] = train_ids_tubingen
        config_to_log["synth_indices"] = synth_ids
        config_to_log["val_indices"] = val_ids
        config_to_log["val_indices_tubingen"] = val_ids_tubingen
        mlflow.log_params(config.to_mlflow_params())
        mlflow.log_text(
            json.dumps(config.to_mlflow_params(), indent=2),
            artifact_file="config.json",
        )

        val_dataset = my_data.PSMARegDataset(
            case_ids=val_ids,
            cfg=config,
            augment=False,
            use_cache=config.use_cache_valid,
            include_intermediate_pairs=False,
            num_workers=config.num_workers,
        )
        val_dataset_tubingen = None

        train_dataset = my_data.PSMARegDataset(
            case_ids=train_ids,
            cfg=config,
            augment=config.augment,
            use_cache=config.use_cache_train_real,
            include_intermediate_pairs=True,
            num_workers=config.num_workers,
        )
        all_train_datasets = [train_dataset]

        if config.use_synthetic:
            synth_dataset = my_data.SyntheticSourceDataset(
                cfg=config,
                source_ids=synth_ids,
                repeat=config.synthetic_repeat,
                use_cache=config.use_cache_train_synthetic,
                num_workers=config.num_workers,
                augment=config.augment,
            )
            all_train_datasets.append(synth_dataset)
        if config.use_tubingen:
            train_tubingen_dataset = my_data.TubingenDataset(
                case_ids=train_ids_tubingen,
                cfg=config,
                augment=config.augment,
                use_cache=config.use_cache_train_real,
                num_workers=config.num_workers,
            )
            all_train_datasets.append(train_tubingen_dataset)

            val_dataset_tubingen = my_data.TubingenDataset(
                case_ids=val_ids_tubingen,
                cfg=config,
                augment=False,
                use_cache=config.use_cache_valid,
                num_workers=config.num_workers,
            )

        train_combined = torch_data.ConcatDataset(all_train_datasets)

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
                    train_tubingen_dataset,
                    batch_size=config.batch_size,
                    shuffle=config.shuffle,
                )

                paths_model_level1 = level1.train_lvl1(
                    config, train_generator, valid_generator, valid_tubingen_generator
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

                path_model_level3 = level3.train_lvl3(
                    config,
                    path_model_level_2,
                    train_generator,
                    valid_generator,
                )
            else:
                # paths_model_level1 = level1.train_lvl1(
                #     config, train_generator, valid_generator, valid_tubingen_generator
                # )
                # path_model_level_1 = paths_model_level1["best"]
                print("skipping level 1, already trained")
                # path_model_level_1 = Path(
                #     "/lustre/groups/iml/data/PSMAReg/models/PSMAReg_LapIRN_stylish-kite-38590187_stagelvl1_best.pth"
                # )
                print("skipping level 2, already trained")
                # paths_model_level2 = level2.train_lvl2(
                #     config,
                #     path_model_level_1,
                #     train_generator,
                #     valid_generator,
                #     valid_tubingen_generator,
                # )
                # path_model_level_2 = paths_model_level2["best"]
                # path_model_level_2 = Path(
                #     "/home/iml/fryderyk.koegl/data/PSMAReg/models/PSMAReg_LapIRN_nosy-shrike-707_stagelvl2_best.pth"
                # )
                path_model_level_2 = Path(
                    "/lustre/groups/iml/data/PSMAReg/models/PSMAReg_LapIRN_adventurous-wren-38715925_stagelvl2_best.pth"
                )
                path_model_level3 = level3.train_lvl3(
                    config,
                    path_model_level_2,
                    train_generator,
                    valid_generator,
                    valid_tubingen_generator,
                )

        # print(f"Final model path: {path_model_level3}")


if __name__ == "__main__":
    main()
