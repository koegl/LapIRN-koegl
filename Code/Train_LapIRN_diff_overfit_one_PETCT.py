import json
import os

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"


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
        train_ids, val_ids = my_data.get_train_val_split(
            data_dir=config.data_dir,
            split_path=config.split_path,
            val_fraction=config.val_fraction,
        )
        synth_ids = [
            c
            for c, _ in my_data.list_single_session_sources(
                config.data_dir, exclude_case_ids=train_ids + val_ids
            )
        ]
        config_to_log = config.to_mlflow_params()
        config_to_log["train_indices"] = train_ids
        config_to_log["val_indices"] = val_ids
        mlflow.log_params(config.to_mlflow_params())
        mlflow.log_text(
            json.dumps(config.to_mlflow_params(), indent=2),
            artifact_file="config.json",
        )

        train_dataset = my_data.PSMARegDataset(
            case_ids=train_ids,
            cfg=config,
            augment=config.augment,
            use_cache=config.use_cache_train,
            include_intermediate_pairs=True,
            num_workers=config.num_workers,
            # overfit="0049",
        )
        synth_dataset = my_data.SyntheticSourceDataset(
            cfg=config,
            source_ids=synth_ids,
            repeat=config.synthetic_repeat,
            use_cache=config.use_cache_train,
            num_workers=config.num_workers,
            augment=config.augment,
        )
        if config.use_synthetic:
            train_combined = torch_data.ConcatDataset([train_dataset, synth_dataset])
        else:
            train_combined = torch_data.ConcatDataset([train_dataset])

        val_dataset = my_data.PSMARegDataset(
            case_ids=val_ids,
            cfg=config,
            augment=False,
            use_cache=config.use_cache_valid,
            include_intermediate_pairs=False,
            num_workers=config.num_workers,
            # overfit="0049",
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

        with utils.track_peak_memory("training"):
            Path()
            level1.train_lvl1
            level2.train_lvl2
            level3.train_lvl3

            paths_model_level1 = level1.train_lvl1(
                config, train_generator, valid_generator
            )
            # # print("skipping level 1, already trained")
            # # path_model_level1 = Path(
            # #     "/lustre/groups/iml/data/PSMAReg/models/PSMAReg_LapIRN_stagelvl1_best.pth"
            # # )
            # paths_model_level2 = level2.train_lvl2(
            #     config,
            #     Path(
            #         "/home/iml/fryderyk.koegl/data/PSMAReg/models/PSMAReg_LapIRN_secretive-crane-465_stagelvl1_300.pth"
            #     ),
            #     train_generator,
            #     valid_generator,
            # )
            paths_model_level2 = level2.train_lvl2(
                config, paths_model_level1["best"], train_generator, valid_generator
            )
            # path_model_level_2 = Path(
            #     "/home/iml/fryderyk.koegl/data/PSMAReg/models/PSMAReg_LapIRN_nosy-shrike-707_stagelvl2_best.pth"
            # )
            # path_model_level_2 = Path(
            #     "/lustre/groups/iml/data/PSMAReg/models/PSMAReg_LapIRN_luminous-colt-866_stagelvl2_best.pth"
            # )
            path_model_level_2 = paths_model_level2["best"]
            path_model_level3 = level3.train_lvl3(
                config, path_model_level_2, train_generator, valid_generator
            )

        # print(f"Final model path: {path_model_level3}")


if __name__ == "__main__":
    main()
