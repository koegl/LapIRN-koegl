import os

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"


from pathlib import Path

import level1
import level2
import level3
import torch
import utils
from config import TrainingConfig
from torch.utils import data as torch_data


def main() -> None:
    # input shapes are fixed → cuDNN picks optimal 3D-conv kernels. Nearly free, often 10–30%
    torch.backends.cudnn.benchmark = False

    config = TrainingConfig()

    with utils.start_logging_run(config):
        (
            train_combined,
            train_dataset,
            train_dataset_synthetic,
            train_dataset_tubingen,
            train_dataset_nlst,
            train_dataset_abdomen,
            val_dataset,
            val_dataset_tubingen,
            val_dataset_nlst,
            val_dataset_abdomen,
            config_to_log,
        ) = utils.create_datasets(
            config,
        )

        utils.log_config(config_to_log)

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
        valid_abdomen_generator = (
            None
            if val_dataset_abdomen is None
            else torch_data.DataLoader(
                val_dataset_abdomen,
                batch_size=config.batch_size,
                shuffle=False,
            )
        )

        if config.val_interval > 2:
            print(f"warning: val_interval is set to {config.val_interval}")

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
                    train_dataset,
                    batch_size=config.batch_size,
                    shuffle=config.shuffle,
                )

                if config.use_lvl1:
                    paths_model_level1 = level1.train_lvl1(
                        config, train_generator, valid_generator
                    )
                    # path_model_level_1 = Path(
                    #     "/home/iml/fryderyk.koegl/data/PSMAReg/models/PSMAReg_LapIRN_sedate-fish_local_stagelvl1_150.pth"
                    # )
                    path_model_level_1 = paths_model_level1["final"]
                else:
                    path_model_level_1 = None

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
                if config.use_lvl1:
                    # paths_model_level1 = level1.train_lvl1(
                    #     config,
                    #     train_generator,
                    #     valid_generator,
                    #     valid_tubingen_generator,
                    #     valid_nlst_generator,
                    #     valid_abdomen_generator,
                    # )
                    # path_model_level_1 = paths_model_level1["best"]
                    # path_model_level_1 = paths_model_level1["final"]
                    print("skipping level 1, already trained")
                    # path_model_level_1 = Path(
                    #     "/lustre/groups/iml/data/PSMAReg/models/PSMAReg_LapIRN_useful-pig-39155220_stagelvl1_best.pth"
                    # )
                else:
                    path_model_level_1 = None
                # paths_model_level2 = level2.train_lvl2(
                #     config,
                #     path_model_level_1,
                #     train_generator,
                #     valid_generator,
                #     valid_tubingen_generator,
                #     valid_nlst_generator,
                #     valid_abdomen_generator,
                # )
                # path_model_level_2 = paths_model_level2["best"]
                # path_model_level_2 = paths_model_level2["final"]
                print("skipping level 2, already trained")
                # path_model_level_2 = Path(
                # "/home/iml/fryderyk.koegl/data/PSMAReg/models/PSMAReg_LapIRN_nosy-shrike-707_stagelvl2_best.pth"
                # )
                path_model_level_2 = Path(
                    "/lustre/groups/iml/data/PSMAReg/models/PSMAReg_LapIRN_gentle-rat-39450513_stagelvl2_best.pth"
                )
                path_model_level3 = level3.train_lvl3(
                    config,
                    path_model_level_2,
                    train_generator,
                    valid_generator,
                    valid_tubingen_generator,
                    valid_nlst_generator,
                    valid_abdomen_generator,
                )
                # resume_model_path_lvl3 = Path(
                #     "/lustre/groups/iml/data/PSMAReg/models/PSMAReg_LapIRN_angry-mare-39459564_stagelvl3_best_combined.pth"
                # )
                # resume_optimizer_path_lvl3 = Path(
                #     "/lustre/groups/iml/data/PSMAReg/models/PSMAReg_LapIRN_angry-mare-39459564_stagelvl3_best_combined_optimizer.pth"
                # )
                # path_model_level3 = level3.train_lvl3(
                #     config,
                #     path_model_level_2,
                #     train_generator,
                #     valid_generator,
                #     valid_tubingen_generator,
                #     valid_nlst_generator,
                #     valid_abdomen_generator,
                #     resume_model_path=resume_model_path_lvl3,
                #     resume_optimizer_path=resume_optimizer_path_lvl3,
                # )

        # print(f"Final model path: {path_model_level3}")


if __name__ == "__main__":
    main()
