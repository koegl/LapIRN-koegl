import json

import level1
import level2
import level3
import mlflow
import my_data
from config import TrainingConfig
from torch.utils import data as torch_data


def main() -> None:

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

        train_dataset = my_data.PSMARegDataset(
            data_dir=config.data_dir,
            case_ids=train_ids,
            use_cache=config.use_cache_train,
            # overfit="0049",
        )
        val_dataset = my_data.PSMARegDataset(
            data_dir=config.data_dir,
            case_ids=val_ids,
            use_cache=config.use_cache_valid,
            # overfit="0049",
        )

        train_generator = torch_data.DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            num_workers=config.num_workers,
        )
        valid_generator = torch_data.DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )

        path_model_level1 = level1.train_lvl1(config, train_generator, valid_generator)
        path_model_level2 = level2.train_lvl2(
            config, path_model_level1, train_generator, valid_generator
        )
        level3.train_lvl3(config, path_model_level2, train_generator, valid_generator)


if __name__ == "__main__":
    main()
