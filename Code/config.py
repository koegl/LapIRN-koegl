from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Tuple


@dataclass
class TrainingConfig:
    save_dir: Path = Path("./saved")

    # Dataset
    data_dir = Path("/home/iml/fryderyk.koegl/data/PSMAReg_dataset/")
    split_path = Path("/home/iml/fryderyk.koegl/data/PSMAReg_dataset/split.json")
    val_fraction: float = 0.15
    use_cache_train: bool = True
    use_cache_valid: bool = True
    img_shape: Tuple[int, int, int] = (192, 192, 288)

    range_flow: float = 0.4

    in_channel: int = 4
    n_classes: int = 3
    lr: float = 1e-3
    start_channel: int = 2

    # train val
    epochs_lvl1: int = 6
    epochs_lvl2: int = 6
    epochs_lvl3: int = 6
    unfreeze_epoch_in_lvl2: int = 2
    unfreeze_epoch_in_lvl3: int = 3
    val_interval = 2

    # sum to 10
    w_jacobian: float = 1.0
    w_smooth: float = 1.0
    w_ct: float = 3.0
    w_pet: float = 1.0
    w_dice_ct: float = 1.0
    w_dice_pet: float = 1.0
    w_mtv: float = 0.7
    w_tlg: float = 0.7
    w_masked_jac: float = 0.6

    batch_size: int = 1
    shuffle: bool = False
    num_workers: int = 0

    lvl1_ncc_win: int = 7
    lvl1_ncc_scale: int = 1
    lvl2_ncc_win: int = 5
    lvl2_ncc_scale: int = 2
    lvl3_ncc_win: int = 7
    lvl3_ncc_scale: int = 3

    mlflow_tracking_uri: str = "sqlite:////home/iml/fryderyk.koegl/code/mlruns.db"
    mlflow_experiment: str = "PSMAReg_LapIRN"

    @property
    def img_shape_2(self) -> Tuple[int, int, int]:
        return (self.img_shape[0] // 2, self.img_shape[1] // 2, self.img_shape[2] // 2)

    @property
    def img_shape_4(self) -> Tuple[int, int, int]:
        return (self.img_shape[0] // 4, self.img_shape[1] // 4, self.img_shape[2] // 4)

    def to_dict(self) -> Dict[str, Any]:
        config = asdict(self)
        config["img_shape_2"] = self.img_shape_2
        config["img_shape_4"] = self.img_shape_4
        return config

    def to_mlflow_params(self) -> Dict[str, Any]:
        return {
            key: str(value) if isinstance(value, (Path, tuple)) else value
            for key, value in self.to_dict().items()
        }
