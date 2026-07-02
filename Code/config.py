import socket
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

COMPUTER_NAME = socket.gethostname()

if COMPUTER_NAME == "janus":
    DATA_PATH = Path("/home/iml/fryderyk.koegl/data")
else:
    DATA_PATH = Path("/lustre/groups/iml/data")


@dataclass
class TrainingConfig:
    save_dir: Path = Path("/home/iml/fryderyk.koegl/code/LapIRN-koegl/saved")
    model_save_dir: Path = DATA_PATH / "PSMAReg/models"

    # Dataset
    data_dir = DATA_PATH / "PSMAReg/PSMAReg_dataset"
    cache_dir = DATA_PATH / "PSMAReg/affine_cache"
    split_path = DATA_PATH / "PSMAReg/PSMAReg_dataset/split.json"
    val_fraction: float = 0.15
    use_cache_train: bool = True
    use_cache_valid: bool = True
    img_shape: Tuple[int, int, int] = (192, 192, 288)

    # augmentation
    aug_use_flip: bool = True
    aug_use_ct_intensity: bool = True
    aug_use_pet_intensity: bool = True
    aug_use_z_crop: bool = True
    aug_use_z_crop_asym: bool = True
    aug_flip_prob: float = 0.5
    aug_ct_shift_range: Tuple[float, float] = (
        -0.010,
        0.010,
    )  # in normalized [0,1] CT space (~±50 HU)
    aug_ct_scale_range: Tuple[float, float] = (0.9, 1.1)
    aug_pet_scale_range: Tuple[float, float] = (0.85, 1.15)
    aug_max_crop_z_head: int = 40  # max z-slices removed from superior (head) end
    aug_max_crop_z_feet: int = 40  # max z-slices removed from inferior (feet) end
    aug_max_crop_z_head_asym: int = 10  # smaller than symmetric (40)
    aug_max_crop_z_feet_asym: int = 10
    aug_seed: int = 42

    range_flow: float = 0.4

    in_channel: int = 4
    n_classes: int = 3
    lr_lvl1: float = 1e-4
    lr_lvl2: float = 1e-3 * 0.5
    lr_lvl3: float = 1e-3 * 0.25
    start_channel_lvl1: int = 7
    start_channel_lvl2: int = 7
    start_channel_lvl3: int = 7

    # train val
    epochs_lvl1: int = 131
    epochs_lvl2: int = 121
    epochs_lvl3: int = 201
    unfreeze_epoch_in_lvl2: int = 10
    unfreeze_epoch_in_lvl3: int = 10
    val_interval: int = 2
    checkpoint_interval: int = 50

    accumulation_steps: int = 4

    # sum to 10
    w_jacobian: float = 3.0
    w_smooth: float = 3.0
    w_ct: float = 3.0
    w_pet: float = 1.0
    w_dice_ct: float = 3.0
    w_dice_pet: float = 3.0
    w_mtv: float = 1.5
    w_tlg: float = 1.5
    w_masked_jac: float = 0.6

    batch_size: int = 1
    shuffle: bool = True
    num_workers: int = 4

    lvl1_ncc_win: int = 9
    lvl2_ncc_win: int = 7
    lvl3_ncc_win: int = 9

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
