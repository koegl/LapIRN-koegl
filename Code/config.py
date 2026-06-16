from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Tuple


@dataclass
class TrainingConfig:
    data_path = (
        "/home/iml/fryderyk.koegl/data/PSMAReg_dataset/imagesTr/PSMARegPSMA_0006"
    )
    save_dir: Path = Path("./saved")

    imgshape: Tuple[int, int, int] = (192, 192, 288)

    range_flow: float = 0.4

    in_channel: int = 2
    n_classes: int = 3
    lr: float = 1e-3
    start_channel: int = 2

    iteration_lvl1: int = 10
    iteration_lvl2: int = 10
    iteration_lvl3: int = 20
    freeze_step: int = 5

    # sum to 10
    antifold: float = 1.0
    smooth: float = 1.0
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
    def imgshape_2(self) -> Tuple[int, int, int]:
        return (self.imgshape[0] // 2, self.imgshape[1] // 2, self.imgshape[2] // 2)

    @property
    def imgshape_4(self) -> Tuple[int, int, int]:
        return (self.imgshape[0] // 4, self.imgshape[1] // 4, self.imgshape[2] // 4)

    def to_dict(self) -> Dict[str, Any]:
        config = asdict(self)
        config["imgshape_2"] = self.imgshape_2
        config["imgshape_4"] = self.imgshape_4
        return config

    def to_mlflow_params(self) -> Dict[str, Any]:
        return {
            key: str(value) if isinstance(value, (Path, tuple)) else value
            for key, value in self.to_dict().items()
        }
