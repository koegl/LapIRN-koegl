from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass
class TrainingConfig:
    fixed_ct_path: Path = Path(
        "/home/iml/fryderyk.koegl/data/PSMAReg_dataset/imagesTr/PSMARegPSMA_0006_0000_00.nii.gz"
    )
    fixed_pet_path: Path = Path(
        "/home/iml/fryderyk.koegl/data/PSMAReg_dataset/imagesTr/PSMARegPSMA_0006_0001_00.nii.gz"
    )
    moving_ct_path: Path = Path(
        "/home/iml/fryderyk.koegl/data/PSMAReg_dataset/imagesTr/PSMARegPSMA_0006_0000_01.nii.gz"
    )
    moving_pet_path: Path = Path(
        "/home/iml/fryderyk.koegl/data/PSMAReg_dataset/imagesTr/PSMARegPSMA_0006_0001_01.nii.gz"
    )
    model_dir: Path = Path("./checkpoints/overfit/stages")
    ckpt_dir: Path = Path("./checkpoints/overfit/warped")

    imgshape: Tuple[int, int, int] = (192, 192, 288)

    range_flow: float = 0.4
    in_channel: int = 4
    n_classes: int = 3
    lr: float = 1e-5
    start_channel: int = 7
    antifold: float = 0.0
    smooth: float = 1.0
    w_ct: float = 1.0
    w_pet: float = 0.1

    iteration_multiplier: int = 1000
    iteration_lvl1: Optional[int] = None
    iteration_lvl2: Optional[int] = None
    iteration_lvl3: Optional[int] = None
    freeze_step: Optional[int] = None
    n_checkpoint: Optional[int] = None

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
    mlflow_experiment: str = "PSMAReg_LapIRN_overfit"

    train_lvl1: bool = True
    train_lvl2: bool = False
    train_lvl3: bool = False

    def __post_init__(self) -> None:
        if self.iteration_lvl1 is None:
            self.iteration_lvl1 = 1 * self.iteration_multiplier
        if self.iteration_lvl2 is None:
            self.iteration_lvl2 = 1 * self.iteration_multiplier
        if self.iteration_lvl3 is None:
            self.iteration_lvl3 = 2 * self.iteration_multiplier
        if self.freeze_step is None:
            self.freeze_step = int(self.iteration_lvl1 / 5)
        if self.n_checkpoint is None:
            self.n_checkpoint = int(self.iteration_lvl1 / 10)

    @property
    def imgshape_2(self) -> Tuple[int, int, int]:
        return tuple(dim // 2 for dim in self.imgshape)

    @property
    def imgshape_4(self) -> Tuple[int, int, int]:
        return tuple(dim // 4 for dim in self.imgshape)

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
