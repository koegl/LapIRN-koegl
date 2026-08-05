import socket
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

COMPUTER_NAME = socket.gethostname()

if COMPUTER_NAME == "janus":
    DATA_PATH = Path("/home/iml/fryderyk.koegl/data")
else:
    DATA_PATH = Path("/lustre/groups/iml/data")


@dataclass
class TrainingConfig:
    repo_dir: Path = Path("/home/iml/fryderyk.koegl/code/LapIRN-koegl")
    save_dir: Path = repo_dir / "saved"
    # save_dir: Path = Path("/home/iml/fryderyk.koegl/code/LapIRN-koegl/saved")
    model_save_dir: Path = DATA_PATH / "PSMAReg/models"

    # overfit
    overfit: bool = False

    # Dataset
    data_dir = DATA_PATH / "PSMAReg/PSMAReg_dataset"
    cache_dir = DATA_PATH / "PSMAReg/affine_cache"
    cache_dir_poly = DATA_PATH / "PSMAReg/poly_cache"
    split_path = repo_dir / "split.json"
    val_fraction: float = 0.15
    use_cache_train_real: bool = False
    use_cache_train_synthetic: bool = False
    use_cache_valid: bool = False
    img_shape: Tuple[int, int, int] = (192, 192, 288)

    use_poly_affine: bool = False
    use_labels_directly: bool = False

    # False -> skip the quarter-res level entirely: lvl2 (half res) becomes the
    # coarsest level and runs standalone, lvl3 (full res) sits on top of it.
    # NB: lvl2 weights are not interchangeable between the two modes (its input
    # encoder loses the 3 velocity channels coming from lvl1).
    use_lvl1: bool = True

    use_synthetic: bool = False
    use_tubingen: bool = False
    use_nlst: bool = False
    use_abdomen: Optional[int] = None

    label_groups: List[List[int]] = field(
        default_factory=lambda: [
            list(range(92, 116, 2)),  # ribs (92..115)
            list(range(93, 117, 2)),  # ribs (92..115)
            list(range(26, 51, 2)),  # vertebrae (26..50)
            list(range(27, 50, 2)),  # vertebrae (26..50)
        ]
    )
    n_label_groups: int = field(init=False)
    sdt_clip_vox: float = 7.5

    # augmentation
    augment: bool = True
    aug_use_flip: bool = True
    aug_use_ct_intensity: bool = True
    aug_use_pet_intensity: bool = True
    aug_use_z_crop: bool = True
    aug_use_z_crop_asym: bool = True
    aug_flip_prob: float = 0.5
    aug_ct_shift_range: Tuple[float, float] = (
        -0.020,
        0.020,
    )  # in normalized [0,1] CT space (~±50 HU)
    aug_ct_scale_range: Tuple[float, float] = (0.9, 1.1)
    aug_pet_scale_range: Tuple[float, float] = (0.85, 1.15)
    aug_max_crop_z_head: int = 40  # max z-slices removed from superior (head) end
    aug_max_crop_z_feet: int = 40  # max z-slices removed from inferior (feet) end
    aug_max_crop_z_head_asym: int = 10  # smaller than symmetric (40)
    aug_max_crop_z_feet_asym: int = 10
    aug_seed: int = 42
    synthetic_repeat: int = 1

    range_flow: float = 0.4

    in_channel: int = field(init=False)
    n_classes: int = 3
    lr_lvl1: float = 0.0003
    lr_lvl2: float = 0.0002
    lr_lvl3: float = 0.00025
    # linear LR warmup (0 -> full lr) applied at the start of every level,
    # measured in epochs. Keep below unfreeze_epoch_in_lvl2/3 so the fresh
    # level head is fully warmed before the previous level is unfrozen.
    warmup_epochs: float = 5
    start_channel: int = 14

    # PWC-Net style local cost volume in lvl1, fused into the res-block trunk.
    #   "off"  -> baseline
    #   "corr" -> explicit local 3D correlation between encoded x/y features
    #   "feat" -> ablation control: same encoder and same extra channels, but
    #             the features are concatenated instead of correlated
    # The volume lives at img_shape // 8, so with radius 2 / dilation 1 the
    # search window covers +-2 voxels there == +-16 full-resolution voxels.
    cost_volume_mode: str = "off"
    cost_volume_radius: int = 2
    cost_volume_dilation: int = 1
    cost_volume_feat_channels: int = 16
    cost_volume_out_channels: int = 16

    # train val
    total_steps_lvl1: int = 1  # 80000
    total_steps_lvl2: int = 1  # 100000
    total_steps_lvl3: int = 20
    unfreeze_epoch_in_lvl2: int = 0
    unfreeze_epoch_in_lvl3: int = 0
    val_interval: int = 2
    checkpoint_interval: int = 50

    accumulation_steps: int = field(init=False)

    # loss weights
    w_jacobian: float = 2000.0
    w_smooth: float = 2.0
    w_ct: float = 5.0
    w_pet: float = 0.0
    w_dice_ct_lvl1: float = 3.0
    w_dice_ct_lvl2: float = 4.0
    w_dice_ct_lvl3: float = 5.0
    w_dice_pet: float = 0.0
    w_tlg: float = 2.0
    w_jacobian_tumor: float = 2.0
    w_bone_rigidity: float = 2.0
    w_dvf: float = 100.0

    # meta-learned / unrolled instance optimization (IO)
    # During lvl3 training, after the net emits F_X_Y we run a few differentiable
    # IO steps (same half-res SVF re-parametrization as run_io) and add the loss
    # on the *refined* field to the objective, so the net is trained to be a good
    # seed for IO rather than a good final answer on its own.
    use_unrolled_io: bool = False
    # number of inner IO steps to unroll (small on purpose: 3-8)
    unroll_K: int = 3
    # inner-loop step size (plain gradient descent, not Adam; needs its own tune,
    # Adam lr=0.1 in run_io is NOT the same as GD lr here)
    unroll_inner_lr: float = 0.1
    # scaling-and-squaring integration steps for the inner SVF
    unroll_n_integration: int = 7
    # "full"  : differentiate through the whole K-step trajectory (K x memory,
    #           double-backward). Best signal, heaviest.
    # "fomaml": first-order approximation - run the inner loop without retaining
    #           the trajectory graph, then backprop only through the final
    #           `base + refinement` construction. ~1x memory, most of the benefit.
    unroll_mode: str = "fomaml"
    # start unrolling only after this many epochs (let the feed-forward head warm
    # up first; 0 = from the beginning)
    unroll_start_epoch: int = 0
    # weight of the unrolled (post-IO) loss term, added on top of the normal
    # feed-forward lvl3 loss
    w_unrolled: float = 1.0
    # Which terms the unrolled *inner* loop descends. Default: match the deployed
    # run_io objective term-for-term (PET/TLG + tumor-jac and bone rigidity on),
    # so the net is seeded for the exact trajectory we run at test time. Turn a
    # group off to fall back to the CT-only subset (dice + NCC + Jacobian).
    # NB: with unroll_mode="full" these extra terms are differentiated twice; they
    # are only guaranteed first-order smooth, so keep parity paired with "fomaml".
    unroll_include_pet: bool = True
    unroll_include_rigidity: bool = True

    dice_pet_iou_threshold: float = 0.1

    batch_size: int = 1
    shuffle: bool = True
    num_workers: int = 8

    lvl1_ncc_win: int = 5
    lvl2_ncc_win: int = 7
    lvl3_ncc_win: int = 7

    mlflow_tracking_uri: str = "file:///home/iml/fryderyk.koegl/code/mlruns"
    mlflow_experiment: str = "PSMAReg_LapIRN"
    logger_backend: str = "both"  # one of: "mlflow", "wandb", "both", "none"
    wandb_project: str = "PSMAReg_LapIRN"
    wandb_entity: Optional[str] = None
    wandb_mode: Optional[str] = None  # e.g. "offline" on clusters without internet

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

    frozen_synthetic_path: Path = Path(
        "/home/iml/fryderyk.koegl/code/LapIRN-koegl/saved/synthetic_examle.pth"
    )
    overfit_synthetic: bool = False

    def __post_init__(self) -> None:
        self.n_label_groups = len(self.label_groups) if self.use_labels_directly else 0
        self.in_channel = 4 + 2 * self.n_label_groups

        if self.overfit:
            self.augment = False

            self.use_cache_train_real = True
            self.use_cache_train_synthetic = False
            self.use_cache_valid = False

            self.accumulation_steps = 1
        else:
            self.accumulation_steps = 4
