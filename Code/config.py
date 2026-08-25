import socket
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

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
    use_cache_train_real: bool = True
    use_cache_train_synthetic: bool = True
    use_cache_valid: bool = True
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
    start_channel: int = 7

    # Per-level res-block trunk capacity, independent of start_channel (so no
    # inter-level tensor shape changes).
    #   *_n_resblocks       -> depth of the res-block group. Also grows the
    #                          receptive field (~+4 voxels per block at the
    #                          trunk's own resolution), which caps how much
    #                          misalignment the level can perceive; widening
    #                          does not.
    #   *_resblock_expansion -> inverted bottleneck inside each block: conv1
    #                          lifts to 4x*expansion channels, conv2 projects
    #                          back. Roughly multiplies per-block params.
    n_resblocks: int = 5  # 5
    resblock_expansion: int = 1  # 1

    # PWC-Net style local cost volume in lvl1, fused into the res-block trunk.
    #   "off"  -> baseline
    #   "corr" -> explicit local 3D correlation between encoded x/y features
    #   "feat" -> ablation control: same encoder and same extra channels, but
    #             the features are concatenated instead of correlated
    # The volume lives at img_shape // 8, so with radius 2 / dilation 1 the
    # search window covers +-2 voxels there == +-16 full-resolution voxels.
    # Diagnostic. When true, lvl1 runs a single validation pass, writes the
    # per case/label centroid distance left over after pre-registration to
    # save_dir/prereg_residual.csv, prints the percentiles, and exits without
    # training. Measured from the label maps, so it does not depend on what
    # the network manages to correct.
    measure_prereg_residual: bool = False

    cost_volume_mode: str = "off"
    cost_volume_radius: int = 2
    cost_volume_dilation: int = 1
    cost_volume_feat_channels: int = 16
    cost_volume_out_channels: int = 16

    # train val
    total_steps_lvl1: int = 100000
    total_steps_lvl2: int = 100000
    total_steps_lvl3: int = 140000
    unfreeze_epoch_in_lvl2: int = 10
    unfreeze_epoch_in_lvl3: int = 10
    val_interval: int = 2
    checkpoint_interval: int = 50

    accumulation_steps: int = field(init=False)

    # --- lvl3 checkpoint selection ---------------------------------------
    sel_ref_dice_ct: float = 0.278
    sel_ref_hd95: float = 8.3319
    sel_ref_mtv: float = 0.044138
    sel_ref_tlg: float = 0.045699

    sel_scale_hd95: float = 0.13427
    sel_scale_ndv: float = 0.005
    sel_scale_dice_ct: float = 0.0090181
    sel_scale_mtv: float = 0.0051981
    sel_scale_tlg: float = 0.0050091

    # --- HD95 validation logging ------------------------------------------
    hd95_spacing_mm: Tuple[float, float, float] = (2.7344, 2.7344, 3.27)

    # loss weights
    w_non_diff: float = 10000.0
    w_smooth: float = 5.0
    w_ct: float = 5.0
    w_pet: float = 0.0
    w_dice_ct_lvl1: float = 3.0
    w_dice_ct_lvl2: float = 4.0
    w_dice_ct_lvl3: float = 5.0
    w_dice_pet: float = 0.0

    # --- per-label CT dice weighting --------------------------------------
    pet_visible_labels: List[int] = field(
        default_factory=lambda: [1, 2, 3, 4, 5, 7, 19, 21, 22, 64]
    )
    # 1.0 disables the weighting entirely (all labels equal)
    w_dice_pet_visible: float = 2.0
    w_tlg: float = 5.0
    w_jacobian_tumor: float = 5.0
    w_mtv: float = 20.0
    w_mtv_avg: float = 0.5
    w_bone_rigidity: float = 0.2

    # io params
    io_lr: float = 0.5e-1
    io_it: int = 9
    # Axial band the CT labels for the IO dice term are computed on, as a
    # half-open slice range [start, stop) -- so (141, 241) is the 100 slices
    # 141..240 inclusive. An explicit range rather than a symmetric margin:
    # TotalSegmentator's runtime scales with axial extent, so the slices it is
    # spent on should be the ones whose pre-IO alignment is worst, and those are
    # not centred in the volume. Labels are zero outside the band, so the dice
    # term simply does not see that anatomy.
    io_seg_z_range: Tuple[int, int] = (-1, -1)  # (141, 241)
    w_io_ncc: float = 5.0
    w_io_dice: float = 5.0
    w_io_non_diff: float = 10.0  # 2000.0
    w_io_smooth: float = 0.0  # 2.0
    w_io_mtv: float = 200.0  # 500.0
    w_io_mtv_avg: float = 0.0  # 5000.0
    w_io_tlg: float = 80.0  # 100.0
    w_io_jacobian_tumor: float = 20  # 5.0
    w_io_bone_rigidity: float = 1.0  # 2.0

    # --- per-lesion (connected component) tumour bias terms in IO -----------
    # The scored MTV / TLG are global sums, so per-lesion errors cancel: on the
    # validation set the size-weighted per-lesion bias is ~9.5% / ~9.8% while
    # the scored global bias is ~1.8%. These terms pin each lesion individually
    # and sit alongside (not instead of) the global ones. Set the weights to 0
    # to disable and skip the component warp entirely.
    w_io_mtv_cc: float = 80.0
    w_io_tlg_cc: float = 80.0
    w_io_mtv_avg_cc: float = 150.0
    # each kept component costs one full-resolution channel in the warp, so this
    # caps the memory; components beyond it are still covered by the global terms
    io_cc_max_components: int = 8
    # below this the relative bias is dominated by interpolation noise
    io_cc_min_voxels: int = 20

    # Sub-weights inside the rigidity term, applied before w_bone_rigidity.
    # The three conditions have different natural magnitudes (det and ortho are
    # dimensionless strain measures, affine is a bending energy in voxel units),
    # so summing them with equal weight lets whichever is largest be the only
    # one w_bone_rigidity actually controls. Log train_lvl3/rig_{det,ortho,
    # affine} first, then set these. Defaults reproduce the old equal sum.
    w_rig_det: float = 1.0
    w_rig_ortho: float = 1.0
    w_rig_affine: float = 1.0
    # Swap the local finite-difference rigidity term for a per-label rigid fit
    # (utils.per_label_rigid_loss). The old term's stencils read one voxel past
    # the mask, so ~53% of the voxels entering it were soft tissue and ~45% of
    # its gradient landed outside bone; it scored ~50 on fields where bones do
    # not move at all. The per-label fit reads no neighbourhood, so it leaks
    # nothing and lets bones move relative to each other.
    # Units are voxel^2 rather than dimensionless strain, so w_bone_rigidity
    # MUST be retuned when this is switched on. w_rig_* are ignored.
    use_per_label_rigidity: bool = True
    # labels smaller than this are skipped: the rigid fit is ill-posed
    rigidity_min_voxels: int = 50
    w_dvf: float = 100.0

    # --- auxiliary PET-tumour segmentation head (lvl3 only) ---------------
    # A second head on the lvl3 trunk predicts the fixed and the (lvl2-warped)
    # moving lesion mask, both in the fixed frame. Labels are needed at training
    # time only, so this costs nothing at inference -- and the predicted mask
    # comes out of the same forward pass, which is what makes a tumour-aware IO
    # objective affordable inside the challenge time limit.
    # Single-session patients can be used here through the synthetic branch,
    # which is the point: they carry no registration signal but full lesion
    # supervision.
    use_seg_pet_head: bool = False
    # width of the head's hidden conv. It runs at full resolution, so this is
    # the memory knob: each channel costs a 192x192x288 activation.
    seg_pet_head_channels: int = 32
    w_seg_pet: float = 0.1
    # linear ramp of w_seg over this many epochs (0 = full weight immediately).
    # The head starts from noise; without a ramp its early gradients reach the
    # shared trunk while the flow head is still warming up.
    seg_pet_warmup_epochs: float = 5.0

    # --- auxiliary bone segmentation head (lvl3 only) ---------------------
    # Same shape as the lesion head but predicting BONE_LABEL_VALUES vs rest.
    # Separate from the lesion head on purpose: the two tasks key off different
    # evidence (CT density vs PET uptake) and should not share one narrow
    # bottleneck. The dice loss only supervises the *field* (make warped labels
    # overlap); nothing in it forces the trunk to decide "this voxel is bone",
    # which is what this head adds.
    # NB: not wired into IO. Rigidity is the most forgiving consumer of a
    # predicted mask (a regional regulariser tolerates an eroded/dilated mask),
    # but validate bone_dice_moving before feeding it anything.
    use_seg_bone_head: bool = False
    seg_bone_head_channels: int = 32
    w_seg_bone: float = 0.1
    seg_bone_warmup_epochs: float = 5.0

    # gradient-conflict diagnostic: logs the full pairwise matrix of cosines
    # between every lvl3 loss term's gradient w.r.t. the shared parameters, plus
    # each term's share of the total gradient norm and its cosine against the
    # whole objective. Persistently negative pairs genuinely fight, so gradient
    # surgery or a multi-stage model is justified; near-zero means the terms are
    # orthogonal and any tension is a weighting problem. There is no
    # accuracy/tumour grouping: an aggregate dilutes the cosines of everything
    # inside it (see the comment at the call site in level3.py).
    # Costs one extra backward pass per term per measurement.
    log_grad_conflict: bool = False
    # measured every N validation intervals (1 = at every validation)
    grad_conflict_every_n_val: int = 1
    # window length for the running cos mean / std / fraction-negative
    grad_conflict_window: int = 20

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

    # image similarity term used at every pyramid level: "ncc" (local
    # normalized cross correlation, uses the lvl*_ncc_win above) or "mind"
    # (MIND-SSC descriptor MSE, uses mind_radius / mind_dilation below).
    # NB: MIND is a plain MSE in [0, 1] while NCC is in [-1, 0], so w_ct / w_pet
    # have to be re-tuned when switching.
    similarity_metric: Literal["ncc", "mind"] = "ncc"
    mind_radius: int = 2
    mind_dilation: int = 2

    mlflow_tracking_uri: str = "file:///home/iml/fryderyk.koegl/code/mlruns"
    mlflow_experiment: str = "PSMAReg_LapIRN"
    logger_backend: str = "wandb"  # one of: "mlflow", "wandb", "both", "none"
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

        if self.similarity_metric == "mind":
            self.w_ct *= 3.4

        if self.overfit:
            self.augment = False

            self.use_cache_train_real = True
            self.use_cache_train_synthetic = False
            self.use_cache_valid = False

            self.accumulation_steps = 1
        else:
            self.accumulation_steps = 4
