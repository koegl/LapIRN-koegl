"""Evaluation of the journal experiments (see journal_extension_plan.md).

Every evaluation run is identified by its ID from the plan (T1, I5, L2, ...).
A run is a trained network plus, optionally, instance optimisation (IO) on top.
For every follow-up -> baseline pair of the chosen split it registers the pair
(affine prereg -> network -> optional IO) and writes one row per pair to
<OUT_DIR>/<split>/<run_id>.csv. Ground-truth labels are used for evaluation
only; IO gets the predicted (TotalSegmentator + nnU-Net) labels.

Run from the Code directory: python evaluate.py
"""

import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import affine_reg
import bone_metrics
import Functions
import hd95_official
import instance_opt
import miccai2020_model_stage
import my_data
import numpy as np
import pandas as pd
import synthetic
import torch
import torch.nn.functional as F
import tqdm
from config import DATA_PATH, TrainingConfig
from scipy import ndimage

# --- training runs: plan ID -> cluster run name (as in the .pth file names) ---
TRAINING_RUNS: Dict[str, str] = {
    "T1": "",  # neither
    "T2": "overjoyed-slug-41355419",  # PET
    "T3": "overjoyed-slug-41355436",  # rigidity
    "T4": "",  # PET + rigidity
}

"""
Command to download latest model
scp "fryderyk.koegl@hpc-submit02.scidom.de:/lustre/groups/iml/data/PSMAReg/models/psmareg_journal_overjoyed-slug-41355419_stagelvl3_best_accuracy_capped.pth" /home/iml/fryderyk.koegl/data/PSMAReg/models_journal/
"""

# config fields that differ from TrainingConfig() for a training run (e.g. the
# architecture); must match the checkpoint
TRAINING_CFG_OVERRIDES: Dict[str, Dict[str, object]] = {
    "T1": {},
    "T2": {},
    "T3": {},
    "T4": {},
}


@dataclass(frozen=True)
class IOSpec:
    pet: bool
    rigidity: bool
    # IO weight overrides on top of the frozen IO settings in TrainingConfig,
    # e.g. a single PET term switched off for the leave-one-out runs
    cfg_overrides: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalRun:
    network: Optional[str] = None  # key of TRAINING_RUNS; None = no network
    io: Optional[IOSpec] = None  # None = network output only
    checkpoint: str = "best_accuracy_capped"
    # without a network: apply our affine prereg (True) or the identity (False)
    affine: bool = True
    # precomputed baseline: a key of BASELINE_METHODS (its chosen config) or a
    # directory of fields; replaces affine and network
    external: Optional[object] = None
    # False: score the moving scan as it is (no resampling at all); only for
    # the identity, where it gives the exact "initial" row (MTV / TLG change 0)
    warp: bool = True


IO_NONE = IOSpec(pet=False, rigidity=False)
IO_PET = IOSpec(pet=True, rigidity=False)
IO_RIG = IOSpec(pet=False, rigidity=True)
IO_BOTH = IOSpec(pet=True, rigidity=True)

# one PET term removed per leave-one-out run
LOO_TERMS: Dict[str, Dict[str, float]] = {
    "L1": {"w_io_mtv": 0.0},  # MTV
    "L2": {"w_io_mtv_avg": 0.0},  # MTV-avg
    "L3": {"w_io_tlg": 0.0},  # TLG
    "L4": {"w_io_jacobian_tumor": 0.0},  # tumour Jacobian
}

# rigidity comparison (experiment 2): PET IO on the PET-only network (T2) plus
# one rigidity method; R1 (no rigidity) is I2. The weight of each method is
# swept on val by overriding w_io_bone_rigidity (step 7).
RIGIDITY_METHODS: Dict[str, str] = {
    "R2": "staring",
    "R3": "staring_eroded",
    "R4": "kabsch",
}

# --- evaluation runs: plan ID -> what to run ---
# W* (PET-weight sweep) is added once the IO settings are frozen; B*
# (baselines) are not networks.
EVAL_RUNS: Dict[str, EvalRun] = {
    "T1": EvalRun("T1"),
    "T2": EvalRun("T2"),
    "T3": EvalRun("T3"),
    "T4": EvalRun("T4"),
    "T1-tum": EvalRun("T1", checkpoint="best_tumour"),
    "I1": EvalRun("T1", IO_NONE),
    "I2": EvalRun("T2", IO_PET),
    "I3": EvalRun("T3", IO_RIG),
    "I4": EvalRun("T4", IO_BOTH),
    "I5": EvalRun("T1", IO_PET),
    "I6": EvalRun("T1", IO_RIG),
    "I7": EvalRun("T1", IO_BOTH),
    **{
        run_id: EvalRun("T1", IOSpec(pet=True, rigidity=False, cfg_overrides=terms))
        for run_id, terms in LOO_TERMS.items()
    },
    "R1": EvalRun("T2", IO_PET),
    **{
        run_id: EvalRun(
            "T2",
            IOSpec(
                pet=True,
                rigidity=True,
                cfg_overrides={"io_rigidity_method": method},
            ),
        )
        for run_id, method in RIGIDITY_METHODS.items()
    },
    # baselines (experiment 4); B5 / B6 (full method with / without IO) are T4 / I4
    "B1": EvalRun(affine=False, warp=False),  # initial alignment, not resampled
    # the identity pushed through the scorer's warp: its MTV / TLG change is the
    # resampling floor every warped method carries (findings.md)
    "B1-floor": EvalRun(affine=False),
    "B2": EvalRun(),  # our affine prereg only
    "B3": EvalRun(external="niftyreg"),
    "B4": EvalRun(external="convexadam"),
}

# precomputed baseline fields, written by journal_results/tune_*.py:
# <BASELINE_DIR>/<method>/<config>/fields/disp_<case>_<tp moving>_<tp fixed>.nii.gz,
# the config chosen on val named in <BASELINE_DIR>/<method>/chosen.json
BASELINE_DIR = Path("/home/iml/fryderyk.koegl/code/LapIRN-koegl/journal_results/baselines")
BASELINE_METHODS = ("niftyreg", "convexadam")

# --- what to run (no argparse) ---
SPLIT = "val"  # "val" while developing; "test" only for frozen final runs
RUN_IDS: List[str] = ["T1"]
OUT_DIR = Path("/home/iml/fryderyk.koegl/code/LapIRN-koegl/evaluation")
# predicted CT organ labels (TotalSegmentator) and PET lesion labels (nnU-Net),
# same file names as labelsTr; the only labels IO may see
PRED_LABEL_DIR = DATA_PATH / "PSMAReg/PSMAReg_dataset/labelsTr_pred"
GT_LABEL_DIR = DATA_PATH / "PSMAReg/PSMAReg_dataset/labelsTr"
# a run whose CSV exists is skipped; delete the CSV to rerun it
SKIP_EXISTING = True

# --- per-lesion analysis ---
# a lesion is a 26-connected component of the moving scan's lesion mask
LESION_STRUCTURE = np.ones((3, 3, 3), dtype=bool)
# bone-associated if at least this fraction of its voxels lies on a bone label
LESION_BONE_FRACTION = 0.5
# size bins by volume; below LESION_MIN_VOXELS the relative change is mostly
# interpolation noise ("tiny", excluded from the per-lesion analysis)
LESION_MIN_VOXELS = 5
LESION_SIZE_BINS_ML = ((0.5, "small"), (5.0, "medium"), (float("inf"), "large"))


class SpatialTransformer(torch.nn.Module):
    """Voxel-displacement warp as used by the challenge scorer (from
    voxelmorph); evaluation warps go through this, not through the training
    transforms, so the numbers match the official metrics."""

    def __init__(self, size: Tuple[int, int, int], mode: str = "bilinear") -> None:
        super().__init__()
        self.mode = mode
        vectors = [torch.arange(0, s) for s in size]
        grid = torch.stack(torch.meshgrid(*vectors, indexing="ij"))[None].float()
        self.register_buffer("grid", grid)

    def forward(self, src: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        new_locs = self.grid + flow
        shape = flow.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        new_locs = new_locs.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]
        return F.grid_sample(src, new_locs, align_corners=False, mode=self.mode)


def model_path_for(cfg: TrainingConfig, run: EvalRun) -> Path:
    run_name = TRAINING_RUNS[run.network]
    if not run_name:
        raise ValueError(f"no cluster run name set for {run.network} in TRAINING_RUNS")
    return (
        cfg.model_save_dir
        / f"{cfg.mlflow_experiment}_{run_name}_stagelvl3_{run.checkpoint}.pth"
    )


def create_model(
    cfg: TrainingConfig, model_path: Path, device: torch.device
) -> torch.nn.Module:
    model_lvl1 = None
    if cfg.use_lvl1:
        model_lvl1 = miccai2020_model_stage.Miccai2020_LDR_laplacian_unit_add_lvl1(
            in_channel=cfg.in_channel,
            n_classes=cfg.n_classes,
            start_channel=cfg.start_channel,
            is_train=True,
            imgshape=cfg.img_shape_4,
            range_flow=cfg.range_flow,
            cost_volume_mode=cfg.cost_volume_mode,
            cost_volume_radius=cfg.cost_volume_radius,
            cost_volume_dilation=cfg.cost_volume_dilation,
            cost_volume_feat_channels=cfg.cost_volume_feat_channels,
            cost_volume_out_channels=cfg.cost_volume_out_channels,
            n_resblocks=cfg.n_resblocks,
            resblock_expansion=cfg.resblock_expansion,
        ).to(device)
    model_lvl2 = miccai2020_model_stage.Miccai2020_LDR_laplacian_unit_add_lvl2(
        in_channel=cfg.in_channel,
        n_classes=cfg.n_classes,
        start_channel=cfg.start_channel,
        is_train=True,
        imgshape=cfg.img_shape_2,
        range_flow=cfg.range_flow,
        model_lvl1=model_lvl1,
        n_resblocks=cfg.n_resblocks,
        resblock_expansion=cfg.resblock_expansion,
    ).to(device)
    model = miccai2020_model_stage.Miccai2020_LDR_laplacian_unit_add_lvl3(
        in_channel=cfg.in_channel,
        n_classes=cfg.n_classes,
        start_channel=cfg.start_channel,
        is_train=True,
        imgshape=cfg.img_shape,
        range_flow=cfg.range_flow,
        model_lvl2=model_lvl2,
        n_resblocks=cfg.n_resblocks,
        resblock_expansion=cfg.resblock_expansion,
        use_seg_pet_head=cfg.use_seg_pet_head,
        seg_pet_head_channels=cfg.seg_pet_head_channels,
        use_bone_head=cfg.use_seg_bone_head,
        bone_head_channels=cfg.seg_bone_head_channels,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def split_pairs(cfg: TrainingConfig, split: str) -> List[Tuple[str, str, str]]:
    """(case_id, tp_moving, tp_fixed) for every follow-up -> baseline pair."""
    with open(cfg.split_path_journal) as f:
        case_ids = json.load(f)[split]
    case_timepoints = my_data.list_case_timepoints(cfg.data_dir)
    return my_data.build_registration_pairs(case_timepoints, case_ids=case_ids)


def load_label(
    label_dir: Path, case_id: str, channel: str, tp: str, device
) -> torch.Tensor:
    path = label_dir / f"PSMARegPSMA_{case_id}_{channel}_{tp}.nii.gz"
    arr = my_data.nib.load(str(path)).get_fdata().astype(np.int16)
    return torch.from_numpy(arr)[None, None].to(device).float()


def baseline_field_dir(external: object) -> Path:
    """Field directory of a baseline: the val-chosen config of a method in
    BASELINE_METHODS, or an explicit directory (used while tuning)."""
    if external in BASELINE_METHODS:
        method_dir = BASELINE_DIR / str(external)
        with open(method_dir / "chosen.json") as f:
            return method_dir / json.load(f)["config"] / "fields"
    return Path(external)


def baseline_field_path(field_dir: Path, case_id: str, tp_x: str, tp_y: str) -> Path:
    return field_dir / f"disp_{case_id}_{tp_x}_{tp_y}.nii.gz"


def load_external_unit_flow(
    path: Path, full_shape: Tuple[int, int, int], device: torch.device
) -> torch.Tensor:
    """A baseline's total displacement field -> total unit flow (1, 3, H, W, D).

    The field is stored channel-first, channel i along array axis i, in
    full-resolution voxels, on the full grid or a coarser one (ConvexAdam).
    It is converted with the (n - 1)/2 convention of transform_unit_flow_to_
    flow_cuda, so evaluate_pair hands the scorer exactly the stored field.
    Unit flow is resolution-independent, so a coarse field is upsampled after
    the conversion."""
    field = my_data.nib.load(str(path)).get_fdata().astype(np.float32)
    if field.shape[0] != 3:
        raise ValueError(f"{path}: expected a channel-first field, got {field.shape}")
    shape = field.shape[1:]
    # unit-flow layout: channel 0 along the LAST array axis
    unit = torch.from_numpy(field[::-1].copy())[None].to(device)
    for c, n in enumerate(reversed(full_shape)):
        unit[:, c] /= (n - 1) / 2.0
    if tuple(shape) != tuple(full_shape):
        unit = F.interpolate(unit, size=full_shape, mode="trilinear", align_corners=False)
    return unit


def register_pair(
    model: Optional[torch.nn.Module],
    run: EvalRun,
    cfg: TrainingConfig,
    case_id: str,
    tp_x: str,
    tp_y: str,
    transform: torch.nn.Module,
    transform_nearest: torch.nn.Module,
    grid_full: torch.Tensor,
    device: torch.device,
    history_csv: Path,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Affine prereg -> network -> optional IO.

    Returns the total unit flow (1, 3, H, W, D) that maps the ORIGINAL moving
    image onto the fixed one, and the moving / fixed images X, Y (1, 2, H, W, D).
    """
    pair = my_data.load_pair_to_dict(cfg.data_dir, case_id, tp_x, tp_y)
    X = torch.cat([pair["x_ct"], pair["x_pet"]], dim=0)[None].to(device).float()
    Y = torch.cat([pair["y_ct"], pair["y_pet"]], dim=0)[None].to(device).float()

    if run.external is not None:
        path = baseline_field_path(baseline_field_dir(run.external), case_id, tp_x, tp_y)
        return load_external_unit_flow(path, tuple(cfg.img_shape), device), X, Y
    if model is None and not run.affine:
        return torch.zeros(1, 3, *cfg.img_shape, device=device), X, Y

    image_dir = cfg.data_dir / "imagesTr"
    dvf = affine_reg.get_affine_dvf(
        case_id_x=case_id,
        case_id_y=case_id,
        tp_x=tp_x,
        tp_y=tp_y,
        fixed_ct_path=image_dir / f"PSMARegPSMA_{case_id}_0000_{tp_y}.nii.gz",
        moving_ct_path=image_dir / f"PSMARegPSMA_{case_id}_0000_{tp_x}.nii.gz",
        make_lowres_ants_image_fn=affine_reg.make_lowres_ants_image,
        preprocess_ct_fn=affine_reg.preprocess_ct,
        ants_affine_to_fullres_voxel_disp_fn=affine_reg.ants_affine_to_fullres_voxel_disp,
    )
    dvf_tensor = affine_reg.dvf_to_tensor(dvf, device)
    h, w, d = cfg.img_shape
    flow_affine = torch.stack(
        [
            dvf_tensor[:, 2] / (d / 2.0),
            dvf_tensor[:, 1] / (w / 2.0),
            dvf_tensor[:, 0] / (h / 2.0),
        ],
        dim=1,
    ).permute(0, 2, 3, 4, 1)

    if model is None:
        # affine prereg only
        F_X_Y = torch.zeros(1, 3, *cfg.img_shape, device=device)
    else:
        with torch.no_grad():
            X_affine = transform(X, flow_affine, grid_full)
            F_X_Y, _, _, _, _, _, _ = model(X_affine, Y)

    # compose affine (outer) with the network field (inner), so IO optimises
    # and the evaluation scores the TOTAL transform of the original moving image
    deform_grid = grid_full + F_X_Y.permute(0, 2, 3, 4, 1)
    affine_grid = (grid_full + flow_affine).permute(0, 4, 1, 2, 3)
    composed_grid = F.grid_sample(
        affine_grid,
        deform_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    ).permute(0, 2, 3, 4, 1)
    total_unit_flow = (composed_grid - grid_full).permute(0, 4, 1, 2, 3)

    if run.io is not None:
        io_cfg = copy.copy(cfg)
        for key, value in run.io.cfg_overrides.items():
            if not hasattr(io_cfg, key):
                raise ValueError(f"unknown config field {key}")
            setattr(io_cfg, key, value)
        # labels stay in the original moving frame; IO warps them with the
        # total field (single interpolation)
        x_lbl_ct = load_label(PRED_LABEL_DIR, case_id, "0000", tp_x, device)
        x_lbl_pet = load_label(PRED_LABEL_DIR, case_id, "0001", tp_x, device)
        y_lbl_ct = load_label(PRED_LABEL_DIR, case_id, "0000", tp_y, device)
        total_unit_flow = instance_opt.run_io(
            Y,
            total_unit_flow,
            X,
            x_lbl_ct,
            x_lbl_pet,
            y_lbl_ct,
            transform,
            transform_nearest,
            grid_full,
            io_cfg,
            device,
            include_pet=run.io.pet,
            include_rigidity=run.io.rigidity,
            history_csv=history_csv,
        )

    return total_unit_flow.detach(), X, Y


def relative_change(after: torch.Tensor, before: torch.Tensor) -> float:
    """Signed relative change (after - before) / before; NaN if before is 0."""
    before = float(before)
    return (float(after) - before) / before if before > 0 else float("nan")


def lesion_size_bin(n_voxels: int, voxel_ml: float) -> str:
    if n_voxels < LESION_MIN_VOXELS:
        return "tiny"
    volume_ml = n_voxels * voxel_ml
    return next(name for upper, name in LESION_SIZE_BINS_ML if volume_ml < upper)


def per_lesion_metrics(
    lesion_moving: torch.Tensor,
    organs_moving: torch.Tensor,
    suv_moving: torch.Tensor,
    suv_warped: torch.Tensor,
    disp_eval: torch.Tensor,
    warp_nearest: SpatialTransformer,
    voxel_ml: float,
) -> List[Dict[str, float]]:
    """One row per lesion (connected component) of the moving scan: its MTV and
    TLG after vs. before warping, size, and bone association.

    The component map is warped once with nearest neighbour (components are
    disjoint, so this equals warping each lesion mask on its own)."""
    components, n_lesions = ndimage.label(
        lesion_moving[0, 0].cpu().numpy() > 0, structure=LESION_STRUCTURE
    )
    if n_lesions == 0:
        return []
    components_t = torch.from_numpy(components)[None, None].to(lesion_moving.device)
    warped = warp_nearest(components_t.float(), disp_eval).round().long()[0, 0]
    components_t = components_t[0, 0]

    bone_values = torch.tensor(synthetic.BONE_LABEL_VALUES, device=organs_moving.device)
    on_bone = torch.isin(organs_moving[0, 0].round().long(), bone_values)
    suv_moving, suv_warped = suv_moving[0, 0], suv_warped[0, 0]

    rows = []
    for lesion_id in range(1, n_lesions + 1):
        mask_before = components_t == lesion_id
        mask_after = warped == lesion_id
        n_voxels = int(mask_before.sum())
        bone_fraction = float(on_bone[mask_before].float().mean())
        mtv_change = relative_change(mask_after.sum(), n_voxels)
        tlg_change = relative_change(
            suv_warped[mask_after].sum(), suv_moving[mask_before].sum()
        )
        rows.append(
            {
                "lesion_id": lesion_id,
                "lesion_voxels": n_voxels,
                "lesion_ml": n_voxels * voxel_ml,
                "size_bin": lesion_size_bin(n_voxels, voxel_ml),
                "bone_fraction": bone_fraction,
                "bone_associated": bone_fraction >= LESION_BONE_FRACTION,
                "mtv_change": mtv_change,
                "mtv_bias": abs(mtv_change),
                "tlg_change": tlg_change,
                "tlg_bias": abs(tlg_change),
            }
        )
    return rows


def evaluate_pair(
    total_unit_flow: torch.Tensor,
    Y: torch.Tensor,
    cfg: TrainingConfig,
    case_id: str,
    tp_x: str,
    tp_y: str,
    warp_nearest: SpatialTransformer,
    warp_linear: SpatialTransformer,
    device: torch.device,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """Metrics of one registered pair, all against the ground-truth labels.

    Accuracy: CT Dice and HD95 over the 117 CT labels (official definitions),
    before and after registration. Regularity: %NDV in the fixed body mask.
    Preservation: MTV and TLG of the moving scan's lesions after vs. before
    warping, signed and absolute. TLG uses the raw SUV image, not the network's
    clipped input; the lesion mask is warped nearest, the PET image linearly,
    as by the challenge scorer.
    """
    total_voxel = Functions.transform_unit_flow_to_flow_cuda(
        total_unit_flow.permute(0, 2, 3, 4, 1).clone()
    ).permute(0, 4, 1, 2, 3)
    # the scorer's transformer expects the components in the opposite order
    disp_eval = total_voxel.flip(1)

    # --- CT organ labels: Dice and HD95 ---
    seg_moving = load_label(GT_LABEL_DIR, case_id, "0000", tp_x, device)
    seg_fixed = load_label(GT_LABEL_DIR, case_id, "0000", tp_y, device)
    seg_warped = warp_nearest(seg_moving, disp_eval)
    moving_i = seg_moving[0, 0].round().long()
    fixed_i = seg_fixed[0, 0].round().long()
    warped_i = seg_warped[0, 0].round().long()

    fixed_nii = my_data.nib.load(
        str(GT_LABEL_DIR / f"PSMARegPSMA_{case_id}_0000_{tp_y}.nii.gz")
    )
    spacing_mm = tuple(float(z) for z in fixed_nii.header.get_zooms()[:3])
    moving_np = moving_i.cpu().numpy().astype(np.int16)
    fixed_np = fixed_i.cpu().numpy().astype(np.int16)
    warped_np = warped_i.cpu().numpy().astype(np.int16)

    # --- deformation regularity ---
    body_mask = Y[0, 0].detach().cpu().numpy() > 0

    # --- PET lesion preservation ---
    lesion_moving = (
        load_label(GT_LABEL_DIR, case_id, "0001", tp_x, device) == 1
    ).float()
    lesion_warped = (warp_nearest(lesion_moving, disp_eval) > 0.5).float()
    pet_path = cfg.data_dir / "imagesTr" / f"PSMARegPSMA_{case_id}_0001_{tp_x}.nii.gz"
    suv_moving = torch.from_numpy(
        my_data.nib.load(str(pet_path)).get_fdata().astype(np.float32)
    )[None, None].to(device)
    suv_warped = warp_linear(suv_moving, disp_eval)
    mtv_change = relative_change(lesion_warped.sum(), lesion_moving.sum())
    tlg_change = relative_change(
        (suv_warped * lesion_warped).sum(), (suv_moving * lesion_moving).sum()
    )
    voxel_ml = float(np.prod(spacing_mm)) / 1000.0
    lesion_rows = per_lesion_metrics(
        lesion_moving,
        seg_moving,
        suv_moving,
        suv_warped,
        disp_eval,
        warp_nearest,
        voxel_ml,
    )

    # per-label HD95 once; its mean over all labels is the official HD95, and
    # the bone / soft-tissue subsets feed bone_metrics
    hd95_per_label = bone_metrics.per_label_hd95(
        fixed_np, moving_np, warped_np, spacing_mm
    )
    bone = bone_metrics.bone_pair_metrics(
        total_unit_flow,
        disp_eval,
        fixed_i,
        warped_i,
        hd95_per_label,
        body_mask,
        spacing_mm,
    )

    pair_metrics = {
        "dice_ct_before": instance_opt.multilabel_dice(moving_i, fixed_i),
        "dice_ct": instance_opt.multilabel_dice(warped_i, fixed_i),
        "hd95_before": float(
            hd95_official.compute_average_ct_label_hd95(
                fixed_np, moving_np, moving_np, spacing_mm
            )
        ),
        "hd95": bone_metrics.mean_over(hd95_per_label, range(1, 118)),
        "ndv_percent": bone.pop("ndv_body_percent"),
        **bone,
        "lesion_voxels": float(lesion_moving.sum()),
        "mtv_change": mtv_change,
        "mtv_bias": abs(mtv_change),
        "tlg_change": tlg_change,
        "tlg_bias": abs(tlg_change),
        "n_lesions": len(lesion_rows),
    }
    return pair_metrics, lesion_rows


def evaluate_run(
    run_id: str,
    split: str,
    device: torch.device,
    run: Optional[EvalRun] = None,
    out_dir: Optional[Path] = None,
) -> Path:
    """Evaluate one run on every pair of the split; returns the pair CSV.
    `run` / `out_dir` default to EVAL_RUNS[run_id] / OUT_DIR (the tuning
    scripts pass their own)."""
    out_csv = (out_dir or OUT_DIR) / split / f"{run_id}.csv"
    if SKIP_EXISTING and out_csv.exists():
        print(f"[{run_id}] {out_csv} exists, skipping")
        return out_csv

    run = run or EVAL_RUNS[run_id]
    cfg = TrainingConfig()
    model = None
    if run.network is not None:
        for key, value in TRAINING_CFG_OVERRIDES[run.network].items():
            setattr(cfg, key, value)
        model_path = model_path_for(cfg, run)
        model = create_model(cfg, model_path, device)
        source = model_path.name
    elif run.external is not None:
        source = str(baseline_field_dir(run.external))
    else:
        source = "affine" if run.affine else "identity"

    transform = miccai2020_model_stage.SpatialTransform_unit().to(device)
    transform_nearest = miccai2020_model_stage.SpatialTransformNearest_unit().to(device)
    grid = Functions.generate_grid_unit(cfg.img_shape)
    grid_full = torch.from_numpy(grid[None]).to(device).float()
    warp_nearest = SpatialTransformer(cfg.img_shape, mode="nearest").to(device)
    warp_linear = SpatialTransformer(cfg.img_shape, mode="bilinear").to(device)
    if not run.warp:
        if run.network is not None or run.external is not None or run.affine:
            raise ValueError("warp=False is only meaningful for the identity")
        warp_nearest = warp_linear = lambda src, flow: src

    rows = []
    lesion_rows = []
    for case_id, tp_x, tp_y in tqdm.tqdm(split_pairs(cfg, split), desc=run_id):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_start = time.perf_counter()
        total_unit_flow, _, Y = register_pair(
            model,
            run,
            cfg,
            case_id,
            tp_x,
            tp_y,
            transform,
            transform_nearest,
            grid_full,
            device,
            history_csv=OUT_DIR
            / split
            / "io_history"
            / run_id
            / f"{case_id}_{tp_x}.csv",
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        runtime_s = time.perf_counter() - t_start

        metrics, pair_lesion_rows = evaluate_pair(
            total_unit_flow,
            Y,
            cfg,
            case_id,
            tp_x,
            tp_y,
            warp_nearest,
            warp_linear,
            device,
        )
        pair_keys = {
            "run_id": run_id,
            "case_id": case_id,
            "tp_moving": tp_x,
            "tp_fixed": tp_y,
        }
        rows.append(
            {
                **pair_keys,
                "model": source,
                "runtime_s": runtime_s,
                **metrics,
            }
        )
        lesion_rows += [{**pair_keys, **row} for row in pair_lesion_rows]
        tqdm.tqdm.write(
            f"[{run_id}] {case_id} {tp_x}->{tp_y}: "
            + " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(
        f"[{run_id}] wrote {out_csv}\n{df.drop(columns=['model']).mean(numeric_only=True)}"
    )

    # one row per lesion; "tiny" lesions are kept here and excluded in the
    # analysis
    lesions_csv = out_csv.with_name(f"{run_id}_lesions.csv")
    df_lesions = pd.DataFrame(lesion_rows)
    df_lesions.to_csv(lesions_csv, index=False)
    if not df_lesions.empty:
        kept = df_lesions[df_lesions["size_bin"] != "tiny"]
        summary = kept.groupby(["bone_associated", "size_bin"])["mtv_bias"].describe(
            percentiles=[0.5, 0.95]
        )
        print(f"[{run_id}] wrote {lesions_csv}\nper-lesion |MTV change|:\n{summary}")
    return out_csv


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for run_id in RUN_IDS:
        evaluate_run(run_id, SPLIT, device)


if __name__ == "__main__":
    main()
