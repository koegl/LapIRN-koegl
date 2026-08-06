import contextlib
import json
import math
import os
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Tuple

import mlflow
import my_data
import numpy as np
import torch
import tqdm
from config import TrainingConfig
from miccai2020_model_stage import (
    SpatialTransform_unit,
)
from torch.utils import checkpoint as torch_checkpoint
from torch.utils import data as torch_data

import wandb

_ACTIVE_RUN_NAME: str | None = None
_WANDB_RUN = None


def stop_flag_path(save_dir: Path, level: int) -> Path:
    return save_dir / "stop_flags" / f"stop_{7777}_lvl{level}.flag"

    job_id = get_slurm_job_id()
    array_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if array_id is not None:
        job_id = f"{job_id}_{array_id}"
    return save_dir / "stop_flags" / f"stop_{job_id}_lvl{level}.flag"


def check_stop_flag(save_dir: Path, level: int) -> bool:
    p = stop_flag_path(save_dir, level)
    if not p.exists():
        return False
    p.unlink(missing_ok=True)
    return True


def _enabled_logger_backends(config: TrainingConfig) -> set[str]:
    backend = config.logger_backend.lower()
    if backend == "none":
        return set()
    if backend == "both":
        return {"mlflow", "wandb"}
    if backend not in {"mlflow", "wandb"}:
        raise ValueError(
            f"Unknown logger_backend={config.logger_backend!r}; "
            'expected "mlflow", "wandb", "both", or "none".'
        )
    return {backend}


def get_slurm_job_id() -> str:
    return os.environ.get("SLURM_JOB_ID", "local")


def get_run_name_with_job_id(base_name: str) -> str:
    job_id = get_slurm_job_id()
    if base_name.endswith(f"-{job_id}"):
        return base_name
    base_name_no_id = base_name.rsplit("-", 1)[0]
    return f"{base_name_no_id}-{job_id}"


@contextlib.contextmanager
def start_logging_run(config: TrainingConfig) -> Iterator[None]:
    """Start the configured experiment logger(s)."""
    global _ACTIVE_RUN_NAME, _WANDB_RUN

    backends = _enabled_logger_backends(config)
    previous_run_name = _ACTIVE_RUN_NAME
    previous_wandb_run = _WANDB_RUN
    _ACTIVE_RUN_NAME = None
    _WANDB_RUN = None

    with ExitStack() as stack:
        if not backends:
            _ACTIVE_RUN_NAME = f"{config.mlflow_experiment}-{get_slurm_job_id()}"

        if "mlflow" in backends:
            mlflow.set_tracking_uri(config.mlflow_tracking_uri)
            mlflow.set_experiment(config.mlflow_experiment)
            stack.enter_context(mlflow.start_run())
            add_jobid_to_mlflow_run()
            _ACTIVE_RUN_NAME = get_mlflow_run_name()

        if "wandb" in backends:
            if config.wandb_mode is not None:
                os.environ["WANDB_MODE"] = config.wandb_mode

            run_name = (
                _ACTIVE_RUN_NAME or f"{config.wandb_project}-{get_slurm_job_id()}"
            )
            _WANDB_RUN = wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=run_name,
                group=get_slurm_job_id(),
            )
            _WANDB_RUN.define_metric("global_step")
            _WANDB_RUN.define_metric("*", step_metric="global_step")
            _ACTIVE_RUN_NAME = _WANDB_RUN.name

        try:
            yield
        finally:
            if _WANDB_RUN is not None:
                _WANDB_RUN.finish()
            _ACTIVE_RUN_NAME = previous_run_name
            _WANDB_RUN = previous_wandb_run


def log_params(params: Dict[str, Any]) -> None:
    if mlflow.active_run() is not None:
        mlflow.log_params(params)
    if _WANDB_RUN is not None:
        _WANDB_RUN.config.update(params, allow_val_change=True)


def log_text(text: str, artifact_file: str) -> None:
    if mlflow.active_run() is not None:
        mlflow.log_text(text, artifact_file=artifact_file)
    if _WANDB_RUN is not None:
        import wandb

        _WANDB_RUN.log({artifact_file: wandb.Html(f"<pre>{text}</pre>")})


def log_config(params: Dict[str, Any]) -> None:
    log_params(params)
    log_text(json.dumps(params, indent=2), artifact_file="config.json")


def log_metrics(metrics: Dict[str, float], step: int | None = None) -> None:
    if mlflow.active_run() is not None:
        mlflow.log_metrics(metrics, step=step)
    if _WANDB_RUN is not None:
        wandb_metrics = dict(metrics)
        if step is not None:
            wandb_metrics["global_step"] = step
        _WANDB_RUN.log(wandb_metrics)


def create_datasets(
    config: TrainingConfig,
) -> Tuple[
    torch_data.ConcatDataset,
    my_data.PSMARegDataset,
    my_data.SyntheticSourceDataset | None,
    my_data.TubingenDataset | None,
    my_data.NLSTDataset | None,
    my_data.AbdomenDataset | None,
    my_data.PSMARegDataset,
    my_data.TubingenDataset | None,
    my_data.NLSTDataset | None,
    my_data.AbdomenDataset | None,
    dict,
]:

    train_ids, val_ids = my_data.get_train_val_split(
        data_dir=config.data_dir,
        split_path=config.split_path,
        val_fraction=config.val_fraction,
        tubingen=False,
        nlst=False,
        abdomen=False,
    )
    train_ids_tubingen, val_ids_tubingen = my_data.get_train_val_split(
        data_dir=config.data_dir,
        split_path=config.split_path,
        val_fraction=config.val_fraction,
        tubingen=True,
        nlst=False,
        abdomen=False,
    )
    train_ids_nlst, val_ids_nlst = my_data.get_train_val_split(
        data_dir=config.data_dir,
        split_path=config.split_path,
        val_fraction=config.val_fraction,
        tubingen=False,
        nlst=True,
        abdomen=False,
    )
    train_ids_abdomen, val_ids_abdomen = my_data.get_train_val_split(
        data_dir=config.data_dir,
        split_path=config.split_path,
        val_fraction=config.val_fraction,
        tubingen=False,
        nlst=False,
        abdomen=True,
    )
    synth_ids = [
        c
        for c, _ in my_data.list_single_session_sources(
            config.data_dir,
            exclude_case_ids=train_ids
            + val_ids
            + train_ids_tubingen
            + val_ids_tubingen
            + train_ids_nlst
            + val_ids_nlst
            + train_ids_abdomen
            + val_ids_abdomen,
        )
    ]

    config_to_log = config.to_mlflow_params()
    config_to_log["train_indices"] = train_ids
    config_to_log["train_indices_synthetic"] = synth_ids
    config_to_log["train_indices_tubingen"] = train_ids_tubingen
    config_to_log["train_indices_nlst"] = train_ids_nlst
    config_to_log["train_indices_abdomen"] = train_ids_abdomen

    config_to_log["val_indices"] = val_ids
    config_to_log["val_indices_tubingen"] = val_ids_tubingen
    config_to_log["val_indices_nlst"] = val_ids_nlst
    config_to_log["val_indices_abdomen"] = val_ids_abdomen

    val_dataset_tubingen = None
    val_dataset_nlst = None
    val_dataset_abdomen = None
    train_dataset_synthetic = None
    train_dataset_tubingen = None
    train_dataset_nlst = None
    train_dataset_abdomen = None

    val_dataset = my_data.PSMARegDataset(
        case_ids=val_ids,
        cfg=config,
        augment=False,
        use_cache=config.use_cache_valid,
        include_intermediate_pairs=False,
        num_workers=config.num_workers,
    )

    train_dataset = my_data.PSMARegDataset(
        case_ids=train_ids,
        cfg=config,
        augment=config.augment,
        use_cache=config.use_cache_train_real,
        include_intermediate_pairs=True,
        num_workers=config.num_workers,
    )
    all_train_datasets = [train_dataset]

    if config.use_tubingen:
        train_dataset_tubingen = my_data.TubingenDataset(
            case_ids=train_ids_tubingen,
            cfg=config,
            augment=config.augment,
            use_cache=config.use_cache_train_real,
            num_workers=config.num_workers,
        )
        all_train_datasets.append(train_dataset_tubingen)

        val_dataset_tubingen = my_data.TubingenDataset(
            case_ids=val_ids_tubingen,
            cfg=config,
            augment=False,
            use_cache=config.use_cache_valid,
            num_workers=config.num_workers,
        )

    if config.use_nlst:
        train_dataset_nlst = my_data.NLSTDataset(
            case_ids=train_ids_nlst,
            cfg=config,
            augment=config.augment,
            use_cache=config.use_cache_train_real,
            num_workers=config.num_workers,
        )
        all_train_datasets.append(train_dataset_nlst)

        val_dataset_nlst = my_data.NLSTDataset(
            case_ids=val_ids_nlst,
            cfg=config,
            augment=False,
            use_cache=config.use_cache_valid,
            num_workers=config.num_workers,
        )

    if config.use_abdomen is not None:
        train_dataset_abdomen = my_data.AbdomenDataset(
            case_ids=train_ids_abdomen,
            cfg=config,
            augment=config.augment,
            use_cache=config.use_cache_train_real,
            num_workers=config.num_workers,
        )
        all_train_datasets.append(train_dataset_abdomen)

        val_dataset_abdomen = my_data.AbdomenDataset(
            case_ids=val_ids_abdomen,
            cfg=config,
            augment=False,
            use_cache=config.use_cache_valid,
            num_workers=config.num_workers,
        )

    if config.use_synthetic:
        train_dataset_synthetic = my_data.SyntheticSourceDataset(
            cfg=config,
            source_ids=synth_ids,
            repeat=config.synthetic_repeat,
            use_cache=config.use_cache_train_synthetic,
            num_workers=config.num_workers,
            augment=config.augment,
        )
        all_train_datasets.append(train_dataset_synthetic)

    train_combined = torch_data.ConcatDataset(all_train_datasets)

    return (
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
    )


def warmup_lr_factor(global_step: int, warmup_steps: int) -> float:
    """Linear LR warmup factor: ramps 0 -> 1 over the first `warmup_steps`
    training steps, then stays at 1. `warmup_steps <= 0` disables warmup."""
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, (global_step + 1) / warmup_steps)


def apply_warmup_lr(
    optimizer: torch.optim.Optimizer,
    base_lr: float,
    global_step: int,
    warmup_steps: int,
) -> float:
    """Set the optimizer LR to `base_lr` scaled by the linear warmup factor.
    Returns the LR applied (for logging)."""
    lr = base_lr * warmup_lr_factor(global_step, warmup_steps)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def add_jobid_to_mlflow_run() -> None:
    job_id = os.environ.get("SLURM_JOB_ID", "local")

    auto_name = str(mlflow.active_run().info.run_name)
    new_name = get_run_name_with_job_id(auto_name)

    mlflow.set_tag("mlflow.runName", new_name)
    mlflow.set_tag("slurm_job_id", job_id)


def get_mlflow_run_name() -> str:
    run_id = mlflow.active_run().info.run_id
    run_name = mlflow.get_run(run_id).info.run_name

    if run_name is None:
        raise ValueError("MLflow run name is None. Please ensure an active run exists.")

    return run_name


def get_run_name() -> str:
    if _ACTIVE_RUN_NAME is not None:
        return _ACTIVE_RUN_NAME
    if mlflow.active_run() is not None:
        return get_mlflow_run_name()
    raise ValueError("No active logger run. Please ensure a logging run exists.")


def optimizer_step_with_guard(
    loss: torch.Tensor,
    loss_scaled: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    is_step: bool,
    global_step: int,
    level: int,
) -> None:

    if torch.isfinite(loss):
        loss_scaled.backward()

        if is_step:
            # clip_grad_norm_ returns the PRE-clip norm and already scales the
            # applied gradients down to max_norm, so a large pre-clip norm is
            # safe to step on. Only skip on non-finite gradients (NaN/Inf);
            # skipping on magnitude here deadlocks (skipped step -> weights
            # unchanged -> same large norm -> skipped forever).
            total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=5.0
            )
            log_metrics({f"lvl{level}/grad_norm": total_norm.item()}, step=global_step)
            if not torch.isfinite(total_norm):
                tqdm.tqdm.write(
                    f"[lvl{level}] step {global_step}: non-finite grad_norm "
                    f"loss={loss.item():.4f} (skipped)"
                )
                optimizer.zero_grad()
            else:
                optimizer.step()
                optimizer.zero_grad()
    else:
        tqdm.tqdm.write(f"[lvl{level}] step {global_step}: non-finite loss (skipped)")
        optimizer.zero_grad()


def cycle(loader: torch.utils.data.DataLoader):
    while True:
        for batch in loader:
            yield batch


@contextlib.contextmanager
def track_peak_memory(label: str = ""):
    torch.cuda.reset_peak_memory_stats()
    try:
        yield
    finally:
        allocated = torch.cuda.max_memory_allocated() / 1024**3
        reserved = torch.cuda.max_memory_reserved() / 1024**3
        tag = f"[{label}] " if label else ""
        print(
            f"{tag}Peak allocated: {allocated:.2f} GiB | Peak reserved: {reserved:.2f} GiB"
        )


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    stage: str,
    config: TrainingConfig,
    lossall: np.ndarray,
) -> None:
    """Save a rolling resume checkpoint every config.checkpoint_interval epochs.

    Writes a single "latest" checkpoint per stage, overwritten each time, so
    disk usage stays flat over a long run. It holds model weights, optimizer
    state, current epoch and loss history -- everything needed to resume the
    stage after a wall-time kill. The final per-stage weight file (used to
    gate the next stage) is still saved separately by the train_lvlX function.

    Args:
        model: Model being trained.
        optimizer: Optimizer whose state is saved for resuming.
        epoch: Current epoch index.
        stage: Stage tag, one of "lvl1", "lvl2", "lvl3".
        config: Training configuration; must define checkpoint_interval.
        lossall: Running loss-history array to persist alongside weights.
    """
    if epoch % config.checkpoint_interval != 0:
        return

    config.model_save_dir.mkdir(parents=True, exist_ok=True)
    latest_path = (
        config.model_save_dir / f"{config.mlflow_experiment}_{stage}_latest.pth"
    )
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lossall": lossall,
        },
        latest_path,
    )
    tqdm.tqdm.write(f"[{stage}] checkpoint @ epoch {epoch} -> {latest_path.name}")


def downsample_label(label: torch.Tensor, scale_factor: float) -> torch.Tensor:
    """Downsample integer label map with nearest-neighbour (B, 1, D, H, W)."""
    return torch.nn.functional.interpolate(
        label.float(), scale_factor=scale_factor, mode="nearest"
    ).long()


def soft_dice_loss_binary(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """Soft dice for a single binary channel (B, 1, D, H, W)."""
    intersection = (pred * target).sum()
    cardinality = pred.sum() + target.sum()
    return 1.0 - (2.0 * intersection + eps) / (cardinality + eps)


def soft_dice_loss_multiclass(
    pred_onehot: torch.Tensor,
    target_onehot: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Mean foreground soft dice over classes present in target.

    Args:
        pred_onehot: (B, C, D, H, W) float, warped one-hot.
        target_onehot: (B, C, D, H, W) float, fixed one-hot.
        eps: Smoothing term.

    Returns:
        Scalar 1 - mean_foreground_dice.
    """
    # sum over spatial dims and batch: (C,)
    dims = (0, 2, 3, 4)
    intersection = (pred_onehot * target_onehot).sum(dim=dims)
    cardinality = pred_onehot.sum(dim=dims) + target_onehot.sum(dim=dims)
    dice_per_class = (2.0 * intersection + eps) / (cardinality + eps)
    # exclude background (class 0)
    return 1.0 - dice_per_class[1:].mean()


def to_one_hot(label: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert integer label volume to one-hot float tensor.

    Args:
        label: (B, 1, D, H, W) integer tensor.
        num_classes: Number of classes including background.

    Returns:
        (B, C, D, H, W) float tensor.
    """
    B, _, D, H, W = label.shape
    one_hot = torch.zeros(
        B, num_classes, D, H, W, device=label.device, dtype=torch.float32
    )
    return one_hot.scatter_(1, label.long(), 1.0)


def dice_loss_with_grad(
    moving_label: torch.Tensor,
    fixed_label: torch.Tensor,
    disp: torch.Tensor,
    grid: torch.Tensor,
    transform: SpatialTransform_unit,
    eps: float = 1e-5,
    return_each: bool = False,
    class_weights: torch.Tensor | None = None,
    chunk_size: int = 16,
) -> torch.Tensor | None:
    """Per-class soft dice loss with gradients flowing through disp.

    Builds one-hot only for classes present in *both* labels, warps the moving
    one-hot channels with bilinear interpolation, then computes dice per class
    and averages. Robust to variable class sets across subjects.

    Memory note: the sampling grid `grid + flow` is identical for every class,
    so it is built once and shared. Warping one class at a time via
    SpatialTransform_unit would instead rebuild it per class and keep every
    full-resolution copy alive for backward (~122 MiB each at 192x192x288,
    i.e. many GB for a ~117-label volume). Classes are processed in chunks of
    `chunk_size` channels so the one-hot tensors stay bounded too.

    Args:
        moving_label: (B, 1, D, H, W) integer tensor, moving labels.
        fixed_label: (B, 1, D, H, W) integer tensor, fixed labels.
        disp: (B, 3, D, H, W) displacement field (gradients flow through this).
        grid: level grid for SpatialTransform_unit.
        transform: kept for API compatibility; the bilinear warp is inlined
            here so the sampling grid can be hoisted out of the class loop.
        eps: Smoothing term.
        return_each: return the per-class losses instead of their mean.
        class_weights: optional per-label weights, indexed by integer label
            value. Renormalized over the classes actually used so the loss
            scale stays comparable across subjects.
        chunk_size: number of label channels warped per grid_sample call.

    Returns:
        Scalar 1 - mean_foreground_dice, or None if no class is usable.
    """

    classes = fixed_label.unique()
    classes = classes[classes != 0]  # exclude background

    if classes.numel() == 0:
        return None

    # Drop classes absent from the moving image: a class present in only one
    # image (e.g. cropped out of the moving by augmentation) yields dice~0 with
    # no usable registration gradient and only biases the mean. Testing this
    # with unique() is far cheaper than materializing a one-hot per class.
    classes = classes[torch.isin(classes, moving_label.unique())]
    if classes.numel() == 0:
        return None

    flow = disp.permute(0, 2, 3, 4, 1)  # (B, D, H, W, 3)
    sample_grid = grid + flow  # built once, reused by every chunk

    dims = (0, 2, 3, 4)  # sum over batch and space, keep the class channel
    dice_scores = []
    for start in range(0, classes.numel(), chunk_size):
        chunk = classes[start : start + chunk_size].view(1, -1, 1, 1, 1)

        moving_c = (moving_label == chunk).float()  # (B, C, D, H, W)
        fixed_c = (fixed_label == chunk).float()

        warped_c = torch.nn.functional.grid_sample(
            moving_c,
            sample_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )

        intersection = (warped_c * fixed_c).sum(dim=dims)
        cardinality = warped_c.sum(dim=dims) + fixed_c.sum(dim=dims)
        dice_scores.append((2.0 * intersection + eps) / (cardinality + eps))

    dice_stack = torch.cat(dice_scores)

    if class_weights is not None:
        w = class_weights[classes.round().long()]
        w = w / (w.mean() + eps)
        return 1.0 - (w * dice_stack).sum() / w.sum()

    if return_each:
        return 1.0 - dice_stack
    else:
        return 1.0 - dice_stack.mean()


def _mask_bbox(mask: torch.Tensor) -> Tuple[int, int, int, int, int, int] | None:
    """Tight half-open bbox (d0, d1, h0, h1, w0, w1) of a (B, 1, D, H, W) bool
    mask, or None if the mask is empty. Uses per-axis `any` reductions rather
    than nonzero() so nothing proportional to the number of set voxels is
    materialized."""
    m = mask.reshape(-1, *mask.shape[-3:]).any(0)  # (D, H, W)

    dh = m.any(2)  # (D, H)
    occ_d = dh.any(1)  # (D,)
    occ_h = dh.any(0)  # (H,)
    occ_w = m.any(1).any(0)  # (W,)

    nz_d = occ_d.nonzero().flatten()
    if nz_d.numel() == 0:
        return None
    nz_h = occ_h.nonzero().flatten()
    nz_w = occ_w.nonzero().flatten()

    # one device->host sync for all six bounds instead of six
    bounds = torch.stack(
        [nz_d[0], nz_d[-1] + 1, nz_h[0], nz_h[-1] + 1, nz_w[0], nz_w[-1] + 1]
    )
    d0, d1, h0, h1, w0, w1 = bounds.tolist()
    return d0, d1, h0, h1, w0, w1


def _class_dice_cropped(
    sample_grid: torch.Tensor,
    moving_label: torch.Tensor,
    fixed_label: torch.Tensor,
    class_value: float,
    box: Tuple[int, int, int, int, int, int],
    eps: float,
) -> torch.Tensor:
    """Soft dice for one class inside `box`, as a 0-dim tensor.

    Split out so it can be wrapped in `torch.utils.checkpoint`: every tensor it
    allocates is then dropped at the end of the forward pass and recomputed
    during backward, leaving only `sample_grid` (which the caller shares across
    classes anyway) alive in between.
    """
    d0, d1, h0, h1, w0, w1 = box
    shape_d, shape_h, shape_w = moving_label.shape[-3:]
    n_d, n_h, n_w = d1 - d0, h1 - h0, w1 - w0

    moving_c = (moving_label[:, :, d0:d1, h0:h1, w0:w1] == class_value).float()
    fixed_c = (fixed_label[:, :, d0:d1, h0:h1, w0:w1] == class_value).float()

    # remap the full-volume normalized coordinates into the crop's own
    # normalized frame: u' = u * (N / n) + ((N - 2 * o) / n - 1). Components are
    # ordered (x, y, z) = (W, H, D).
    scale = torch.tensor(
        [shape_w / n_w, shape_h / n_h, shape_d / n_d],
        device=sample_grid.device,
        dtype=sample_grid.dtype,
    )
    offset = torch.tensor(
        [
            (shape_w - 2 * w0) / n_w - 1.0,
            (shape_h - 2 * h0) / n_h - 1.0,
            (shape_d - 2 * d0) / n_d - 1.0,
        ],
        device=sample_grid.device,
        dtype=sample_grid.dtype,
    )
    grid_c = sample_grid[:, d0:d1, h0:h1, w0:w1, :] * scale + offset

    warped_c = torch.nn.functional.grid_sample(
        moving_c,
        grid_c,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )

    intersection = (warped_c * fixed_c).sum()
    cardinality = warped_c.sum() + fixed_c.sum()
    return (2.0 * intersection + eps) / (cardinality + eps)


def dice_loss_with_grad_bbox(
    moving_label: torch.Tensor,
    fixed_label: torch.Tensor,
    disp: torch.Tensor,
    grid: torch.Tensor,
    transform: SpatialTransform_unit,
    eps: float = 1e-5,
    return_each: bool = False,
    class_weights: torch.Tensor | None = None,
    extra_margin: int = 2,
    max_disp_vox: Tuple[float, float, float] | None = None,
    use_checkpoint: bool = False,
) -> torch.Tensor | None:
    """Memory-lean drop-in for `dice_loss_with_grad`, cropped per class.

    Same loss, computed one class at a time inside the union bounding box of
    (moving == c) and (fixed == c), dilated by the largest displacement in the
    field plus `extra_margin`. Most labels occupy well under 5% of the volume,
    so the tensors kept alive for backward shrink by roughly the same factor.

    Why the crop is exact, not an approximation. Let B be the dilated box and
    `maxd` the per-axis bound on |displacement| in voxels, so B contains
    bbox(moving == c) dilated by maxd:

    * warped(p) for p outside B is 0. It could only be non-zero if p + d(p)
      landed in bbox(moving == c), which would put p within maxd of that box,
      i.e. inside B. So nothing is missed from `warped.sum()`.
    * for p inside B whose source p + d(p) falls outside B, that source is more
      than maxd away from bbox(moving == c), so the true value is 0; the
      `padding_mode="border"` clamp returns the value on B's boundary shell,
      which is background for this class because the box was dilated by at
      least one voxel. So the clamp returns 0 too.
    * bbox(fixed == c) is contained in B by construction.

    All three sums therefore match the uncropped computation exactly, up to
    floating-point summation order. The bbox itself is derived from the integer
    labels under no_grad, so it is a constant and gradients flow only through
    the (cropped, coordinate-remapped) sampling grid.

    Args:
        moving_label: (B, 1, D, H, W) integer tensor, moving labels.
        fixed_label: (B, 1, D, H, W) integer tensor, fixed labels.
        disp: (B, 3, D, H, W) displacement field (gradients flow through this).
        grid: level grid for SpatialTransform_unit.
        transform: kept for API compatibility; the bilinear warp is inlined.
        eps: Smoothing term.
        return_each: return the per-class losses instead of their mean.
        class_weights: optional per-label weights, indexed by integer label
            value. Renormalized over the classes actually used.
        extra_margin: voxels of slack added on top of the displacement bound.
            Must be >= 1 for the border-clamp argument above to hold.
        max_disp_vox: per-axis (D, H, W) bound on |displacement| in voxels. If
            None it is derived from `disp`, which costs one reduction over the
            field. Pass it if you already know a bound and want to skip that.
        use_checkpoint: recompute each class's crop during backward instead of
            keeping it alive, at the cost of one extra (cropped) forward pass.

    Measured on a real 114-label pair at 192x192x288 (Code/check_dice_bbox.py),
    memory retained from the end of the forward until backward:

        dice_loss_with_grad   9357 MiB    1.0x
        this, crop only        540 MiB   17.3x
        this, + checkpoint     122 MiB   76.7x

    and the crop is also ~4x faster in the forward, because it does ~4x less
    work: the per-class boxes sum to ~2.5 volumes rather than 114. What autograd
    retains per grid_sample call is (2C + 3)|box| - the one-hots for C classes
    plus the sampling grid - which is why the crop cannot reach the checkpointed
    figure on its own: cropping turns the single shared grid into a per-class
    one and pays 3|box| for it.

    Returns:
        Scalar 1 - mean_foreground_dice, or None if no class is usable.
    """

    classes = fixed_label.unique()
    classes = classes[classes != 0]  # exclude background

    if classes.numel() == 0:
        return None

    classes = classes[torch.isin(classes, moving_label.unique())]
    if classes.numel() == 0:
        return None

    shape_d, shape_h, shape_w = moving_label.shape[-3:]
    flow = disp.permute(0, 2, 3, 4, 1)  # (B, D, H, W, 3)
    sample_grid = grid + flow  # built once, sliced per class

    # grid_sample's last-dim components are ordered (x, y, z) = (W, H, D), so
    # component k indexes spatial axis 2 - k. Displacement is in normalized
    # units where a span of 2 covers N voxels, hence the N / 2 factor.
    if max_disp_vox is None:
        with torch.no_grad():
            amax = disp.detach().abs().amax(dim=(0, 2, 3, 4))  # (3,) -> (W, H, D)
            max_w, max_h, max_d = (amax * 0.5).tolist()
            max_d *= shape_d
            max_h *= shape_h
            max_w *= shape_w
    else:
        max_d, max_h, max_w = max_disp_vox

    margin = max(extra_margin, 1)
    pad_d = int(math.ceil(max_d)) + margin
    pad_h = int(math.ceil(max_h)) + margin
    pad_w = int(math.ceil(max_w)) + margin

    dice_scores = []
    used_classes = []
    for c in classes.tolist():
        with torch.no_grad():
            occupied = (moving_label == c) | (fixed_label == c)
            box = _mask_bbox(occupied)
        if box is None:
            continue

        d0, d1, h0, h1, w0, w1 = box
        box = (
            max(0, d0 - pad_d),
            min(shape_d, d1 + pad_d),
            max(0, h0 - pad_h),
            min(shape_h, h1 + pad_h),
            max(0, w0 - pad_w),
            min(shape_w, w1 + pad_w),
        )

        args = (sample_grid, moving_label, fixed_label, c, box, eps)
        if use_checkpoint:
            score = torch_checkpoint.checkpoint(
                _class_dice_cropped, *args, use_reentrant=False
            )
        else:
            score = _class_dice_cropped(*args)

        dice_scores.append(score)
        used_classes.append(c)

    if not dice_scores:
        return None

    dice_stack = torch.stack(dice_scores)

    if class_weights is not None:
        idx = torch.tensor(used_classes, device=dice_stack.device).round().long()
        w = class_weights[idx]
        w = w / (w.mean() + eps)
        return 1.0 - (w * dice_stack).sum() / w.sum()

    if return_each:
        return 1.0 - dice_stack
    else:
        return 1.0 - dice_stack.mean()


def mtv_bias_loss(
    warped_pet_mask: torch.Tensor,
    moving_pet_mask: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """MTV bias: relative volume change of lesion mask after warping.

    Args:
        warped_pet_mask: (B, 1, D, H, W) float, warped moving lesion mask.
        moving_pet_mask: (B, 1, D, H, W) float, original moving lesion mask.
        eps: Smoothing term.

    Returns:
        Scalar absolute relative volume bias.
    """
    mtv_moving = moving_pet_mask.sum()
    mtv_warped = warped_pet_mask.sum()
    return torch.abs(mtv_warped - mtv_moving) / (mtv_moving + eps)


def tlg_bias_loss(
    warped_pet_image: torch.Tensor,
    warped_pet_mask: torch.Tensor,
    moving_pet_image: torch.Tensor,
    moving_pet_mask: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """TLG bias: relative change in lesion-weighted PET intensity after warping.

    Args:
        warped_pet_image: (B, 1, D, H, W) float, warped moving PET image.
        warped_pet_mask: (B, 1, D, H, W) float, warped moving lesion mask.
        moving_pet_image: (B, 1, D, H, W) float, original moving PET image.
        moving_pet_mask: (B, 1, D, H, W) float, original moving lesion mask.
        eps: Smoothing term.

    Returns:
        Scalar absolute relative TLG bias.
    """
    tlg_moving = (moving_pet_image * moving_pet_mask).sum()
    tlg_warped = (warped_pet_image * warped_pet_mask).sum()
    return torch.abs(tlg_warped - tlg_moving) / (tlg_moving + eps)


def masked_jac_det_loss(
    jac_det: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Penalise deviation of det(J) from 1 inside mask.

    Args:
        jac_det: (B, 1, D, H, W) Jacobian determinant field.
        mask: (B, 1, D, H, W) binary float mask (tumor region).
        eps: Smoothing term.

    Returns:
        Scalar mean squared deviation from 1 inside mask.
    """
    masked_det = jac_det * mask
    target = mask  # det(J)=1 inside mask
    return ((masked_det - target) ** 2).sum() / (mask.sum() + eps)


def orthonormality_loss(
    jac: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Penalise deviation of the Jacobian from orthonormality inside mask.

    Rigid motions have an orthonormal Jacobian (J^T J = I), which forbids local
    shear and anisotropic scaling. This penalises ||J^T J - I||_F^2 per voxel.
    Consumes the interior Jacobian returned by ``jacobian.jacobian_matrix``;
    boundary voxels contribute 0.

    Args:
        jac: (B, 3, 3, D-2, H-2, W-2) Jacobian J = I + du/dx, indexed [:, row, col].
        mask: (B, 1, D, H, W) binary float mask (e.g. bones).
        eps: Smoothing term.

    Returns:
        Scalar mean orthonormality error inside mask.
    """
    # (J^T J)_ij = sum_k jac[:, k, i] * jac[:, k, j]; jac indexed [b, row, col, d, h, w]
    jtj = torch.einsum("bkidhw,bkjdhw->bijdhw", jac, jac)
    eye = torch.eye(3, device=jac.device, dtype=jac.dtype).view(1, 3, 3, 1, 1, 1)
    m = jtj - eye
    err = (m * m).sum(dim=(1, 2))  # ||J^T J - I||_F^2, (B, D-2, H-2, W-2)

    err = torch.nn.functional.pad(err, (1, 1, 1, 1, 1, 1), value=0.0).unsqueeze(1)
    return (err * mask).sum() / (mask.sum() + eps)


def affine_loss(
    flow: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Penalise deviation from local affine-ness inside mask.

    A displacement field is affine iff all its second-order derivatives vanish,
    i.e. the Jacobian is constant over the region. This penalises the 3D bending
    energy ||d^2 u||^2 (all second partials, including mixed) per voxel, forcing
    the masked region to share a single transform. Combined with
    ``orthonormality_loss`` and ``masked_jac_det_loss`` this yields local
    rigidity. Boundary voxels contribute 0.

    Args:
        flow: (B, D, H, W, 3) displacement field in voxel units.
        mask: (B, 1, D, H, W) binary float mask (e.g. bones).
        eps: Smoothing term.

    Returns:
        Scalar mean bending energy inside mask.
    """
    energy = 0.0
    for c in range(3):
        f = flow[..., c]
        center = f[:, 1:-1, 1:-1, 1:-1]
        # pure second derivatives
        fxx = f[:, 2:, 1:-1, 1:-1] - 2 * center + f[:, :-2, 1:-1, 1:-1]
        fyy = f[:, 1:-1, 2:, 1:-1] - 2 * center + f[:, 1:-1, :-2, 1:-1]
        fzz = f[:, 1:-1, 1:-1, 2:] - 2 * center + f[:, 1:-1, 1:-1, :-2]
        # mixed second derivatives
        fxy = (
            f[:, 2:, 2:, 1:-1]
            - f[:, 2:, :-2, 1:-1]
            - f[:, :-2, 2:, 1:-1]
            + f[:, :-2, :-2, 1:-1]
        ) / 4
        fxz = (
            f[:, 2:, 1:-1, 2:]
            - f[:, 2:, 1:-1, :-2]
            - f[:, :-2, 1:-1, 2:]
            + f[:, :-2, 1:-1, :-2]
        ) / 4
        fyz = (
            f[:, 1:-1, 2:, 2:]
            - f[:, 1:-1, 2:, :-2]
            - f[:, 1:-1, :-2, 2:]
            + f[:, 1:-1, :-2, :-2]
        ) / 4
        energy = energy + (
            fxx * fxx + fyy * fyy + fzz * fzz + 2 * (fxy * fxy + fxz * fxz + fyz * fyz)
        )  # (B, D-2, H-2, W-2)

    energy = torch.nn.functional.pad(energy, (1, 1, 1, 1, 1, 1), value=0.0).unsqueeze(1)
    return (energy * mask).sum() / (mask.sum() + eps)


def enforce_rigidity_loss(
    jac_det: torch.Tensor,
    jac: torch.Tensor,
    flow: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-5,
):
    """Combined local-rigidity loss over a masked region (e.g. bones).

    Enforces the three Staring/Modersitzki conditions (summed with equal weight;
    scale the whole term with ``config.w_bone_rigidity`` at the call site):
      * properness    -- det(J) = 1   (``masked_jac_det_loss``)
      * orthonormality -- J^T J = I    (``orthonormality_loss``)
      * affine-ness    -- d^2 u = 0    (``affine_loss``)

    Args:
        jac_det: (B, 1, D, H, W) Jacobian determinant field.
        jac: (B, 3, 3, D-2, H-2, W-2) Jacobian from ``jacobian.jacobian_matrix``.
        flow: (B, D, H, W, 3) displacement field in voxel units.
        mask: (B, 1, D, H, W) binary float mask.
        eps: Smoothing term.

    Returns:
        (total, (det, ortho, affine)) scalar tensors.
    """
    det = masked_jac_det_loss(jac_det, mask, eps)
    ortho = orthonormality_loss(jac, mask, eps)
    affine = affine_loss(flow, mask, eps)
    total = det + ortho + affine
    return total, (det, ortho, affine)


def warp_binary_mask(
    mask: torch.Tensor,
    disp: torch.Tensor,
    grid: torch.Tensor,
    transform: SpatialTransform_unit,
) -> torch.Tensor:
    """Warp a binary mask with bilinear interpolation (differentiable).

    Args:
        mask: (B, 1, D, H, W) float binary mask.
        disp: (B, 3, D, H, W) displacement field.
        grid: level grid for SpatialTransform_unit.
        transform: SpatialTransform_unit instance.

    Returns:
        (B, 1, D, H, W) warped mask, values in [0, 1].
    """
    flow = disp.permute(0, 2, 3, 4, 1)
    return transform(mask.float(), flow, grid)


def affine_pet_iou(
    x_lbl_pet: torch.Tensor,
    y_lbl_pet: torch.Tensor,
    flow_affine: torch.Tensor,
    grid_full: torch.Tensor,
    transform_nearest: Callable,
    tumour_label: int = 1,
) -> float:
    """IoU of the affine-aligned moving PET mask vs fixed PET mask."""
    x_mask = (x_lbl_pet == tumour_label).float()
    x_mask_aff = transform_nearest(x_mask, flow_affine, grid_full)
    x_bin = x_mask_aff > 0.5
    y_bin = y_lbl_pet == tumour_label

    intersection = (x_bin & y_bin).sum().float()
    union = (x_bin | y_bin).sum().float()
    iou = (intersection / (union + 1e-8)).item()
    return iou


def prereg_residual_rows(
    x_label: torch.Tensor,
    y_label: torch.Tensor,
    flow_prereg: torch.Tensor,
    transform_nearest: torch.nn.Module,
    grid_full: torch.Tensor,
    case_id: str,
    min_voxels: int = 10,
) -> list:
    """Per-label centroid distance between the pre-registered moving labels and
    the fixed labels, in full-resolution voxels.

    This measures the misalignment the pre-registration leaves behind using the
    label maps only, so unlike the magnitude of the network's own flow it does
    not shrink when the network fails to correct a case.
    """
    x_label_prereg = transform_nearest(x_label.float(), flow_prereg, grid_full)
    moving = x_label_prereg[0, 0]
    fixed = y_label[0, 0].float()

    rows = []
    for label in fixed.unique().tolist():
        if label == 0:
            continue
        mask_moving = moving == label
        mask_fixed = fixed == label
        if mask_moving.sum() < min_voxels or mask_fixed.sum() < min_voxels:
            continue
        centroid_moving = mask_moving.nonzero().float().mean(0)
        centroid_fixed = mask_fixed.nonzero().float().mean(0)
        distance = (centroid_moving - centroid_fixed).norm().item()
        rows.append((case_id, int(label), distance))
    return rows


def write_prereg_residual_csv(rows: list, out_path: Path) -> None:
    """Write the centroid distances and print the percentiles that matter."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as handle:
        handle.write("case_id,label,centroid_distance_voxels\n")
        for case_id, label, distance in rows:
            handle.write(f"{case_id},{label},{distance:.4f}\n")

    if not rows:
        print(f"[prereg-residual] no rows written to {out_path}")
        return

    distances = np.array([row[2] for row in rows])
    print(f"\n[prereg-residual] {len(rows)} case/label pairs -> {out_path}")
    print(
        "[prereg-residual] centroid distance (full-res voxels): "
        f"median {np.median(distances):.1f}  "
        f"p95 {np.percentile(distances, 95):.1f}  "
        f"p99 {np.percentile(distances, 99):.1f}  "
        f"max {distances.max():.1f}"
    )

    worst = {}
    for case_id, label, distance in rows:
        worst[label] = max(worst.get(label, 0.0), distance)
    ranked = sorted(worst.items(), key=lambda item: item[1], reverse=True)[:10]
    print(
        "[prereg-residual] worst labels (label: max voxels): "
        + ", ".join(f"{label}: {distance:.0f}" for label, distance in ranked)
    )
