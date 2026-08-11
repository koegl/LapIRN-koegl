import contextlib
import json
import math
import os
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

import jacobian
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


class GradConflictTracker:
    """Running window over the gradient-conflict measurements.

    Single-step cosines are noisy, so we keep the last `window` values and log
    their mean / std / fraction-negative alongside the instantaneous value.
    """

    def __init__(self, window: int = 20) -> None:
        self.window = window
        self._hist: Dict[str, list[float]] = {}

    def update(self, values: Dict[str, float]) -> Dict[str, float]:
        """Push one measurement, get the windowed stats back.

        Every key gets a `_mean`. Keys naming a cosine additionally get the
        spread stats, because for a cosine the spread is the whole question:
        a mean of 0 with a large std is violent conflict cancelling itself out,
        not the absence of conflict.
        """
        out: Dict[str, float] = {}
        for key, value in values.items():
            hist = self._hist.setdefault(key, [])
            hist.append(value)
            if len(hist) > self.window:
                hist.pop(0)
            arr = np.asarray(hist)
            out[f"{key}_mean"] = float(arr.mean())
            if key == "cos" or key.startswith("cos_"):
                out[f"{key}_std"] = float(arr.std())
                out[f"{key}_frac_negative"] = float((arr < 0).mean())
                out[f"{key}_min"] = float(arr.min())
                out[f"{key}_max"] = float(arr.max())
        out["n_samples"] = float(max((len(h) for h in self._hist.values()), default=0))
        return out


def _flat_grad(
    loss: torch.Tensor, params: list[torch.nn.Parameter]
) -> torch.Tensor | None:
    """Flattened dL/dparams, or None if `loss` is not attached to the graph."""
    if not loss.requires_grad:
        return None
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=True,
        allow_unused=True,
    )
    flat = [
        (g if g is not None else torch.zeros_like(p)).reshape(-1).float()
        for g, p in zip(grads, params)
    ]
    return torch.cat(flat)


def _cos(u: torch.Tensor, v: torch.Tensor, eps: float = 1e-12) -> float | None:
    nu, nv = u.norm(), v.norm()
    if nu < eps or nv < eps:
        return None
    return float((torch.dot(u, v) / (nu * nv)).item())


def gradient_conflict_report(
    named_losses_a: Dict[str, torch.Tensor],
    named_losses_b: Dict[str, torch.Tensor],
    model: torch.nn.Module,
    named_probes: Optional[Dict[str, torch.Tensor]] = None,
    loss_a_single: Optional[torch.Tensor] = None,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """Measure how every loss term pulls the shared parameters against every other.

    `named_losses_a` are the accuracy + regularisation terms, `named_losses_b` the
    tumour terms, each mapping a name to its *weighted* value. Weights matter for
    the norms (which say whether a term is loud enough to matter at all) but not
    for the cosines, which are scale invariant -- which is why a conflict shown
    here cannot be removed by scaling a whole group.

    Gradients are linear, so each group's aggregate gradient is the sum of its
    term gradients: the group-level numbers cost no extra backward passes on top
    of the per-term ones. Total cost is len(a) + len(b) backward passes.

    Keys:
      cos              group A vs group B as a whole. > 0 they agree (no conflict
                       to architect around); ~ 0 orthogonal, the parameters have
                       room for both; < 0 genuine conflict, every update trades
                       one against the other.
      norm_ratio       |g_a| / |g_b|; far from 1 means one group dominates the
                       update regardless of the cosine, so the cosine is moot.
      cos_a_vs_<b>     group A against one tumour term - which term fights.
      cos_<a>_vs_b     one accuracy term against group B.
      cos_<x>_vs_<y>   the full pairwise matrix, within and across groups. Within
                       a group this says whether that group is one coherent
                       objective or is already fighting itself.
      share_<x>        |g_x| / sum of that group's term norms; a term with a tiny
                       share cannot matter however hostile its cosine. Sums to 1
                       within each group, per step and after windowing.
      present_<x>      1 when the term had a gradient this step, 0 when it was a
                       constant. Its windowed mean is the fraction of steps the
                       term was live, i.e. the sample base behind its cosine.

    `named_probes` are diagnostic-only losses: they get norms, presence and every
    pairwise cosine, but are deliberately excluded from both group aggregates and
    from the shares. Use them for quantities that are not additive parts of the
    objective -- e.g. Dice restricted to bone vs to soft tissue, whose class means
    do not sum back to the full Dice term. Including them in a group would corrupt
    `cos`, `norm_ratio` and the shares.

    Uses autograd.grad, so it never touches `.grad` and does not disturb gradient
    accumulation. Call it sparingly.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        return {}

    # Per-term gradients. A term can be a constant this step and contribute no
    # gradient at all (a case with no PET-positive lesion zeroes tlg/jactum; an
    # abdomen batch zeroes rigidity). Such a term is recorded as *present=0 with
    # norm 0* rather than omitted: dropping the key would shrink the share
    # denominator for that step AND leave each term's running window covering a
    # different subset of steps, so the windowed shares would not sum to 1.
    probes = named_probes or {}
    grads: Dict[str, Optional[torch.Tensor]] = {}
    for group in (named_losses_a, named_losses_b, probes):
        for name, term in group.items():
            g = _flat_grad(term, params)
            grads[name] = g if (g is not None and g.norm() >= eps) else None

    def aggregate(group: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        live = [grads[n] for n in group if grads[n] is not None]
        if not live:
            return None
        g = torch.stack(live).sum(dim=0)
        return g if g.norm() >= eps else None

    g_a = aggregate(named_losses_a)
    g_b = aggregate(named_losses_b)
    if g_a is None or g_b is None:
        return {}

    n_a, n_b = g_a.norm(), g_b.norm()
    out: Dict[str, float] = {
        "cos": float((torch.dot(g_a, g_b) / (n_a * n_b)).item()),
        "norm_a": float(n_a.item()),
        "norm_b": float(n_b.item()),
        "norm_ratio": float((n_a / n_b).item()),
    }

    # Verification: g_a is assembled by summing per-term gradients, which is
    # exact only in exact arithmetic. Under bf16 autocast and checkpointed
    # recomputation it need not match one backward on the summed loss. Pass
    # loss_a_single to measure the discrepancy directly instead of assuming it.
    if loss_a_single is not None:
        g_a_single = _flat_grad(loss_a_single, params)
        if g_a_single is not None and g_a_single.norm() >= eps:
            out["cos_singlepass"] = float(
                (torch.dot(g_a_single, g_b) / (g_a_single.norm() * n_b)).item()
            )
            out["agg_rel_err"] = float(
                ((g_a - g_a_single).norm() / g_a_single.norm()).item()
            )
            out["agg_cos_delta"] = out["cos"] - out["cos_singlepass"]

    # shares are normalised within each group, so each group's shares sum to 1
    for group, aggr, aggr_key in (
        (named_losses_a, g_b, "vs_b"),
        (named_losses_b, g_a, "a_vs"),
    ):
        norms = {
            n: (float(grads[n].norm().item()) if grads[n] is not None else 0.0)
            for n in group
        }
        norm_sum = sum(norms.values())
        for name in group:
            out[f"norm_{name}"] = norms[name]
            out[f"share_{name}"] = norms[name] / norm_sum if norm_sum > eps else 0.0
            out[f"present_{name}"] = 1.0 if grads[name] is not None else 0.0
            if grads[name] is not None:
                c = _cos(aggr, grads[name], eps)
                if c is not None:
                    key = f"cos_{name}_vs_b" if aggr_key == "vs_b" else f"cos_a_vs_{name}"
                    out[key] = c

    # probes: norms and presence only, no share (they belong to no group)
    for name in probes:
        g = grads[name]
        out[f"norm_{name}"] = float(g.norm().item()) if g is not None else 0.0
        out[f"present_{name}"] = 1.0 if g is not None else 0.0
        if g is not None:
            for aggr, key in ((g_a, f"cos_a_vs_{name}"), (g_b, f"cos_{name}_vs_b")):
                c = _cos(aggr, g, eps)
                if c is not None:
                    out[key] = c

    # full pairwise matrix over every live term, within and across groups
    live_names = [n for n in grads if grads[n] is not None]
    for i, n1 in enumerate(live_names):
        for n2 in live_names[i + 1 :]:
            c = _cos(grads[n1], grads[n2], eps)
            if c is not None:
                out[f"cos_{n1}_vs_{n2}"] = c

    return out


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


def seg_head_loss(
    probs: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Auxiliary segmentation loss: soft Dice + BCE, per channel.

    Takes probabilities rather than logits because one of the two channels is
    warped into another frame before it is scored: interpolating probabilities
    is meaningful and grid_sample's zero padding reads as "background", whereas
    a zero-padded logit would read as p=0.5 outside the field of view.

    Args:
        probs: (B, C, D, H, W) head output after sigmoid, in [0, 1].
        target: (B, C, D, H, W) binary float ground truth.
        mask: optional (B, 1, D, H, W) binary float mask (e.g. body); voxels
            outside it are excluded from both terms.
        eps: Smoothing term.

    Returns:
        (loss, (dice_term, bce_term)), each a scalar meaned over channels. The
        two terms are returned separately because they behave very differently
        during training: BCE is near-zero from the start (lesions are ~0.1% of
        the volume, so predicting background is already almost right), while
        Dice is the term that actually reports whether lesions are found.
    """
    if mask is not None:
        probs = probs * mask
        target = target * mask

    intersection = (probs * target).sum(dim=(2, 3, 4))
    denominator = probs.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
    dice = 1.0 - (2.0 * intersection + eps) / (denominator + eps)

    bce = torch.nn.functional.binary_cross_entropy(
        probs.clamp(eps, 1.0 - eps), target, reduction="none"
    )
    if mask is not None:
        bce = (bce * mask).sum(dim=(2, 3, 4)) / (mask.sum(dim=(2, 3, 4)) + eps)
    else:
        bce = bce.mean(dim=(2, 3, 4))

    dice = dice.mean()
    bce = bce.mean()
    return dice + bce, (dice, bce)


def seg_head_dice_scores(
    probs: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    threshold: float = 0.5,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Hard Dice per channel, for logging (not for backprop).

    The soft Dice inside seg_head_loss is a training signal; this is the number
    you would quote. A channel whose target is empty (no lesion in that scan)
    scores 1.0 when the prediction is empty too and 0.0 otherwise.

    Args:
        probs: (B, C, D, H, W) head output after sigmoid, in [0, 1].
        target: (B, C, D, H, W) binary float ground truth.
        mask: optional (B, 1, D, H, W) binary float mask (e.g. body).
        threshold: probability above which a voxel counts as lesion.
        eps: Smoothing term.

    Returns:
        (C,) tensor of Dice scores, meaned over the batch.
    """
    prediction = (probs > threshold).float()
    target = (target > 0.5).float()
    if mask is not None:
        prediction = prediction * mask
        target = target * mask

    intersection = (prediction * target).sum(dim=(2, 3, 4))
    denominator = prediction.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
    return ((2.0 * intersection + eps) / (denominator + eps)).mean(dim=0)


def seg_head_moving_probs(
    seg_logits: torch.Tensor,
    lvl2_disp_up_inv: torch.Tensor,
    transform: torch.nn.Module,
    grid_full: torch.Tensor,
) -> torch.Tensor:
    """Channel 1 of the seg head, pulled back into the moving (prereg) frame.

    The head predicts in the fixed frame, but the moving lesion mask is what the
    IO objective consumes (MTV / TLG / masked Jacobian all key off the moving
    tumour). Warping the soft probabilities with the inverse lvl2 field, rather
    than warping a hard label the other way, keeps the interpolation continuous
    and avoids the nearest-neighbour quantisation that would otherwise put a
    small lesion a voxel off its own intensity blob.

    Args:
        seg_logits: (B, 2, D, H, W) head output stashed by the lvl3 forward.
        lvl2_disp_up_inv: (B, 3, D, H, W) inverse lvl2 displacement, detached.
        transform: trilinear warper.
        grid_full: full-resolution unit sampling grid.

    Returns:
        (B, 1, D, H, W) lesion probabilities in the moving frame.
    """
    moving_prob_fixed_frame = torch.sigmoid(seg_logits[:, 1:2])
    return transform(
        moving_prob_fixed_frame,
        lvl2_disp_up_inv.permute(0, 2, 3, 4, 1),
        grid_full,
    )


def seg_head_terms(
    seg_logits: torch.Tensor,
    lvl2_disp_up_inv: torch.Tensor,
    y_label_pet: torch.Tensor,
    moving_pet_mask: torch.Tensor,
    transform: torch.nn.Module,
    grid_full: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Score the auxiliary seg head against both lesion labels, each in its own frame.

    Channel 0 stays in the fixed frame and is compared with the fixed scan's own
    label. Channel 1 is warped back into the moving (prereg) frame and compared
    with the untouched moving label -- so neither target is resampled, and
    channel 1 is scored as the thing IO will actually consume.

    Args:
        seg_logits: (B, 2, D, H, W) head output stashed by the lvl3 forward.
        lvl2_disp_up_inv: (B, 3, D, H, W) inverse lvl2 displacement, detached.
        y_label_pet: (B, 1, D, H, W) fixed lesion label.
        moving_pet_mask: (B, 1, D, H, W) moving lesion mask, affine/prereg frame.
        transform: trilinear warper.
        grid_full: full-resolution unit sampling grid.
        mask: optional body mask. It is the fixed scan's, applied to both
            channels: the prereg brings the moving body into close enough
            alignment that it serves as a support for the moving frame too.

    Returns:
        (loss, metrics) where metrics holds the split Dice/BCE terms and the
        hard Dice of each channel, as plain floats for logging.
    """
    probs = torch.cat(
        [
            torch.sigmoid(seg_logits[:, 0:1]),
            seg_head_moving_probs(seg_logits, lvl2_disp_up_inv, transform, grid_full),
        ],
        dim=1,
    )
    target = torch.cat([(y_label_pet == 1).float(), moving_pet_mask], dim=1)

    loss, (dice_term, bce_term) = seg_head_loss(probs, target, mask)
    with torch.no_grad():
        scores = seg_head_dice_scores(probs, target, mask)

    metrics = {
        "seg_dice_loss": dice_term.item(),
        "seg_bce": bce_term.item(),
        "seg_dice_fixed": scores[0].item(),
        "seg_dice_moving": scores[1].item(),
    }
    return loss, metrics


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
    w_det: float = 1.0,
    w_ortho: float = 1.0,
    w_affine: float = 1.0,
):
    """Combined local-rigidity loss over a masked region (e.g. bones).

    Enforces the three Staring/Modersitzki conditions (scale the whole term with
    ``config.w_bone_rigidity`` at the call site):
      * properness    -- det(J) = 1   (``masked_jac_det_loss``)
      * orthonormality -- J^T J = I    (``orthonormality_loss``)
      * affine-ness    -- d^2 u = 0    (``affine_loss``)

    The three are returned unweighted so they can be logged on their own scale;
    only ``total`` carries the sub-weights. They are not interchangeable: det
    alone is blind to shear, and orthonormality pins only |det J| = 1, so det is
    what excludes reflections.

    Args:
        jac_det: (B, 1, D, H, W) Jacobian determinant field.
        jac: (B, 3, 3, D-2, H-2, W-2) Jacobian from ``jacobian.jacobian_matrix``.
        flow: (B, D, H, W, 3) displacement field in voxel units.
        mask: (B, 1, D, H, W) binary float mask.
        eps: Smoothing term.
        w_det, w_ortho, w_affine: relative weights of the three conditions.

    Returns:
        (total, (det, ortho, affine)) scalar tensors, the three unweighted.
    """
    det = masked_jac_det_loss(jac_det, mask, eps)
    ortho = orthonormality_loss(jac, mask, eps)
    affine = affine_loss(flow, mask, eps)
    total = w_det * det + w_ortho * ortho + w_affine * affine
    return total, (det, ortho, affine)


def per_label_rigid_loss(
    flow: torch.Tensor,
    labels: torch.Tensor,
    label_values: torch.Tensor,
    min_voxels: int = 50,
    eps: float = 1e-8,
):
    """Per-structure rigidity: distance from each label's own best rigid fit.

    For every label separately, fit the single rigid transform (R, t) that best
    explains that structure's displacement, then penalise the squared residual
    of each of its voxels against that fit::

        loss_l = mean_i || (p_i + u_i) - (R_l p_i + t_l) ||^2      (voxel^2)
        loss   = mean over labels of loss_l

    This replaces ``enforce_rigidity_loss`` and fixes both of its failure modes,
    for the same reason: it reads no neighbourhood at all. There is no finite
    difference, so no stencil can cross into soft tissue, and each label is
    fitted independently, so two adjacent bones may move relative to each other
    at zero cost. Measured on the old term (see test_bone_rigidity_loss.py), a
    field with zero displacement inside every bone still scored ~50, of which
    100% vanished after one erosion, and ~45% of its gradient landed on
    non-bone voxels.

    (R, t) are detached. That is exact rather than an approximation: they
    minimise the residual, so the partial derivative of the loss with respect to
    them is zero and by the envelope theorem the total derivative w.r.t. ``u``
    equals the partial taken at fixed (R, t). Detaching only avoids
    differentiating through the SVD, which is numerically fragile.

    Labels with fewer than ``min_voxels`` voxels are skipped: the fit is
    ill-posed for a handful of points. Each surviving label contributes its own
    mean, so a rib counts as much as the pelvis rather than being drowned by it.

    NOTE: the units are voxel^2, not the dimensionless strain measures of
    ``enforce_rigidity_loss``, so ``config.w_bone_rigidity`` must be retuned when
    switching. Compare ``train_lvl3/rig_residual`` against the old
    ``train_lvl3/rigidity`` to pick the new scale.

    Args:
        flow: (B, D, H, W, 3) displacement in voxel units.
        labels: (B, 1, D, H, W) or (B, D, H, W) integer label map.
        label_values: 1-D tensor of the label values to constrain (e.g. bones).
        min_voxels: labels smaller than this are ignored.
        eps: guard for empty groups.

    Returns:
        (loss, {"n_labels": .., "worst": ..}) -- the count of labels that
        contributed and the largest per-label residual, both as scalar tensors.
    """
    work = torch.float32  # the SVD is not safe in bf16/fp16
    flow = flow.to(work)
    b, d, h, w, _ = flow.shape
    device = flow.device
    zero = torch.zeros((), device=device, dtype=work)

    lab = labels.reshape(b, -1).long()
    values = label_values.reshape(-1).long()
    if values.numel() == 0 or lab.numel() == 0:
        return zero, {"n_labels": zero, "worst": zero}

    # value -> compact column index, -1 for everything we do not constrain
    n_lut = int(max(int(lab.max().item()), int(values.max().item()))) + 1
    lut = torch.full((n_lut,), -1, device=device, dtype=torch.long)
    lut[values] = torch.arange(values.numel(), device=device)
    lab_idx = lut[lab.clamp_min(0)]

    n_lab = values.numel()
    batch_of = torch.arange(b, device=device).view(b, 1).expand_as(lab_idx)
    group_all = (lab_idx + batch_of * n_lab).reshape(-1)
    keep = (lab_idx >= 0).reshape(-1)
    if not bool(keep.any()):
        return zero, {"n_labels": zero, "worst": zero}
    group = group_all[keep]

    grid = jacobian.identity_grid((d, h, w), device, work)  # (1, D, H, W, 3)
    p_all = grid.reshape(1, -1, 3).expand(b, -1, -1).contiguous().reshape(-1, 3)
    p = p_all[keep]
    q = p + flow.reshape(-1, 3)[keep]

    n_groups = b * n_lab
    counts = torch.zeros(n_groups, device=device, dtype=work).index_add_(
        0, group, torch.ones_like(group, dtype=work)
    )
    sum_p = torch.zeros(n_groups, 3, device=device, dtype=work).index_add_(0, group, p)
    sum_q = torch.zeros(n_groups, 3, device=device, dtype=work).index_add_(0, group, q)

    safe_n = counts.clamp_min(1.0).unsqueeze(-1)
    pbar = sum_p / safe_n
    qbar = sum_q / safe_n

    # Second pass, centring BEFORE the outer products. The one-pass identity
    # sum(p q^T) - n pbar qbar^T is algebraically equal but numerically wrong
    # here: p holds absolute voxel indices, so for a small bone far from the
    # origin (centroid ~200, spread ~10) the two terms are ~400x larger than
    # their difference and cancel away most of the mantissa. The rotation then
    # comes out visibly wrong for exactly-rigid input on small structures.
    # Centred vectors are O(the structure's radius), so no cancellation.
    p_c = p - pbar[group]
    q_c = q - qbar[group]
    cov = torch.zeros(n_groups, 3, 3, device=device, dtype=work).index_add_(
        0, group, p_c.unsqueeze(2) * q_c.unsqueeze(1)
    )

    # Kabsch: with cov = sum p~ q~^T = U S V^T, the minimiser of
    # sum ||q~ - R p~||^2 is R = V diag(1, 1, sign(det(V U^T))) U^T. The sign
    # term is what forbids a reflection -- the role det(J) = 1 played before.
    u_mat, _, vh = torch.linalg.svd(cov.double())
    v_mat = vh.transpose(-2, -1)
    sign = torch.sign(torch.det(v_mat @ u_mat.transpose(-2, -1)))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    flip = torch.eye(3, device=device, dtype=torch.float64).expand(n_groups, 3, 3)
    flip = flip.clone()
    flip[:, 2, 2] = sign
    rot = (v_mat @ flip @ u_mat.transpose(-2, -1)).to(work).detach()

    # Residual in centred coordinates. With t = qbar - R pbar this is exactly
    #     q - (R p + t) = (q - qbar) - R (p - pbar)
    # but both sides are O(radius) instead of O(distance from the origin), so
    # the subtraction does not cancel away the mantissa. pbar/qbar are detached,
    # which makes this identical to the detached-t formulation.
    pbar_d, qbar_d = pbar.detach(), qbar.detach()
    resid_vec = (q - qbar_d[group]) - torch.einsum(
        "gij,gj->gi", rot[group], p - pbar_d[group]
    )
    resid = (resid_vec**2).sum(-1)
    sum_res = torch.zeros(n_groups, device=device, dtype=work).index_add_(
        0, group, resid
    )
    per_label = sum_res / counts.clamp_min(eps)

    usable = counts >= min_voxels
    if not bool(usable.any()):
        return zero, {"n_labels": zero, "worst": zero, "worst_label": zero}
    kept = per_label[usable]
    # Which bone is worst, not just how bad. The loss is a mean over labels, so
    # one structure blowing up is invisible in the total; this names it.
    worst_pos = torch.argmax(kept)
    worst_group = torch.nonzero(usable, as_tuple=False).reshape(-1)[worst_pos]
    worst_label = values[worst_group % n_lab]
    return (
        kept.mean(),
        {
            "n_labels": usable.sum().to(work),
            "worst": kept.max(),
            "worst_label": worst_label.to(work),
        },
    )


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
