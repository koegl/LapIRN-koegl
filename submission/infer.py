"""PSMAReg (Learn2Reg 2026) test-phase entrypoint.

Registers ONE moving/fixed PET+CT set per run:

    infer.py <fixed_ct> <fixed_pet> <moving_ct> <moving_pet> <output_disp>

fixed  = baseline  (timepoint 00), moving = follow-up (timepoint 01).

Pipeline (mirrors Code/inference.py, chase_best_model, use_io=False):
  1. body-mask + normalise CT/PET -> 2-channel moving (X) / fixed (Y) volumes
  2. ANTs affine prereg on half-res windowed CT -> full-res voxel DVF
  3. warp X by the affine flow, run LapIRN lvl3 (X_affine, Y) -> deformable flow
  4. compose affine (outer) with the deformable field (inner)
  5. convert the composed unit flow to full-res voxel displacements and save
  6. (optional) nnU-Net PET lesion segmentation of the moving timepoint
  7. (optional) instance optimisation of the total field against the tumour
     terms, seeded by that mask

Output: NIfTI, channel-first (3, X, Y, Z), float32, voxel displacements on the
fixed-image grid, identity affine -- the convention of
Code/inference.py:save_disp.
"""

import argparse
import atexit
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

# --- wall-clock anchor --------------------------------------------------------
# The challenge budget is 90 s of WALL time per pair and this container is
# launched once per pair, so the container start, the interpreter, and every
# import below are all inside it. Tuning a fixed IO step count against that is
# fragile -- it would be tuned on this workstation and run on the organizers'
# machine -- so the pipeline instead spends whatever time is left when it gets
# to IO. That needs an anchor as early as possible, and earlier than this file:
# by the time the first statement here runs, the interpreter has already booted
# and `import torch` is still ahead.
#
# /proc/self/stat field 22 is the process start time in clock ticks since boot;
# read against /proc/uptime it gives this process's true age. The ENTRYPOINT is
# `python3 -u /app/infer.py`, so this process IS PID 1 and its exec time is the
# earliest instant observable from inside the container. What remains invisible
# is the runtime's own create/start work before exec -- see --startup-reserve.
_IMPORT_TIME = time.time()


def _process_start_time() -> float:
    """Wall-clock epoch at which this process was exec'd.

    Falls back to module-import time where /proc is unavailable (macOS during
    development), which merely omits the interpreter+import span rather than
    failing -- the container always has /proc.
    """
    try:
        with open("/proc/self/stat", "rb") as fh:
            # the comm field can contain spaces and parentheses, so split only
            # what follows the LAST ')': starttime is then index 19.
            fields = fh.read().rpartition(b")")[2].split()
        starttime_ticks = float(fields[19])
        with open("/proc/uptime") as fh:
            uptime = float(fh.read().split()[0])
        now = time.time()
        return now - (uptime - starttime_ticks / os.sysconf("SC_CLK_TCK"))
    except Exception:
        return _IMPORT_TIME


PROCESS_START = _process_start_time()

# (label, epoch) in the order they were reached. Printed as one block at exit so
# the per-stage spans and the parts that are NOT any stage -- interpreter boot,
# imports, the final write, teardown -- are visible in the same units.
_MARKS: List[Tuple[str, float]] = [("process exec", PROCESS_START)]


def mark(label: str) -> float:
    """Record a timeline point and return its timestamp."""
    now = time.time()
    _MARKS.append((label, now))
    return now


def print_timeline() -> None:
    """Dump the timeline. Registered with atexit so the last mark is as close to
    process death as Python can observe -- the span after the field is written is
    exactly the part the IO deadline has to reserve for."""
    mark("process exit")
    end = _MARKS[-1][1]
    print("timeline (s since process exec):", flush=True)
    previous = PROCESS_START
    for label, stamp in _MARKS[1:]:
        print(
            f"  {stamp - PROCESS_START:7.2f}  +{stamp - previous:6.2f}  {label}",
            flush=True,
        )
        previous = stamp
    print(
        f"  in-container total {end - PROCESS_START:.2f}s "
        f"(exec at epoch {PROCESS_START:.3f}; the container runtime's start-up "
        f"before exec is NOT in this number)",
        flush=True,
    )


# Everything the container writes must land in a writable place: /app and the
# repo copy are read-only for the evaluation user, and $HOME may not exist.
for _var, _val in (
    ("MPLCONFIGDIR", "/tmp/mpl"),
    ("TORCH_HOME", "/tmp/torch"),
    ("XDG_CACHE_HOME", "/tmp/cache"),
    ("HOME", "/tmp"),
):
    os.environ.setdefault(_var, _val)

sys.path.insert(0, os.environ.get("LAPIRN_CODE", "/app/lapirn/Code"))

import affine_reg  # noqa: E402
import Functions  # noqa: E402
import miccai2020_model_stage  # noqa: E402
import my_data  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from config import TrainingConfig  # noqa: E402

# torch alone is several seconds, and all of it is inside the 90 s budget.
mark("imports done")

DEFAULT_WEIGHTS = Path(os.environ.get("LAPIRN_WEIGHTS", "/app/weights/model.pth"))
DEFAULT_SEG_MODEL = Path(os.environ.get("NNUNET_MODEL_DIR", "/app/nnunet_model"))
AUTOPET_DIR = os.environ.get("AUTOPET_DIR", "/app")

# TotalSegmentator runs in its own interpreter (see totalseg_runner.py) and only
# on an axial band. The band itself is cfg.io_seg_z_range, alongside the other IO
# settings in Code/config.py -- it is a property of the IO dice term, not of the
# container.
TOTALSEG_PYTHON = os.environ.get("TOTALSEG_PYTHON", "/opt/tsvenv/bin/python")
TOTALSEG_RUNNER = os.environ.get("TOTALSEG_RUNNER", "/app/totalseg_runner.py")


def build_config() -> TrainingConfig:
    """The model hyper-parameters of the submitted checkpoint.

    These mirror the overrides at the top of Code/inference.py:main; every other
    field keeps its TrainingConfig default. They must match the checkpoint or
    load_state_dict rejects it.
    """
    cfg = TrainingConfig()
    cfg.in_channel = 4
    cfg.start_channel = 7
    cfg.resblock_expansion = 1
    cfg.n_resblocks = 5
    return cfg


def create_model(
    device: torch.device, cfg: TrainingConfig, model_path: Path
) -> torch.nn.Module:
    """Same three-level construction as Code/inference.py:create_model."""
    model_lvl1 = (
        miccai2020_model_stage.Miccai2020_LDR_laplacian_unit_add_lvl1(
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
        if cfg.use_lvl1
        else None
    )

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


def load_pair(
    fixed_ct_path: Path,
    fixed_pet_path: Path,
    moving_ct_path: Path,
    moving_pet_path: Path,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Read one pair and normalise it exactly as Code/inference.py:load_val_pair.

    Returns (X, Y) of shape (1, 2, H, W, D): X = moving (follow-up),
    Y = fixed (baseline); channel 0 = CT, channel 1 = PET.
    """

    def load(path: Path) -> np.ndarray:
        return nib.load(str(path)).get_fdata().astype(np.float32)

    x_ct_raw, y_ct_raw = load(moving_ct_path), load(fixed_ct_path)

    x_mask = my_data.get_body_mask(x_ct_raw)
    y_mask = my_data.get_body_mask(y_ct_raw)

    x_ct_raw = my_data.apply_body_mask(x_ct_raw, x_mask, fill_value=my_data.CT_AIR_HU)
    y_ct_raw = my_data.apply_body_mask(y_ct_raw, y_mask, fill_value=my_data.CT_AIR_HU)

    x_pet_raw = my_data.apply_body_mask(load(moving_pet_path), x_mask, fill_value=0.0)
    y_pet_raw = my_data.apply_body_mask(load(fixed_pet_path), y_mask, fill_value=0.0)

    def t(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(arr).unsqueeze(0)

    x = torch.cat([t(my_data.norm_ct(x_ct_raw)), t(my_data.norm_pet(x_pet_raw))], dim=0)
    y = torch.cat([t(my_data.norm_ct(y_ct_raw)), t(my_data.norm_pet(y_pet_raw))], dim=0)
    return x.unsqueeze(0).float(), y.unsqueeze(0).float()


def affine_flow(
    fixed_ct_path: Path,
    moving_ct_path: Path,
    cfg: TrainingConfig,
    device: torch.device,
) -> torch.Tensor:
    """ANTs affine prereg as a unit flow of shape (1, H, W, D, 3).

    compute_affine_dvf is called directly rather than affine_reg.get_affine_dvf:
    the latter reads/writes a cache under the repo, which is read-only here and
    meaningless for a one-shot run.
    """
    dvf = affine_reg.compute_affine_dvf(
        fixed_ct_path=fixed_ct_path,
        moving_ct_path=moving_ct_path,
        make_lowres_ants_image_fn=affine_reg.make_lowres_ants_image,
        preprocess_ct_fn=affine_reg.preprocess_ct,
        ants_affine_to_fullres_voxel_disp_fn=affine_reg.ants_affine_to_fullres_voxel_disp,
    )
    dvf_tensor = affine_reg.dvf_to_tensor(dvf, device)
    h, w, d = cfg.img_shape
    d_h = dvf_tensor[:, 0] / (h / 2.0)
    d_w = dvf_tensor[:, 1] / (w / 2.0)
    d_d = dvf_tensor[:, 2] / (d / 2.0)
    return torch.stack([d_d, d_w, d_h], dim=1).permute(0, 2, 3, 4, 1)


def compose(
    flow_outer: torch.Tensor, flow_inner: torch.Tensor, grid: torch.Tensor
) -> torch.Tensor:
    """Compose two unit flows (..., 3) into one; returns channel-first (1,3,...).

    `flow_outer` is applied after `flow_inner` -- the affine goes outside the
    network field, so the result is the TOTAL transform from the original moving
    image onto the fixed grid.
    """
    deform_grid = grid + flow_inner
    affine_grid_ch = (grid + flow_outer).permute(0, 4, 1, 2, 3)
    composed = torch.nn.functional.grid_sample(
        affine_grid_ch,
        deform_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    ).permute(0, 2, 3, 4, 1)
    return (composed - grid).permute(0, 4, 1, 2, 3)


def save_disp(total_unit_flow: torch.Tensor, out_path: Path) -> None:
    """Write the submission field: (3, X, Y, Z) float32 voxel displacements.

    Mirrors Code/inference.py:save_disp -- unit flow -> full-res voxel units,
    then the channel axis is reversed so channel 0 corresponds to the first
    spatial axis, which is the order the organizers' warper expects.
    """
    voxel = Functions.transform_unit_flow_to_flow_cuda(
        total_unit_flow.permute(0, 2, 3, 4, 1).clone()
    ).permute(0, 4, 1, 2, 3)
    disp_np = voxel[0].detach().cpu().numpy().astype(np.float32)[::-1].copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(disp_np, np.eye(4)), str(out_path))


def segment_pet(
    ct_path: Path,
    pet_path: Path,
    model_dir: Path,
    device: torch.device,
    use_mirroring: bool,
) -> Tuple[np.ndarray, float, float]:
    """nnU-Net PET lesion mask for one timepoint.

    Reuses autopet-3-submission's own helpers rather than re-deriving the
    preprocessing: `main.py` there stacks [CT, PET] into the two channels the
    Dataset501 model was trained on and hands them to predict_single_npy_array.

    Returns (mask, load_seconds, inference_seconds). The split matters for
    scheduling: building the predictor is CPU and disk (a 245 MB checkpoint plus
    the network build) and could overlap with GPU work, whereas the tiled
    inference is GPU and would only contend with it.
    """
    sys.path.insert(0, AUTOPET_DIR)
    import autopet_main  # noqa: E402  (heavy; imported only when segmenting)

    load_start = time.time()
    # build_predictor prints device and configuration banners, and hardcodes
    # allow_tqdm=True, whose per-tile bar is unreadable in a piped log. Silenced
    # here rather than by editing autopet-3-submission, using that repo's own
    # suppressor. run_inference_in_memory already wraps itself.
    with autopet_main.suppress_output():
        predictor = autopet_main.build_predictor(
            str(model_dir), folds=(0,), device=device, use_mirroring=use_mirroring
        )
    predictor.allow_tqdm = False
    load_seconds = time.time() - load_start

    infer_start = time.time()
    seg = autopet_main.run_inference_in_memory(predictor, str(ct_path), str(pet_path))
    return seg.astype(np.uint8), load_seconds, time.time() - infer_start


def save_seg(seg: np.ndarray, reference_path: Path, out_path: Path) -> None:
    """Write a mask on the geometry of the image it was predicted from."""
    ref = nib.load(str(reference_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(seg.astype(np.uint8), ref.affine, ref.header), str(out_path))


class BackgroundPetSegmentation:
    """Run the PET nnU-Net on a thread, concurrently with registration.

    Registration is ~20 s of CPU (reading and body-masking the volumes, then the
    single-threaded ANTs affine) against ~0.4 s of GPU, so the device is idle for
    almost all of it. PET inference is ~14 s of GPU and nothing else needs it
    until IO, so it fits inside that window almost entirely.

    A thread rather than a process: this model needs the same environment, and
    both antspyx (C++) and torch's CUDA calls release the GIL, so the two really
    do proceed at once. A subprocess would add a fresh `import torch` -- more
    than the overlap saves.
    """

    def __init__(self, ct_path, pet_path, model_dir, device, use_mirroring):
        self.result = None
        self.error = None
        self.load_seconds = self.infer_seconds = 0.0
        self._start = time.time()
        self._thread = threading.Thread(
            target=self._run,
            args=(ct_path, pet_path, model_dir, device, use_mirroring),
            daemon=True,
        )
        self._thread.start()

    def _run(self, ct_path, pet_path, model_dir, device, use_mirroring):
        try:
            self.result, self.load_seconds, self.infer_seconds = segment_pet(
                ct_path, pet_path, model_dir, device, use_mirroring
            )
        except Exception as exc:  # surfaced by join(), never silently swallowed
            self.error = exc

    def join(self):
        """Block for the mask. Returns None on failure rather than raising -- a
        pair with a slightly worse field still scores; a crashed container does
        not."""
        block_start = time.time()
        self._thread.join()
        self.blocking_seconds = time.time() - block_start
        self.elapsed_seconds = time.time() - self._start
        if self.error is not None:
            print(
                f"WARNING: PET segmentation failed ({self.error}); "
                "IO will be skipped and the un-refined field written",
                flush=True,
            )
            return None
        return self.result


def resolve_z_range(z_range: Tuple[int, int], depth: int) -> Tuple[int, int]:
    """Turn cfg.io_seg_z_range into a concrete half-open range.

    A negative bound means "no crop on that side", so (-1, -1) is the whole
    volume. Resolved here rather than downstream so every print and every
    argument passed to the runner is the range actually segmented -- a bare -1
    reaching a Python slice silently means "all but the last", not "all".
    """
    start = 0 if z_range[0] < 0 else min(max(0, z_range[0]), depth)
    stop = depth if z_range[1] < 0 else min(max(0, z_range[1]), depth)
    if stop <= start:
        raise SystemExit(f"empty io_seg_z_range {z_range} for depth {depth}")
    return start, stop


def start_ct_segmentation(
    fixed_ct: Path, moving_ct: Path, out_dir: Path, z_range: Tuple[int, int]
) -> "subprocess.Popen":
    """Launch TotalSegmentator WITHOUT blocking, to overlap with the GPU work.

    A subprocess rather than an import: TotalSegmentator and the autopet nnunetv2
    fork cannot live in one environment (see totalseg_runner.py). Its startup is
    a second `import torch` in a fresh interpreter plus two model loads -- ~6 s
    and ~23 s respectively, largely CPU-bound, so it overlaps well with
    registration and PET segmentation. Nothing needs the CT labels until IO.

    stdout/stderr are inherited so the runner's own timing line and any failure
    land in the container log.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [
            TOTALSEG_PYTHON,
            TOTALSEG_RUNNER,
            str(fixed_ct),
            str(moving_ct),
            str(out_dir),
            str(z_range[0]),
            str(z_range[1]),
        ]
    )


def collect_ct_segmentation(
    proc: "subprocess.Popen",
    out_dir: Path,
    fixed_ct: Path,
    moving_ct: Path,
    device: torch.device,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Join the subprocess and load its labels, or None if anything went wrong.

    Deliberately non-fatal. On the hidden test set a crash here would leave the
    pair with no displacement field at all, which scores worse than a field
    refined without the dice term. The failure is printed loudly instead, so it
    cannot pass unnoticed in a validation sweep.

    Returns (moving, fixed) label tensors of shape (1, 1, H, W, D) on the full
    grid -- zero outside the z crop -- in the ORIGINAL frame of each image, which
    is the frame run_io expects for x_lbl_ct / y_lbl_ct.
    """
    returncode = proc.wait()
    if returncode != 0:
        print(
            f"WARNING: TotalSegmentator exited with code {returncode}; "
            "continuing with IO but WITHOUT the CT dice term",
            flush=True,
        )
        return None

    paths = (out_dir / moving_ct.name, out_dir / fixed_ct.name)
    missing = [p for p in paths if not p.exists()]
    if missing:
        print(
            f"WARNING: TotalSegmentator produced no {[p.name for p in missing]}; "
            "continuing with IO but WITHOUT the CT dice term",
            flush=True,
        )
        return None

    try:
        def load(path: Path) -> torch.Tensor:
            arr = nib.load(str(path)).get_fdata().astype(np.int16)
            return torch.from_numpy(arr)[None, None].to(device).float()

        return load(paths[0]), load(paths[1])
    except Exception as exc:  # a truncated or unreadable label file
        print(
            f"WARNING: could not read the CT labels ({exc}); "
            "continuing with IO but WITHOUT the CT dice term",
            flush=True,
        )
        return None


def run_instance_optimisation(
    total_unit_flow: torch.Tensor,
    X: torch.Tensor,
    Y: torch.Tensor,
    pet_mask: np.ndarray,
    ct_labels: Optional[Tuple[torch.Tensor, torch.Tensor]],
    transform: torch.nn.Module,
    transform_nearest: torch.nn.Module,
    grid: torch.Tensor,
    cfg: TrainingConfig,
    device: torch.device,
    deadline: Optional[float],
) -> Tuple[torch.Tensor, int]:
    """Refine the TOTAL field with Code/instance_opt.py:run_io.

    Returns (field, steps_actually_taken) -- with a deadline the step count is
    whatever fitted, so it cannot be read off cfg.io_it any more.

    The dice term is on when TotalSegmentator ran, off otherwise -- include_dice
    is the only gate on the CT labels, so with none available the objective falls
    back to NCC + smoothness + the Jacobian barrier + the PET tumour terms.

    include_rigidity stays off regardless: it needs per-label bone values, and the
    best leaderboard configuration ran with w_io_bone_rigidity = 0.

    log_hard_dice=False drops the 117-label progress-bar metric. It is pure
    logging and costs a full-volume comparison per label per step, which the
    container's runtime budget cannot spare.

    The moving image and labels are the ORIGINAL ones and the field is the total
    (affine-composed) transform, so the volume terms see det(A) and match what
    the scorer measures -- same convention as Code/inference.py.
    """
    import instance_opt  # noqa: E402  (pulls utils -> mlflow/wandb; import late)

    x_lbl_pet = torch.from_numpy(pet_mask.astype(np.float32))[None, None].to(device)
    x_lbl_ct, y_lbl_ct = ct_labels if ct_labels is not None else (None, None)

    refined = instance_opt.run_io(
        Y,
        total_unit_flow,
        X,
        x_lbl_ct,
        x_lbl_pet,
        y_lbl_ct,
        transform,
        transform_nearest,
        grid,
        cfg,
        device,
        include_pet=True,
        include_rigidity=False,
        include_dice=ct_labels is not None,
        log_hard_dice=False,
        use_class_weights=False,
        deadline=deadline,
    )
    return refined, instance_opt.run_io.steps_taken


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PSMAReg LapIRN inference: one PET+CT set -> displacement field."
    )
    parser.add_argument("fixed_ct", type=Path, help="baseline CT  (..._0000_00.nii.gz)")
    parser.add_argument("fixed_pet", type=Path, help="baseline PET (..._0001_00.nii.gz)")
    parser.add_argument("moving_ct", type=Path, help="follow-up CT  (..._0000_01.nii.gz)")
    parser.add_argument("moving_pet", type=Path, help="follow-up PET (..._0001_01.nii.gz)")
    parser.add_argument("output_disp", type=Path, help="where to write the field")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--segment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run nnU-Net PET lesion segmentation on the moving timepoint "
        "(does not affect the displacement field yet)",
    )
    parser.add_argument(
        "--seg-dir",
        type=Path,
        default=None,
        help="write the PET mask here, named after the moving PET input. Without "
        "it the mask is computed and discarded -- useful only for timing.",
    )
    parser.add_argument("--seg-model", type=Path, default=DEFAULT_SEG_MODEL)
    parser.add_argument(
        "--seg-mirroring",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="3D mirror TTA: 8x the tiles for a small accuracy gain (default: off, "
        "matching autopet-3-submission/inference_all.py)",
    )
    parser.add_argument(
        "--totalseg",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="TotalSegmentator CT labels for both timepoints. Not consumed by "
        "IO yet -- present so the crop can be dialled in against the budget.",
    )
    parser.add_argument(
        "--ct-seg-dir",
        type=Path,
        default=None,
        help="write the CT labels here, named after the CT inputs",
    )
    parser.add_argument(
        "--io",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="instance optimisation of the total field (requires --segment). "
        "Its hyper-parameters -- io_it, io_lr and the w_io_* weights -- all come "
        "from Code/config.py; there is no override here.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = mark("main() entered")

    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device(args.device)
    cfg = build_config()

    # The 90 s window opens when the CONTAINER starts, which is --startup-reserve
    # before this process was exec'd; IO has to be finished --finish-reserve
    # earlier than that so the field still gets written. None disables the whole
    # mechanism and restores fixed cfg.io_it steps.
    io_deadline: Optional[float] = None
    if cfg.io_time_budget > 0:
        container_start = PROCESS_START - cfg.io_startup_reserve
        io_deadline = container_start + cfg.io_time_budget - cfg.io_finish_reserve
        print(
            f"time budget {cfg.io_time_budget:.0f}s from container start "
            f"(exec +{cfg.io_startup_reserve:.1f}s), reserving "
            f"{cfg.io_finish_reserve:.1f}s to write the field -> IO must end "
            f"{io_deadline - PROCESS_START:.1f}s after exec",
            flush=True,
        )

    # Launched before anything else so its interpreter startup and model loading
    # overlap with the GPU work below; joined just before IO, the first consumer.
    ts_proc = None
    ts_z_range = resolve_z_range(cfg.io_seg_z_range, cfg.img_shape[2])
    ts_dir = args.ct_seg_dir or Path("/tmp/ct_labels")
    ts_launched = time.time()
    if args.totalseg:
        ts_proc = start_ct_segmentation(
            args.fixed_ct, args.moving_ct, ts_dir, ts_z_range
        )

    grid = Functions.generate_grid_unit(cfg.img_shape)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape)).to(device).float()
    transform = miccai2020_model_stage.SpatialTransform_unit().to(device)

    build_start = time.time()
    model = create_model(device, cfg, args.weights)
    build_seconds = time.time() - build_start

    # Launched before registration so its GPU inference fills the window ANTs
    # spends on the CPU; joined below, before IO needs the mask.
    pet_job = None
    if args.segment:
        pet_job = BackgroundPetSegmentation(
            args.moving_ct, args.moving_pet, args.seg_model, device, args.seg_mirroring
        )

    load_start = time.time()
    X, Y = load_pair(args.fixed_ct, args.fixed_pet, args.moving_ct, args.moving_pet)
    X, Y = X.to(device), Y.to(device)
    load_seconds = time.time() - load_start

    # ANTs is CPU-bound and single-threaded; the LapIRN forward is GPU. Reported
    # apart because only the CPU part can usefully overlap with other GPU work.
    affine_start = time.time()
    flow_affine = affine_flow(args.fixed_ct, args.moving_ct, cfg, device)
    affine_seconds = time.time() - affine_start

    net_start = time.time()
    X_affine = transform(X, flow_affine, grid)
    with torch.no_grad():
        F_X_Y, _, _, _, _, _, _ = model(X_affine, Y)
    total_unit_flow = compose(flow_affine, F_X_Y.permute(0, 2, 3, 4, 1), grid)
    net_seconds = time.time() - net_start

    reg_seconds = mark("registration done") - start
    print(
        f"registration {reg_seconds:.1f}s "
        f"(read {load_seconds:.1f}s + model build {build_seconds:.1f}s + "
        f"ANTs affine {affine_seconds:.1f}s [CPU] + network {net_seconds:.1f}s [GPU])",
        flush=True,
    )

    seg_seconds = seg_blocking = io_seconds = 0.0
    seg = None
    if pet_job is not None:
        seg = pet_job.join()
        mark("PET mask joined")
        seg_seconds, seg_blocking = pet_job.elapsed_seconds, pet_job.blocking_seconds
        if seg is not None:
            print(
                f"PET segmentation {seg_seconds:.1f}s elapsed, {seg_blocking:.1f}s blocking "
                f"(model load {pet_job.load_seconds:.1f}s [CPU/disk] + "
                f"inference {pet_job.infer_seconds:.1f}s [GPU]), "
                f"{int(seg.sum())} lesion voxels",
                flush=True,
            )
            if args.seg_dir is not None:
                save_seg(seg, args.moving_pet, args.seg_dir / args.moving_pet.name)

    totalseg_seconds = totalseg_blocking = 0.0
    ct_labels = None
    if ts_proc is not None:
        join_start = time.time()
        ct_labels = collect_ct_segmentation(
            ts_proc, ts_dir, args.fixed_ct, args.moving_ct, device
        )
        totalseg_blocking = mark("CT labels joined") - join_start
        totalseg_seconds = time.time() - ts_launched

    if args.io and seg is None:
        print(
            "WARNING: no PET mask, so IO cannot run its tumour terms; "
            "writing the un-refined field",
            flush=True,
        )
    # Everything IO needs is in hand, so the whole remaining budget is its own:
    # nothing after this point competes for the GPU. Skipping it entirely is a
    # legitimate outcome -- the un-refined field still scores, an overrun pair
    # scores nothing.
    io_steps = 0
    if args.io and seg is not None:
        remaining = None if io_deadline is None else io_deadline - time.time()
        if remaining is not None and remaining < cfg.io_min_step_seconds:
            print(
                f"skipping IO: {remaining:.1f}s left before the deadline, "
                f"below the {cfg.io_min_step_seconds:.1f}s a first step is "
                f"assumed to cost; writing the un-refined field",
                flush=True,
            )
        else:
            if remaining is not None:
                print(f"IO has {remaining:.1f}s of budget left", flush=True)
            io_start = mark("IO start")
            transform_nearest = miccai2020_model_stage.SpatialTransformNearest_unit().to(device)
            total_unit_flow, io_steps = run_instance_optimisation(
                total_unit_flow, X, Y, seg, ct_labels,
                transform, transform_nearest, grid, cfg, device,
                deadline=io_deadline,
            )
            io_seconds = mark("IO done") - io_start
            step_ceiling = cfg.io_it if io_deadline is None else cfg.io_max_steps
            print(
                f"instance optimisation {io_seconds:.1f}s "
                f"({io_steps} of at most {step_ceiling} steps @ lr {cfg.io_lr}, "
                f"{io_seconds / max(io_steps, 1):.2f}s/step)",
                flush=True,
            )

    save_disp(total_unit_flow, args.output_disp)
    mark("field written")
    print(
        f"wrote {args.output_disp}\n"
        f"  registration {reg_seconds:.1f}s | "
        f"PET seg {seg_seconds:.1f}s elapsed, {seg_blocking:.1f}s blocking | "
        f"IO {io_seconds:.1f}s ({io_steps} steps) | CT seg {totalseg_seconds:.1f}s elapsed, "
        f"{totalseg_blocking:.1f}s blocking (z {ts_z_range[0]}..{ts_z_range[1]}, "
        f"{ts_z_range[1] - ts_z_range[0]} slices) | "
        f"total {time.time() - start:.1f}s\n"
        f"  (PET and CT segmentation run concurrently with registration, so the "
        f"stages do not sum to the total; only their blocking parts add to it)",
        flush=True,
    )


if __name__ == "__main__":
    # Registered before main() so the timeline is printed even when a stage
    # raises -- a run that overran is exactly the one whose timing is wanted.
    atexit.register(print_timeline)
    main()
