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
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

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

DEFAULT_WEIGHTS = Path(os.environ.get("LAPIRN_WEIGHTS", "/app/weights/model.pth"))
DEFAULT_SEG_MODEL = Path(os.environ.get("NNUNET_MODEL_DIR", "/app/nnunet_model"))
AUTOPET_DIR = os.environ.get("AUTOPET_DIR", "/app")

# TotalSegmentator runs in its own interpreter (see totalseg_runner.py) and only
# on a z-cropped volume: runtime falls roughly linearly with axial extent, but
# from a large fixed floor -- 47.5 s for both volumes at z=288 against 32.3 s at
# z=108, i.e. ~23 s of it is model loading. TOTALSEG_CROP slices are removed from
# EACH end, so z = 288 - 2*crop. Overridable by env var so the crop can be swept
# without a rebuild.
TOTALSEG_PYTHON = os.environ.get("TOTALSEG_PYTHON", "/opt/tsvenv/bin/python")
TOTALSEG_RUNNER = os.environ.get("TOTALSEG_RUNNER", "/app/totalseg_runner.py")
TOTALSEG_CROP = int(os.environ.get("TOTALSEG_CROP", "100"))


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
    predictor = autopet_main.build_predictor(
        str(model_dir), folds=(0,), device=device, use_mirroring=use_mirroring
    )
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


def start_ct_segmentation(
    fixed_ct: Path, moving_ct: Path, out_dir: Path, crop: int
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
            str(crop),
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
) -> torch.Tensor:
    """Refine the TOTAL field with Code/instance_opt.py:run_io.

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

    return instance_opt.run_io(
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
    )


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
    start = time.time()

    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device(args.device)
    cfg = build_config()

    # Launched before anything else so its interpreter startup and model loading
    # overlap with the GPU work below; joined just before IO, the first consumer.
    ts_proc = None
    ts_dir = args.ct_seg_dir or Path("/tmp/ct_labels")
    ts_launched = time.time()
    if args.totalseg:
        ts_proc = start_ct_segmentation(
            args.fixed_ct, args.moving_ct, ts_dir, TOTALSEG_CROP
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

    reg_seconds = time.time() - start
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
        totalseg_blocking = time.time() - join_start
        totalseg_seconds = time.time() - ts_launched

    if args.io and seg is None:
        print(
            "WARNING: no PET mask, so IO cannot run its tumour terms; "
            "writing the un-refined field",
            flush=True,
        )
    if args.io and seg is not None:
        io_start = time.time()
        transform_nearest = miccai2020_model_stage.SpatialTransformNearest_unit().to(device)
        total_unit_flow = run_instance_optimisation(
            total_unit_flow, X, Y, seg, ct_labels,
            transform, transform_nearest, grid, cfg, device,
        )
        io_seconds = time.time() - io_start
        print(
            f"instance optimisation {io_seconds:.1f}s "
            f"({cfg.io_it} steps @ lr {cfg.io_lr}, {io_seconds / max(cfg.io_it, 1):.2f}s/step)",
            flush=True,
        )

    save_disp(total_unit_flow, args.output_disp)
    print(
        f"wrote {args.output_disp}\n"
        f"  registration {reg_seconds:.1f}s | "
        f"PET seg {seg_seconds:.1f}s elapsed, {seg_blocking:.1f}s blocking | "
        f"IO {io_seconds:.1f}s | CT seg {totalseg_seconds:.1f}s elapsed, "
        f"{totalseg_blocking:.1f}s blocking (crop {TOTALSEG_CROP}, "
        f"z={288 - 2 * TOTALSEG_CROP}) | total {time.time() - start:.1f}s\n"
        f"  (PET and CT segmentation run concurrently with registration, so the "
        f"stages do not sum to the total; only their blocking parts add to it)",
        flush=True,
    )


if __name__ == "__main__":
    main()
