"""Baseline registrations (NiftyReg, ConvexAdam) for the journal evaluation.

Shared by tune_niftyreg.py and tune_convexadam.py. Field generation reuses the
per-case code of Code/baseline_NiftyReg.py and Code/baseline_ConvexAdam.py,
generalised to any pair of the journal split (follow-up -> baseline, imagesTr).

Every field is the TOTAL transform (the method's own affine included), stored
channel-first, channel i along array axis i, in full-resolution voxels, as
<BASELINE_DIR>/<method>/<config>/fields/disp_<case>_<tp moving>_<tp fixed>.nii.gz
(the layout evaluate.py reads).

Tuning: every config of a grid registers all val pairs, evaluate.py scores them,
and the configs are ranked against each other with the challenge's
significance ranking (ranking.py); the winner goes to chosen.json. With split
"test" only the chosen config is registered (evaluate.py runs B3 / B4).
"""

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import torch

CODE_DIR = Path(__file__).resolve().parents[1] / "Code"
sys.path.insert(0, str(CODE_DIR))

import baseline_ConvexAdam as convexadam  # noqa: E402
import baseline_NiftyReg as niftyreg  # noqa: E402
import config  # noqa: E402
import evaluate  # noqa: E402
import ranking  # noqa: E402

BASELINE_DIR = evaluate.BASELINE_DIR
CT_TEMPLATE = "PSMARegPSMA_{case_id}_0000_{tp}.nii.gz"


def ct_paths(cfg: config.TrainingConfig, case_id: str, tp_x: str, tp_y: str) -> Tuple[Path, Path]:
    image_dir = cfg.data_dir / "imagesTr"
    return (
        image_dir / CT_TEMPLATE.format(case_id=case_id, tp=tp_y),
        image_dir / CT_TEMPLATE.format(case_id=case_id, tp=tp_x),
    )


def save_field(path: Path, field: np.ndarray, reference: nib.Nifti1Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(field.astype(np.float32), reference.affine), str(path))


# --- NiftyReg: reg_aladin -> reg_f3d on the GPU (-platf 1, NMI) ---

NIFTYREG_DEFAULTS = dict(
    reg_aladin=niftyreg.resolve_executable(niftyreg.REG_ALADIN, "reg_aladin"),
    reg_f3d=niftyreg.resolve_executable(niftyreg.REG_F3D, "reg_f3d"),
    reg_transform=niftyreg.resolve_executable(niftyreg.REG_TRANSFORM, "reg_transform"),
    platform=1,  # CUDA; NMI is the only similarity it supports (no LNCC)
    omp=None,
    aladin_ln=3,
    aladin_lp=3,
    aladin_maxit=5,
    aladin_rigid_only=False,
    f3d_be=0.005,
    f3d_le=0.0,
    f3d_sx=-5.0,
    f3d_ln=3,
    f3d_lp=3,
    f3d_maxit=300,
    f3d_lncc=None,
    no_remove_bed=False,
    refresh_bed_cache=False,
)


def niftyreg_field(
    cfg: config.TrainingConfig, case_id: str, tp_x: str, tp_y: str, params: Dict, work_dir: Path
) -> np.ndarray:
    """(3, H, W, D) total displacement of one pair, full resolution."""
    args = SimpleNamespace(**{**NIFTYREG_DEFAULTS, **params})
    fixed_path, moving_path = ct_paths(cfg, case_id, tp_x, tp_y)
    fixed_nii, moving_nii = nib.load(str(fixed_path)), nib.load(str(moving_path))

    # bed removal is cached across configs (only the f3d parameters change)
    bed_dir = BASELINE_DIR / "niftyreg" / "bed_removed"
    fixed_reg, _ = niftyreg.preprocess_ct_path(args, fixed_path, bed_dir)
    moving_reg, _ = niftyreg.preprocess_ct_path(args, moving_path, bed_dir)

    work_dir.mkdir(parents=True, exist_ok=True)
    log_path = work_dir / "niftyreg.log"
    affine_path = work_dir / "affine.txt"
    niftyreg.run_aladin(
        args, fixed_reg, moving_reg, affine_path, work_dir / "aladin_res.nii.gz", log_path
    )
    cpp_path = work_dir / "cpp.nii.gz"
    niftyreg.run_f3d(
        args, fixed_reg, moving_reg, affine_path, cpp_path, work_dir / "f3d_res.nii.gz", log_path
    )
    disp_mm_path = niftyreg.convert_transformation_to_displacement_field(
        args, cpp_path, fixed_reg, work_dir / "disp_mm.nii.gz", log_path
    )
    disp_voxel = niftyreg.world_mm_to_voxel_disp(
        niftyreg.load_displacement_mm(disp_mm_path), fixed_nii, moving_nii
    )
    return np.moveaxis(disp_voxel, -1, 0)


# --- ConvexAdam: ANTs affine -> ConvexAdam MIND/SVF, half resolution ---

CONVEXADAM_DEFAULTS = dict(
    no_remove_bed=False,
    downsample_factor=2,
    ct_window=[-300.0, 300.0],
    ants_transform="Affine",
    aff_metric="mattes",
    aff_sampling=64,
    convex_mind_r=1,
    convex_mind_d=2,
    convex_lambda=2.0,
    convex_grid_sp=4,
    convex_disp_hw=4,
    convex_niter=80,
    convex_smooth=3,
    convex_grid_sp_adam=2,
    svf_steps=7,
    ic=True,
)
convexadam.ensure_mir(Path("/home/iml/fryderyk.koegl/code/MIR/src"))


def convexadam_field(
    cfg: config.TrainingConfig, case_id: str, tp_x: str, tp_y: str, params: Dict, work_dir: Path
) -> np.ndarray:
    """(3, H/2, W/2, D/2) total displacement of one pair in full-resolution
    voxels (ConvexAdam's native half-resolution grid)."""
    args = SimpleNamespace(**{**CONVEXADAM_DEFAULTS, **params})
    fixed_path, moving_path = ct_paths(cfg, case_id, tp_x, tp_y)
    fixed_nii, moving_nii = nib.load(str(fixed_path)), nib.load(str(moving_path))
    spacing = tuple(float(s) for s in fixed_nii.header.get_zooms()[:3])

    def lowres(nii):
        volume = convexadam.preprocess_ct(nii.get_fdata(dtype=np.float32), args)
        return convexadam.make_lowres_ants_image(
            volume, spacing, args.downsample_factor, args.ct_window
        )

    fixed_ants, moving_ants = lowres(fixed_nii), lowres(moving_nii)
    tx = convexadam.ants.registration(
        fixed=fixed_ants,
        moving=moving_ants,
        type_of_transform=args.ants_transform,
        aff_metric=args.aff_metric,
        aff_sampling=args.aff_sampling,
        verbose=False,
    )
    affine_vox = convexadam.ants_affine_to_fullres_voxel_disp(
        convexadam.ants.read_transform(tx["fwdtransforms"][0]), fixed_ants, spacing
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deform, _ = convexadam.convex_adam_pt_svf(
        img_fixed=torch.from_numpy(fixed_ants.numpy().astype(np.float32)),
        img_moving=torch.from_numpy(tx["warpedmovout"].numpy().astype(np.float32)),
        mind_r=args.convex_mind_r,
        mind_d=args.convex_mind_d,
        lambda_weight=args.convex_lambda,
        grid_sp=args.convex_grid_sp,
        disp_hw=args.convex_disp_hw,
        selected_niter=args.convex_niter,
        selected_smooth=args.convex_smooth,
        grid_sp_adam=args.convex_grid_sp_adam,
        ic=args.ic,
        svf_steps=args.svf_steps,
        dtype=torch.float32,
        device=device,
        verbose=False,
        save_disp=True,
    )
    deform_vox = np.asarray(deform, dtype=np.float32) * float(args.downsample_factor)
    return convexadam.compose_deform_after_affine(deform_vox, affine_vox)


METHODS = {"niftyreg": niftyreg_field, "convexadam": convexadam_field}


# --- tuning ---


def register_split(method: str, config_name: str, params: Dict, split: str) -> Path:
    """Fields of one config for every pair of the split (existing ones kept)."""
    cfg = config.TrainingConfig()
    config_dir = BASELINE_DIR / method / config_name
    field_dir = config_dir / "fields"
    timings_csv = config_dir / f"timings_{split}.csv"
    timings = pd.read_csv(timings_csv).to_dict("records") if timings_csv.exists() else []
    (config_dir / "params.json").parent.mkdir(parents=True, exist_ok=True)
    (config_dir / "params.json").write_text(json.dumps(params, indent=2))

    pairs = evaluate.split_pairs(cfg, split)
    for i, (case_id, tp_x, tp_y) in enumerate(pairs):
        path = evaluate.baseline_field_path(field_dir, case_id, tp_x, tp_y)
        if path.exists():
            continue
        start = time.time()
        field = METHODS[method](
            cfg, case_id, tp_x, tp_y, params, config_dir / "work" / f"{case_id}_{tp_x}"
        )
        seconds = time.time() - start
        save_field(path, field, nib.load(str(ct_paths(cfg, case_id, tp_x, tp_y)[0])))
        timings.append({"case_id": case_id, "tp_moving": tp_x, "tp_fixed": tp_y, "seconds": seconds})
        pd.DataFrame(timings).to_csv(timings_csv, index=False)
        print(f"[{method}/{config_name}] {i + 1}/{len(pairs)} {case_id} {tp_x}->{tp_y}: {seconds:.0f}s", flush=True)
    return field_dir


def tune(method: str, grid: Dict[str, Dict], split: str) -> None:
    method_dir = BASELINE_DIR / method
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if split == "val":
        results: Dict[str, pd.DataFrame] = {}
        for name, params in grid.items():
            field_dir = register_split(method, name, params, "val")
            csv = evaluate.evaluate_run(
                name,
                "val",
                device,
                run=evaluate.EvalRun(external=field_dir),
                out_dir=method_dir / "eval",
            )
            results[name] = pd.read_csv(csv)
        selection = ranking.select(results)
        chosen = {"config": selection["chosen"], "table": selection["table"]}
        (method_dir / "chosen.json").write_text(json.dumps(chosen, indent=2, default=str))
        print(pd.DataFrame(chosen["table"]).round(4).to_string(index=False))
        print(f"chosen: {chosen['config']}")
    elif split == "test":
        with open(method_dir / "chosen.json") as f:
            name = json.load(f)["config"]
        register_split(method, name, grid[name], "test")
        print(f"test fields of {name} written; evaluate with evaluate.py ({method})")
    else:
        raise ValueError(split)


def main(method: str, grid: Dict[str, Dict]) -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="val")
    tune(method, grid, parser.parse_args().split)
