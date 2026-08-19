#!/usr/bin/env python3
"""Estimate PSMAReg displacement fields with NiftyReg reg_aladin + reg_f3d.

Writes the same submission-format fields as baseline_ConvexAdam.py:
predictions/disp_<id>_00_<id>_01.nii.gz, channel-first (3, X/f, Y/f, Z/f),
full-resolution voxel units, on the half-resolution grid. inference.py reads
that directory as its external_disp_dir.

Pipeline per case: bed removal (the ConvexAdam body mask, cached on disk) ->
reg_aladin (affine) -> reg_f3d initialised with that affine (-aff), so the
resulting control point grid encodes the TOTAL transform -> reg_transform -disp
to expand it to a dense displacement field in world mm -> convert mm to voxel
deltas -> downsample to the submission grid.
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

# --- paths: mirror inference.py ---
DATASET_ROOT = Path("/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset")
VAL_IMAGE_DIR = DATASET_ROOT / "imagesTs"
OUTPUT_DIR = Path(
    "/home/iml/fryderyk.koegl/code/LapIRN-koegl/submission_results/niftyreg"
)
CT_TEMPLATE = "PSMARegPSMA_{case_id}_0000_{tp}.nii.gz"

REG_ALADIN = Path("/home/iml/fryderyk.koegl/.local/bin/reg_aladin")
REG_F3D = Path("/home/iml/fryderyk.koegl/.local/bin/reg_f3d")
REG_TRANSFORM = Path("/home/iml/fryderyk.koegl/.local/bin/reg_transform")

# reg_f3d documents "-platf <uint>", reg_aladin "--platf <uint>". Same meaning:
# 0 = CPU, 1 = CUDA.
ALADIN_PLATFORM_FLAG = "--platf"
F3D_PLATFORM_FLAG = "-platf"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=DATASET_ROOT,
        help="Dataset root holding PSMAReg_dataset.json.",
    )
    parser.add_argument("--dataset-json", type=Path, default=None)
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=VAL_IMAGE_DIR,
        help="Directory the CT volumes are read from. Defaults to the same "
        "imagesTs used by inference.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Predictions land in <output-dir>/predictions, which is what "
        "inference.py reads as its external displacement directory.",
    )
    parser.add_argument("--reg-aladin", type=Path, default=REG_ALADIN)
    parser.add_argument("--reg-f3d", type=Path, default=REG_F3D)
    parser.add_argument(
        "--reg-transform",
        type=Path,
        default=REG_TRANSFORM,
        help="reg_transform executable, needed to expand the f3d control point "
        "grid into a dense displacement field.",
    )
    parser.add_argument(
        "--platform",
        type=int,
        default=1,
        help="NiftyReg platform: 0 = CPU, 1 = CUDA. Use -1 to omit the flag.",
    )
    parser.add_argument("--omp", type=int, default=None, help="Threads (-omp).")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--downsample-factor",
        type=int,
        default=2,
        help="Grid the field is stored on. Must match what inference.py expects "
        "(2 -> the half-resolution submission grid).",
    )
    parser.add_argument(
        "--stage",
        default="aladin-f3d",
        choices=["aladin", "aladin-f3d"],
        help="'aladin' gives the affine-only NiftyReg baseline.",
    )
    # --- reg_aladin ---
    parser.add_argument("--aladin-ln", type=int, default=3)
    parser.add_argument("--aladin-lp", type=int, default=3)
    parser.add_argument("--aladin-maxit", type=int, default=5)
    parser.add_argument("--aladin-rigid-only", action="store_true")
    # --- reg_f3d ---
    parser.add_argument("--f3d-be", type=float, default=0.005, help="Bending energy.")
    parser.add_argument(
        "--f3d-le", type=float, default=0.0, help="Linear elasticity penalty."
    )
    parser.add_argument(
        "--f3d-sx",
        type=float,
        default=-5.0,
        help="Control point spacing. Negative values are voxel counts.",
    )
    parser.add_argument("--f3d-ln", type=int, default=3)
    parser.add_argument("--f3d-lp", type=int, default=3)
    parser.add_argument("--f3d-maxit", type=int, default=300)
    parser.add_argument(
        "--f3d-lncc",
        type=float,
        default=None,
        help="Use LNCC with this kernel std instead of the default NMI.",
    )
    parser.add_argument(
        "--no-remove-bed",
        action="store_true",
        help="Register the raw CTs instead of the bed-removed ones. Bed removal "
        "reuses baseline_ConvexAdam's body mask, so both baselines see the same "
        "inputs.",
    )
    parser.add_argument(
        "--refresh-bed-cache",
        action="store_true",
        help="Recompute the bed-removed volumes even if cached.",
    )
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def resolve_reference_path(reference_dir, relative_path):
    if relative_path.startswith("./"):
        relative_path = relative_path[2:]
    return (reference_dir / relative_path).resolve()


def resolve_ct_path(args, subject_id, tp, fallback_relative):
    """CT path for one timepoint, from --image-dir (the imagesTs that
    inference.py uses). Falls back to the json-relative path if absent."""
    path = args.image_dir / CT_TEMPLATE.format(case_id=subject_id, tp=tp)
    if path.exists():
        return path
    return resolve_reference_path(args.reference_dir, fallback_relative)


def resolve_executable(path, name):
    """Accept an explicit path, else fall back to PATH."""
    if path is not None and Path(path).exists():
        return Path(path)
    found = shutil.which(Path(path).name if path is not None else name)
    if found is not None:
        return Path(found)
    raise FileNotFoundError(
        "{} not found (looked at {} and on PATH). It ships with the same "
        "NiftyReg build as reg_aladin / reg_f3d; point --{} at it.".format(
            name, path, name.replace("_", "-")
        )
    )


def load_remove_bed():
    """The ConvexAdam body mask, reused so both baselines register identical
    volumes. Imported lazily: baseline_ConvexAdam pulls in ants at import time,
    which we do not otherwise need."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from baseline_ConvexAdam import remove_bed

    return remove_bed


def preprocess_ct_path(args, ct_path, cache_dir):
    """Bed-removed copy of one CT, cached on disk — the mask is slow and the
    same volumes get re-registered whenever f3d parameters are retuned. Geometry
    is untouched, so the displacement field still lives on the original grid.
    Returns the path NiftyReg should register."""
    if args.no_remove_bed:
        return ct_path, 0.0
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / ct_path.name
    if out_path.exists() and not args.refresh_bed_cache:
        return out_path, 0.0
    start = time.time()
    source = nib.load(str(ct_path))
    cleaned = load_remove_bed()(source.get_fdata(dtype=np.float32))
    image = nib.Nifti1Image(cleaned.astype(np.float32), source.affine, source.header)
    image.set_data_dtype(np.float32)
    nib.save(image, str(out_path))
    return out_path, time.time() - start


def run_command(command, log_path):
    """Run one NiftyReg call, tee-ing its output into a per-case log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(str(part) for part in command)
    result = subprocess.run(
        [str(part) for part in command],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("$ {}\n{}\n".format(printable, result.stdout))
    if result.returncode != 0:
        raise RuntimeError(
            "Command failed ({}): {}\n{}".format(
                result.returncode, printable, result.stdout[-4000:]
            )
        )


def platform_args(flag, platform):
    return [] if platform < 0 else [flag, str(platform)]


def run_aladin(args, fixed_path, moving_path, affine_path, resampled_path, log_path):
    command = [
        args.reg_aladin,
        "-ref",
        fixed_path,
        "-flo",
        moving_path,
        "-aff",
        affine_path,
        "-res",
        resampled_path,
        "-ln",
        str(args.aladin_ln),
        "-lp",
        str(args.aladin_lp),
        "-maxit",
        str(args.aladin_maxit),
    ]
    if args.aladin_rigid_only:
        command.append("-rigOnly")
    command += platform_args(ALADIN_PLATFORM_FLAG, args.platform)
    if args.omp is not None:
        command += ["-omp", str(args.omp)]
    run_command(command, log_path)


def run_f3d(args, fixed_path, moving_path, affine_path, cpp_path, res_path, log_path):
    command = [
        args.reg_f3d,
        "-ref",
        fixed_path,
        "-flo",
        moving_path,
        # initialise with the aladin affine, so the control point grid that comes
        # out encodes affine + deformable together — the total transform, which
        # is what our submission format stores.
        "-aff",
        affine_path,
        "-cpp",
        cpp_path,
        "-res",
        res_path,
        "-be",
        str(args.f3d_be),
        "-le",
        str(args.f3d_le),
        "-sx",
        str(args.f3d_sx),
        "-ln",
        str(args.f3d_ln),
        "-lp",
        str(args.f3d_lp),
        "-maxit",
        str(args.f3d_maxit),
    ]
    if args.f3d_lncc is not None:
        command += ["-lncc", str(args.f3d_lncc)]
    command += platform_args(F3D_PLATFORM_FLAG, args.platform)
    if args.omp is not None:
        command += ["-omp", str(args.omp)]
    run_command(command, log_path)


def convert_transformation_to_displacement_field(
    args, transformation_path, fixed_path, output_path, log_path
):
    """reg_transform -disp: expand a NiftyReg transform (affine .txt or control
    point grid .nii.gz) into a dense displacement field on the reference grid.
    Values are world-space millimetres."""
    command = [
        args.reg_transform,
        "-ref",
        fixed_path,
        "-disp",
        transformation_path,
        output_path,
    ]
    run_command(command, log_path)
    if not output_path.exists():
        raise RuntimeError("reg_transform produced no output: {}".format(output_path))
    return output_path


def world_mm_to_voxel_disp(disp_mm, fixed_nii, moving_nii):
    """World-mm displacements on the fixed grid -> voxel deltas in our
    convention: moving_index = fixed_index + u, u along the array axes.

    A NiftyReg displacement field holds P_flo(x) - P_ref(x) in millimetres. With
    voxel->world affines A_ref / A_flo, the moving index is
        j = A_flo^-1 (A_ref i + t_ref - t_flo + disp_mm),
    so u = j - i. When both images share an affine (the PSMAReg case, both
    timepoints are resampled to a common grid) this collapses to
    u = A^-1 disp_mm, which avoids materialising an index grid."""
    ref_affine = np.asarray(fixed_nii.affine, dtype=np.float64)
    flo_affine = np.asarray(moving_nii.affine, dtype=np.float64)
    flo_linear_inv = np.linalg.inv(flo_affine[:3, :3])

    flat = disp_mm.reshape(-1, 3).astype(np.float64)
    if np.allclose(ref_affine, flo_affine, atol=1e-4):
        voxel = flat.dot(flo_linear_inv.T)
    else:
        shape = disp_mm.shape[:3]
        grids = np.meshgrid(
            np.arange(shape[0], dtype=np.float64),
            np.arange(shape[1], dtype=np.float64),
            np.arange(shape[2], dtype=np.float64),
            indexing="ij",
        )
        index = np.stack(grids, axis=-1).reshape(-1, 3)
        world = index.dot(ref_affine[:3, :3].T) + ref_affine[:3, 3]
        moved = world + flat - flo_affine[:3, 3]
        voxel = moved.dot(flo_linear_inv.T) - index
    return voxel.reshape(disp_mm.shape).astype(np.float32)


def load_displacement_mm(path):
    """reg_transform writes (X, Y, Z, 1, 3) vector-intent NIfTI."""
    field = nib.load(str(path)).get_fdata(dtype=np.float32)
    field = np.squeeze(field)
    if field.ndim != 4 or field.shape[-1] != 3:
        raise ValueError(
            "Expected a (X, Y, Z, 3) displacement field, got {}".format(field.shape)
        )
    return field


def downsample_field(field_ch_first, factor):
    """Subsample the grid, keep the magnitudes: the submission field lives on the
    coarse grid but is expressed in full-resolution voxel units (the same thing
    inference.py's save_disp does with scale_factor=0.5)."""
    if factor == 1:
        return field_ch_first
    tensor = torch.from_numpy(field_ch_first).unsqueeze(0).float()
    with torch.no_grad():
        low = F.interpolate(
            tensor,
            scale_factor=1.0 / factor,
            mode="trilinear",
            align_corners=False,
        )
    return low.squeeze(0).numpy().astype(np.float32)


def save_disp(path, field_ch_first, affine, spacing, factor):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(field_ch_first.astype(np.float32), affine)
    image.header["pixdim"][1:4] = [float(s) * factor for s in spacing]
    nib.save(image, str(path))


def select_entries(dataset, subjects, limit):
    entries = list(dataset.get("validation_paired", []))
    if subjects:
        subject_set = set(subjects)
        normalized = set(
            "PSMARegPSMA_{}".format(s) if s.isdigit() else s for s in subject_set
        )
        entries = [
            entry
            for entry in entries
            if entry["subject"] in normalized
            or entry["subject"].split("_")[-1] in subject_set
        ]
    if limit is not None:
        entries = entries[:limit]
    return entries


def run_case(entry, args, prediction_dir, temp_root, log_dir):
    subject = entry["subject"]
    subject_id = subject.split("_")[-1]
    prediction_path = prediction_dir / "disp_{0}_00_{0}_01.nii.gz".format(subject_id)
    if prediction_path.exists() and args.skip_existing:
        return {
            "subject": subject,
            "status": "skipped",
            "prediction": str(prediction_path),
        }
    if prediction_path.exists() and not args.overwrite:
        raise FileExistsError("Prediction exists: {}".format(prediction_path))

    t0 = time.time()
    # fixed = baseline (00), moving = follow-up (01) — same direction as
    # inference.py (X = 01 moving, Y = 00 fixed).
    fixed_path = resolve_ct_path(args, subject_id, "00", entry["Baseline CT"])
    moving_path = resolve_ct_path(args, subject_id, "01", entry["Follow-up 01 CT"])
    fixed_nii = nib.load(str(fixed_path))
    moving_nii = nib.load(str(moving_path))
    fullres_spacing = tuple(float(s) for s in fixed_nii.header.get_zooms()[:3])
    fullres_shape = tuple(int(dim) for dim in fixed_nii.shape[:3])

    log_path = log_dir / "{}.log".format(subject)
    if log_path.exists():
        log_path.unlink()
    temp_dir = temp_root / subject
    temp_dir.mkdir(parents=True, exist_ok=True)

    # everything below registers the bed-removed volumes; fixed_nii / fixed_path
    # stay the originals, and the grid is shared, so the geometry used for the
    # mm -> voxel conversion is unaffected.
    bed_cache_dir = args.output_dir / "bed_removed"
    fixed_reg_path, fixed_bed_seconds = preprocess_ct_path(
        args, fixed_path, bed_cache_dir
    )
    moving_reg_path, moving_bed_seconds = preprocess_ct_path(
        args, moving_path, bed_cache_dir
    )
    bed_seconds = fixed_bed_seconds + moving_bed_seconds

    affine_path = temp_dir / "affine.txt"
    aladin_start = time.time()
    run_aladin(
        args,
        fixed_reg_path,
        moving_reg_path,
        affine_path,
        temp_dir / "aladin_res.nii.gz",
        log_path,
    )
    aladin_seconds = time.time() - aladin_start

    f3d_seconds = 0.0
    if args.stage == "aladin-f3d":
        transform_path = temp_dir / "cpp.nii.gz"
        f3d_start = time.time()
        run_f3d(
            args,
            fixed_reg_path,
            moving_reg_path,
            affine_path,
            transform_path,
            temp_dir / "f3d_res.nii.gz",
            log_path,
        )
        f3d_seconds = time.time() - f3d_start
    else:
        transform_path = affine_path

    disp_mm_path = convert_transformation_to_displacement_field(
        args, transform_path, fixed_reg_path, temp_dir / "disp_mm.nii.gz", log_path
    )
    disp_mm = load_displacement_mm(disp_mm_path)
    if tuple(disp_mm.shape[:3]) != fullres_shape:
        raise ValueError(
            "Displacement field {} does not match the reference grid {}".format(
                disp_mm.shape[:3], fullres_shape
            )
        )

    disp_voxel = world_mm_to_voxel_disp(disp_mm, fixed_nii, moving_nii)
    total_field = np.moveaxis(disp_voxel, -1, 0)
    total_field = downsample_field(total_field, args.downsample_factor)

    # inference.py loads this field on the half-resolution grid it composes on,
    # so the spatial dims must be exactly fixed_shape // downsample_factor.
    expected_shape = tuple(dim // args.downsample_factor for dim in fullres_shape)
    if tuple(total_field.shape[1:]) != expected_shape:
        raise ValueError(
            "Field shape {} != expected {} for {}".format(
                tuple(total_field.shape[1:]), expected_shape, subject
            )
        )

    save_disp(
        prediction_path,
        total_field,
        fixed_nii.affine,
        fullres_spacing,
        args.downsample_factor,
    )
    if not args.keep_temp:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return {
        "subject": subject,
        "subject_id": subject_id,
        "status": "ok",
        "prediction": str(prediction_path),
        "stage": args.stage,
        "bed_removed": not args.no_remove_bed,
        "bed_seconds": bed_seconds,
        "aladin_seconds": aladin_seconds,
        "f3d_seconds": f3d_seconds,
        "total_seconds": time.time() - t0,
        "field_shape": "x".join(str(dim) for dim in total_field.shape[1:]),
        "field_abs_mean": float(np.mean(np.abs(total_field))),
        "field_abs_max": float(np.max(np.abs(total_field))),
    }


def write_rows(path, rows):
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-psmareg")

    # resolve every executable up front: a missing reg_transform should not
    # surface after an hour of registrations.
    args.reg_aladin = resolve_executable(args.reg_aladin, "reg_aladin")
    args.reg_f3d = resolve_executable(args.reg_f3d, "reg_f3d")
    args.reg_transform = resolve_executable(args.reg_transform, "reg_transform")

    dataset_json = args.dataset_json or args.reference_dir / "PSMAReg_dataset.json"
    with dataset_json.open("r", encoding="utf-8") as handle:
        dataset = json.load(handle)

    prediction_dir = args.output_dir / "predictions"
    temp_root = args.output_dir / "temp"
    log_dir = args.output_dir / "logs"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    temp_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, default=str), encoding="utf-8"
    )

    entries = select_entries(dataset, args.subjects, args.limit)
    rows = []
    for index, entry in enumerate(entries, start=1):
        try:
            row = run_case(entry, args, prediction_dir, temp_root, log_dir)
        except Exception as error:
            row = {
                "subject": entry.get("subject", ""),
                "status": "error",
                "error": repr(error),
            }
            print(
                "[{}/{}] ERROR {}".format(index, len(entries), json.dumps(row)),
                flush=True,
            )
            traceback.print_exc()
            raise
        rows.append(row)
        write_rows(args.output_dir / "case_results.csv", rows)
        print("[{}/{}] {}".format(index, len(entries), json.dumps(row)), flush=True)

    print("Done. Predictions: {}".format(prediction_dir))


if __name__ == "__main__":
    sys.exit(main())
