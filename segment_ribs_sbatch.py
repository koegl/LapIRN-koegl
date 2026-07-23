"""Batch rib segmentation over the PSMAReg CT volumes, sharded for a Slurm job array.

Builds the work list deterministically from split.json (train + val, IDs starting
with "0", CT channel _0000 only, all sessions), then processes the slice
``files[shard::num_shards]`` so that the array tasks together cover every file
exactly once.

The nnU-Net predictor is built once per job and reused for every volume.
Volumes whose output already exists are skipped, so a timed-out shard can simply
be resubmitted.

Usage:
    python segment_ribs_sbatch.py --shard $SLURM_ARRAY_TASK_ID --num-shards 5
"""

import argparse
import json
import shutil
import tempfile
import time
import traceback
from pathlib import Path

import SimpleITK as sitk
import torch

from segment_ribs import (
    MODEL_DIR,
    build_predictor,
    get_orientation,
    load_inference_config,
)

SPLIT_JSON = Path("/home/iml/fryderyk.koegl/code/LapIRN-koegl/split.json")
IMAGES_DIR = Path("/lustre/groups/iml/data/PSMAReg/PSMAReg_dataset/imagesTr")
LABELS_DIR = Path("/lustre/groups/iml/data/PSMAReg/PSMAReg_dataset/labelsTr_ribs")
CT_CHANNEL = "0000"


def collect_files(split_json, images_dir):
    """All CT volumes of the train+val IDs that start with '0', sorted."""
    with open(split_json) as f:
        split = json.load(f)

    case_ids = sorted(i for i in split["train"] + split["val"] if i.startswith("0"))

    files = []
    for case_id in case_ids:
        matches = sorted(
            images_dir.glob(f"PSMARegPSMA_{case_id}_{CT_CHANNEL}_*.nii.gz")
        )
        if not matches:
            print(f"WARNING: no CT volumes found for case {case_id}", flush=True)
        files.extend(matches)
    return files


def segment_one(predictor, image_path, output_path, target_orientation):
    """Reorient to the model orientation, predict, and map the mask back."""
    image = sitk.ReadImage(str(image_path))
    original_orientation = get_orientation(image)
    reoriented = sitk.DICOMOrient(image, target_orientation)

    tmp_dir = Path(tempfile.mkdtemp(prefix="ribseg_"))
    try:
        in_dir = tmp_dir / "in"
        out_dir = tmp_dir / "out"
        in_dir.mkdir()
        out_dir.mkdir()

        sitk.WriteImage(reoriented, str(in_dir / "case_0000.nii.gz"))

        predictor.predict_from_files(
            str(in_dir),
            str(out_dir),
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=2,
            num_processes_segmentation_export=2,
        )

        prediction = sitk.ReadImage(str(out_dir / "case.nii.gz"))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    prediction = sitk.DICOMOrient(prediction, original_orientation)
    prediction = sitk.Cast(prediction, sitk.sitkUInt8)
    if prediction.GetSize() != image.GetSize():
        raise RuntimeError(
            f"Prediction size {prediction.GetSize()} does not match input size {image.GetSize()}"
        )
    prediction.CopyInformation(image)

    # write to a temporary name first so an interrupted job never leaves a
    # truncated file that the skip-existing check would accept
    partial_path = output_path.with_name(
        output_path.name.replace(".nii.gz", ".partial.nii.gz")
    )
    sitk.WriteImage(prediction, str(partial_path))
    partial_path.replace(output_path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shard", type=int, required=True, help="index of this array task"
    )
    parser.add_argument(
        "--num-shards", type=int, default=5, help="total number of array tasks"
    )
    parser.add_argument("--split-json", type=Path, default=SPLIT_JSON)
    parser.add_argument("--images-dir", type=Path, default=IMAGES_DIR)
    parser.add_argument("--output-dir", type=Path, default=LABELS_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="only print the files this shard would process",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 <= args.shard < args.num_shards:
        raise ValueError(f"shard {args.shard} outside of [0, {args.num_shards})")

    all_files = collect_files(args.split_json, args.images_dir)
    shard_files = all_files[args.shard :: args.num_shards]
    print(
        f"shard {args.shard}/{args.num_shards}: {len(shard_files)} of {len(all_files)} volumes",
        flush=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    todo = [f for f in shard_files if not (args.output_dir / f.name).is_file()]
    print(
        f"{len(shard_files) - len(todo)} already segmented, {len(todo)} to do",
        flush=True,
    )

    if args.dry_run:
        for f in shard_files:
            state = "done" if (args.output_dir / f.name).is_file() else "todo"
            print(f"  [{state}] {f.name}")
        return

    if not todo:
        print("nothing to do", flush=True)
        return

    cfg = load_inference_config(args.model_dir)
    target_orientation = "".join(cfg["model_expected_orientation"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    predictor = build_predictor(
        model_dir=args.model_dir,
        folds=tuple(args.folds),
        step_size=cfg.get("default_step_size", 0.5),
        use_mirroring=bool(cfg.get("inference_augmentation", 1)),
        device=device,
    )

    failures = []
    for i, image_path in enumerate(todo, start=1):
        output_path = args.output_dir / image_path.name
        start = time.time()
        print(f"[{i}/{len(todo)}] {image_path.name}", flush=True)
        try:
            segment_one(predictor, image_path, output_path, target_orientation)
            print(
                f"    saved in {time.time() - start:.1f}s -> {output_path}", flush=True
            )
        except Exception:
            failures.append(image_path.name)
            print(f"    FAILED after {time.time() - start:.1f}s", flush=True)
            traceback.print_exc()

    print(
        f"finished: {len(todo) - len(failures)} succeeded, {len(failures)} failed",
        flush=True,
    )
    for name in failures:
        print(f"  failed: {name}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
