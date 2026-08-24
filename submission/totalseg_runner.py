"""TotalSegmentator CT labels for one fixed/moving pair, cropped in z.

Runs in the ISOLATED interpreter at /opt/tsvenv, not the container's main one.
TotalSegmentator 2.17 imports `custom_trainers` at module level, which needs
`infer_dataset_class` -- a symbol the autopet nnunetv2 fork (2.5.1) does not
have. The fork cannot be dropped either: it is the only place
nnUNetTrainer_PGPSplus exists, so the PET lesion model needs it. The two are
therefore installed into separate environments and this half is invoked as a
subprocess.

    /opt/tsvenv/bin/python totalseg_runner.py <fixed_ct> <moving_ct> <out_dir> <crop>

`crop` is the number of slices removed from EACH end of the z axis, so the
segmented volume is (288 - 2*crop) deep. Labels are written back into the full
grid, zero outside the crop, so the caller always gets a full-resolution volume
on the fixed image's geometry.

Weight loading is memoised across calls (see cache_model_loading): TotalSegmentator
builds a fresh nnUNetPredictor per volume, and with body_seg that is four model
loads per pair for two distinct models. Loading dominates the runtime at small
crops -- ~23 s of the measured 32.3 s at z=108 -- so caching it is worth more
than cropping further.
"""

import contextlib
import json
import os
import sys
import time
from pathlib import Path

# Written rather than baked: TotalSegmentator increments a prediction counter in
# this file, so the directory has to be writable, and the container's /app is
# not. send_usage_stats must be off -- the evaluation runs with --network=none
# and the stats POST would fail or stall.
TOTALSEG_HOME = Path(os.environ.get("TOTALSEG_HOME_DIR", "/tmp/totalseg"))
TOTALSEG_HOME.mkdir(parents=True, exist_ok=True)
config_file = TOTALSEG_HOME / "config.json"
if not config_file.exists():
    config_file.write_text(
        json.dumps(
            {
                "totalseg_id": "totalseg_container",
                "send_usage_stats": False,
                "prediction_counter": 0,
                "statistics_disclaimer_shown": True,
            },
            indent=4,
        )
    )

sys.path.insert(0, os.environ.get("LAPIRN_REPO", "/app/lapirn"))

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402

# The crop/restore geometry is shared with the timing sweep rather than
# reimplemented, so the container crops exactly what time_totalsegmentator.py
# measured.
from time_totalsegmentator import crop_axial, restore_full  # noqa: E402
from totalsegmentator import python_api  # noqa: E402


@contextlib.contextmanager
def suppress_output():
    """Silence stdout/stderr from TotalSegmentator and the nnU-Net underneath.

    Only this runner's own timing line should reach the container log -- the
    prediction chatter makes a 20-case sweep unreadable. Errors still surface:
    a non-zero exit code is what infer.py checks, and the traceback is printed
    after the context exits.
    """
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def cache_model_loading() -> None:
    """Memoise nnUNetPredictor weight loading, keyed by (folder, folds, checkpoint).

    TotalSegmentator constructs its predictors deep inside nnUNet_predict, so
    there is no seam to pass one in. Patching the loader instead leaves its
    pipeline untouched: on a repeat it copies the eight attributes
    initialize_from_trained_model_folder would have set, skipping the checkpoint
    read and the network build.

    Sharing one network module between predictors is safe here -- predictions are
    sequential and the module is only ever used for inference.
    """
    from nnunetv2.inference import predict_from_raw_data

    predictor_cls = predict_from_raw_data.nnUNetPredictor
    original = predictor_cls.initialize_from_trained_model_folder
    attrs = (
        "plans_manager",
        "configuration_manager",
        "list_of_parameters",
        "network",
        "dataset_json",
        "trainer_name",
        "allowed_mirroring_axes",
        "label_manager",
    )
    cache = {}

    def cached(self, model_training_output_dir, use_folds=None, checkpoint_name="checkpoint_final.pth"):
        key = (str(model_training_output_dir), str(use_folds), checkpoint_name)
        if key in cache:
            for name, value in zip(attrs, cache[key]):
                setattr(self, name, value)
            return
        original(self, model_training_output_dir, use_folds=use_folds, checkpoint_name=checkpoint_name)
        cache[key] = tuple(getattr(self, name) for name in attrs)

    predictor_cls.initialize_from_trained_model_folder = cached


def segment_one(img_path: Path, out_path: Path, crop: int, work_dir: Path) -> float:
    img = nib.load(str(img_path))
    full_shape = img.shape

    cropped, z_start, z_end = crop_axial(img, crop)
    crop_path = work_dir / f"{out_path.stem}_crop.nii.gz"
    seg_path = work_dir / f"{out_path.stem}_seg.nii.gz"
    nib.save(cropped, str(crop_path))

    start = time.time()
    # quiet=True drops TotalSegmentator's banners and per-stage chatter; the
    # nnU-Net tile progress bars underneath it are silenced by the redirect.
    with suppress_output():
        python_api.totalsegmentator(
            str(crop_path),
            str(seg_path),
            ml=True,
            task="total",
            fast=True,
            body_seg=True,
            quiet=True,
        )
    elapsed = time.time() - start

    seg_img = nib.load(str(seg_path))
    restored = restore_full(np.asarray(seg_img.dataobj), full_shape, z_start, z_end)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(restored, img.affine, seg_img.header), str(out_path))
    return elapsed


def main() -> None:
    fixed_ct, moving_ct, out_dir, crop = (
        Path(sys.argv[1]),
        Path(sys.argv[2]),
        Path(sys.argv[3]),
        int(sys.argv[4]),
    )
    work_dir = Path(os.environ.get("TOTALSEG_WORK_DIR", "/tmp/totalseg_work"))
    work_dir.mkdir(parents=True, exist_ok=True)

    if os.environ.get("TOTALSEG_CACHE_MODELS", "1") != "0":
        cache_model_loading()

    total = 0.0
    for src in (fixed_ct, moving_ct):
        total += segment_one(src, out_dir / src.name, crop, work_dir)
    z = 288 - 2 * crop
    print(f"totalsegmentator: {total:.1f}s for 2 volumes at z={z} (crop={crop})", flush=True)


if __name__ == "__main__":
    main()
