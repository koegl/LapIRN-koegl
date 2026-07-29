"""Segment the spine (individual vertebrae) in a CT image with the CTSpine1K
nnU-Net weights in ``ctspine1k/``.

Unlike ``data_klinikum_segment_ribs.py`` (which uses an nnU-Net **v2** model), the
CTSpine1K weights are an nnU-Net **v1** checkpoint trained on ``Task056_VerSe``
(``nnunet.training.network_training.nnUNetTrainerV2``). The label map has 25
classes: 0 = background and 1..24 = the vertebrae C1..L5 (VerSe convention).

The weights ship as two loose files (``model_final_checkpoint.model`` and its
``.model.pkl``), whereas nnU-Net v1 inference expects a model *folder* laid out as::

    <model>/plans.pkl
    <model>/all/model_final_checkpoint.model
    <model>/all/model_final_checkpoint.model.pkl

The ``.model.pkl`` is self-contained (it embeds the full ``plans`` dict), so we
rebuild that folder in a temp directory: the big weights are symlinked, and
``plans.pkl`` is extracted from the embedded plans.

Two environment quirks are handled up front, before nnU-Net v1 is imported:
  * torch >= 2.6 defaults ``torch.load(weights_only=True)``, which cannot unpickle
    a v1 checkpoint -> we restore the old ``weights_only=False`` default.
  * numpy >= 2 removed the ``np.int``/``np.float``/... aliases that v1 still uses.

nnU-Net v1 resamples to the model spacing and applies the plans' transpose
internally, and writes the prediction with the input image's original geometry, so
(unlike the rib script) no manual reorientation is required here.
"""

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

# --- numpy >= 2 compatibility shims for nnU-Net v1 -------------------------------
for _alias, _pytype in (
    ("int", int),
    ("float", float),
    ("bool", bool),
    ("object", object),
    ("str", str),
    ("complex", complex),
):
    if not hasattr(np, _alias):
        setattr(np, _alias, _pytype)
if not hasattr(np, "float_"):
    np.float_ = np.float64
if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_

# --- torch >= 2.6 loads v1 checkpoints (numpy/pickle payload, not just tensors) --
_ORIGINAL_TORCH_LOAD = torch.load


def _torch_load_weights_only_false(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _ORIGINAL_TORCH_LOAD(*args, **kwargs)


torch.load = _torch_load_weights_only_false

# nnU-Net v1 imports must come after the shims above.
from batchgenerators.utilities.file_and_folder_operations import (  # noqa: E402
    load_pickle,
    write_pickle,
)
from nnunet.inference.predict import predict_from_folder  # noqa: E402

MODEL_DIR = Path(__file__).parent / "ctspine1k"
CHECKPOINT_NAME = "model_final_checkpoint"


def build_v1_model_folder(model_dir, dest):
    """Recreate the folder layout nnU-Net v1 inference expects from the two loose
    CTSpine1K files, into ``dest``. Returns the model folder to pass to
    ``predict_from_folder``."""
    model_dir = Path(model_dir)
    checkpoint = model_dir / f"{CHECKPOINT_NAME}.model"
    checkpoint_pkl = model_dir / f"{CHECKPOINT_NAME}.model.pkl"
    for f in (checkpoint, checkpoint_pkl):
        if not f.is_file():
            raise FileNotFoundError(f"Missing CTSpine1K weight file: {f}")

    dest = Path(dest)
    fold_dir = dest / "all"
    fold_dir.mkdir(parents=True, exist_ok=True)

    # symlink the large weights, copy the small pkl (both cheap)
    os.symlink(checkpoint.resolve(), fold_dir / checkpoint.name)
    shutil.copy(str(checkpoint_pkl), str(fold_dir / checkpoint_pkl.name))

    # nnU-Net v1's predict_from_folder needs a plans.pkl next to the fold; the
    # plans are embedded in the checkpoint's .model.pkl.
    info = load_pickle(str(checkpoint_pkl))
    write_pickle(info["plans"], str(dest / "plans.pkl"))

    return dest


def segment_spine(
    image_path,
    output_path,
    model_dir=MODEL_DIR,
    use_mirroring=True,
    step_size=0.5,
):
    image_path = Path(image_path)
    output_path = Path(output_path)
    model_dir = Path(model_dir)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="spineseg_"))
    try:
        recon_model = build_v1_model_folder(model_dir, tmp_dir / "model")
        in_dir = tmp_dir / "in"
        out_dir = tmp_dir / "out"
        in_dir.mkdir()
        out_dir.mkdir()

        # nnU-Net v1 expects <caseid>_<modality>.nii.gz; one CT modality -> _0000
        shutil.copy(str(image_path), str(in_dir / "case_0000.nii.gz"))

        predict_from_folder(
            model=str(recon_model),
            input_folder=str(in_dir),
            output_folder=str(out_dir),
            folds="all",
            save_npz=False,
            num_threads_preprocessing=2,
            num_threads_nifti_save=2,
            lowres_segmentations=None,
            part_id=0,
            num_parts=1,
            tta=bool(use_mirroring),
            mixed_precision=True,
            overwrite_existing=True,
            mode="normal",
            step_size=step_size,
            checkpoint_name=CHECKPOINT_NAME,
        )

        prediction = out_dir / "case.nii.gz"
        if not prediction.is_file():
            raise RuntimeError(f"nnU-Net produced no output at {prediction}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(prediction), str(output_path))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"Saved spine segmentation to {output_path}")
    return output_path


def main():
    image_path = "/home/iml/fryderyk.koegl/data/PET_CT_bone/raw_data/sub-4JyXPRGGa-Q/ses-20190103/ct/sub-4JyXPRGGa-Q_ses-20190103_sequ-2_acq-ax_ce-ContrastAgent_ct.nii.gz"
    output_path = "//home/iml/fryderyk.koegl/code/LapIRN-koegl/temp_output/sub-4JyXPRGGa-Q_ses-20190103_sequ-2_acq-ax_ce-ContrastAgent_ct_spine.nii.gz"

    segment_spine(image_path, output_path)


if __name__ == "__main__":
    main()
