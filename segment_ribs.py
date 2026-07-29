"""Segment ribs in a CT image with the nnU-Net weights in stump-rib/ribseg_model_weights.

The model was trained on images in PIR orientation at 0.8mm isotropic spacing
(see ribseg_model_weights/inference_config.json). The input is therefore
reoriented to PIR before inference and the prediction is reoriented back so the
output shares the geometry of the input image.
"""

import json
import shutil
import tempfile
from pathlib import Path

import SimpleITK as sitk
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

MODEL_DIR = Path(__file__).parent / "stump-rib" / "ribseg_model_weights"


def load_inference_config(model_dir):
    with open(Path(model_dir) / "inference_config.json") as f:
        return json.load(f)


def build_predictor(model_dir, folds, step_size, use_mirroring, device):
    predictor = nnUNetPredictor(
        tile_step_size=step_size,
        use_gaussian=True,
        use_mirroring=use_mirroring,
        perform_everything_on_device=device.type == "cuda",
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(
        str(model_dir),
        use_folds=folds,
        checkpoint_name="checkpoint_final.pth",
    )
    return predictor


def get_orientation(image):
    return sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
        image.GetDirection()
    )


def segment_ribs(
    image_path, output_path, model_dir=MODEL_DIR, folds=(0, 1, 2), device=None
):
    image_path = Path(image_path)
    output_path = Path(output_path)
    model_dir = Path(model_dir)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    cfg = load_inference_config(model_dir)
    target_orientation = "".join(cfg["model_expected_orientation"])

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(min(8, torch.get_num_threads()))

    image = sitk.ReadImage(str(image_path))
    original_orientation = get_orientation(image)
    reoriented = sitk.DICOMOrient(image, target_orientation)

    predictor = build_predictor(
        model_dir=model_dir,
        folds=tuple(folds),
        step_size=cfg.get("default_step_size", 0.5),
        use_mirroring=bool(cfg.get("inference_augmentation", 1)),
        device=device,
    )

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

    # back to the orientation / geometry of the input image
    prediction = sitk.DICOMOrient(prediction, original_orientation)
    prediction = sitk.Cast(prediction, sitk.sitkUInt8)
    if prediction.GetSize() != image.GetSize():
        raise RuntimeError(
            f"Prediction size {prediction.GetSize()} does not match input size {image.GetSize()}"
        )
    prediction.CopyInformation(image)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(prediction, str(output_path))
    print(f"Saved rib segmentation to {output_path}")
    return output_path


def main():
    image_path = "/home/iml/fryderyk.koegl/data/PET_CT_bone/raw_data/sub-4JyXPRGGa-Q/ses-20190103/ct/sub-4JyXPRGGa-Q_ses-20190103_sequ-2_acq-ax_ce-ContrastAgent_ct.nii.gz"
    output_path = "/home/iml/fryderyk.koegl/data/PET_CT_bone/segmentations/sub-4JyXPRGGa-Q/ses-20190103/ct/sub-4JyXPRGGa-Q_ses-20190103_sequ-2_acq-ax_ce-ContrastAgent_ct_ribs.nii.gz"

    segment_ribs(image_path, output_path)


if __name__ == "__main__":
    main()
