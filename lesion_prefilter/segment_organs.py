"""Select train cases for the lesion pre-filter and segment their CTs with
TotalSegmentator (task ``total``, not fast).

Cases are train patients from ``split_journal.json`` sorted by ground-truth
tumour burden and picked evenly spaced, so the selection covers low to high
burden. All timepoints of a selected patient are used.

Outputs (in this folder):
    cases.json                      selected images as ``<case>_<tp>``
    organs/<case>_<tp>.nii.gz       TotalSegmentator multilabel mask
"""

import json
import subprocess
from pathlib import Path
from typing import List

import nibabel as nib
import numpy as np
from tqdm import tqdm

HERE = Path(__file__).parent
DATA = Path("/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset")
SPLIT = HERE.parent / "split_journal.json"
N_PATIENTS = 15


def image_path(case: str, tp: str, modality: str) -> Path:
    # modality: "0000" = CT, "0001" = PET
    return DATA / "imagesTr" / f"PSMARegPSMA_{case}_{modality}_{tp}.nii.gz"


def label_path(case: str, tp: str) -> Path:
    return DATA / "labelsTr" / f"PSMARegPSMA_{case}_0001_{tp}.nii.gz"


def timepoints(case: str) -> List[str]:
    paths = (DATA / "imagesTr").glob(f"PSMARegPSMA_{case}_0001_*.nii.gz")
    return sorted(p.name.split("_")[-1].replace(".nii.gz", "") for p in paths)


def burden_ml(case: str) -> float:
    """Largest ground-truth lesion volume over all timepoints."""
    volumes = []
    for tp in timepoints(case):
        label = nib.load(label_path(case, tp))
        voxel_ml = np.prod(label.header.get_zooms()) / 1000
        volumes.append((np.asarray(label.dataobj) > 0).sum() * voxel_ml)
    return max(volumes)


def select_cases(n: int) -> List[str]:
    train = json.loads(SPLIT.read_text())["train"]
    burdens = {c: burden_ml(c) for c in tqdm(train, desc="burden")}
    ranked = sorted(train, key=burdens.get)
    idx = np.linspace(0, len(ranked) - 1, n).round().astype(int)
    patients = [ranked[i] for i in idx]
    return [f"{c}_{tp}" for c in patients for tp in timepoints(c)]


def segment(ct: Path, output: Path) -> None:
    # write to a temp name and rename, so an interrupted run never leaves a
    # partial file that would be skipped on restart
    tmp = output.with_name(f"tmp_{output.name}")
    cmd = ["TotalSegmentator", "-i", str(ct), "-o", str(tmp), "--ml"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        tqdm.write(result.stdout)
        tqdm.write(result.stderr)
        raise subprocess.CalledProcessError(
            result.returncode, cmd, result.stdout, result.stderr
        )
    tmp.rename(output)


def main() -> None:
    cases_file = HERE / "cases.json"
    if not cases_file.exists():
        cases_file.write_text(json.dumps(select_cases(N_PATIENTS), indent=2))
    images = json.loads(cases_file.read_text())

    out_dir = HERE / "organs"
    out_dir.mkdir(exist_ok=True)
    pbar = tqdm(images, desc="TotalSegmentator", unit="image")
    for name in pbar:
        output = out_dir / f"{name}.nii.gz"
        pbar.set_postfix_str(name)
        if output.exists():
            continue
        segment(image_path(*name.split("_"), "0000"), output)


if __name__ == "__main__":
    main()
