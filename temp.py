import json
from pathlib import Path
from typing import Dict


def load_mapping(mapping_path) -> Dict[str, str]:
    with open(mapping_path, "r") as f:
        mapping = json.load(f)
    return mapping


mapping_path = Path(
    "/home/iml/fryderyk.koegl/code/LapIRN-koegl/data_klinikum_mapping.json"
)
mapping = load_mapping(mapping_path)

ts_im_path = Path(
    "/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/imagesTs_klinikum"
)
ts_seg_path = Path(
    "/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/labelsTs_klinikum"
)

images = [p.name for p in ts_im_path.iterdir() if p.is_file()]
segs = [p.name for p in ts_seg_path.iterdir() if p.is_file()]

assert len(images) == len(segs), (
    f"number of images ({len(images)}) and segmentations ({len(segs)}) does not match"
)
assert set(images) == set(segs), "image and segmentation names do not match"

pat_ids = set()

for im in images:
    patient_id = im.split("_")[1]
    pat_ids.add(patient_id)

for patient_id in sorted(pat_ids):
    print(f'"{patient_id}",')
