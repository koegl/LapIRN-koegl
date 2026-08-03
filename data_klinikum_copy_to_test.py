import json
import shutil
from pathlib import Path
from typing import Dict


def load_mapping(mapping_path) -> Dict[str, str]:
    with open(mapping_path, "r") as f:
        mapping = json.load(f)
    return mapping


def main():
    mapping = load_mapping(
        Path("/home/iml/fryderyk.koegl/code/LapIRN-koegl/data_klinikum_mapping.json")
    )

    dir_orig_seg_rib = Path(
        "/home/iml/fryderyk.koegl/data/PET_CT_bone/segmentations_ribs"
    )

    images_dir = Path(
        "/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/imagesTr_klinikum"
    )
    images_dir_test = Path(
        "/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/imagesTs_klinikum"
    )

    rib_patients = [p for p in dir_orig_seg_rib.iterdir() if p.is_dir()]

    keys_to_move = []

    for patient in rib_patients:
        keys = [k for k in mapping.keys() if k.startswith(patient.name)]

        if len(keys) == 0:
            print(f"no mapping for {patient.name}")

        keys_to_move.extend(keys)

    for key in keys_to_move:
        new_name = mapping[key]
        source_path = images_dir / new_name
        destination_path = images_dir_test / new_name

        shutil.move(source_path, destination_path)

    x = 0


if __name__ == "__main__":
    main()
