import shutil
from pathlib import Path

import numpy as np

dir_rib = Path("/home/iml/fryderyk.koegl/data/temp")

dir_pet_labels = Path("/home/iml/fryderyk.koegl/data/PET_CT_bone/pet_labels")
patients_pet_labels = [p.name for p in sorted(dir_pet_labels.iterdir()) if p.is_dir()]
dir_total_labels = Path(
    "/home/iml/fryderyk.koegl/data/PET_CT_bone/segmentations_total_segmentator"
)
patients_totalsegmentator = [
    p.name for p in sorted(dir_total_labels.iterdir()) if p.is_dir()
]

patients_ribs = [p.name for p in sorted(dir_rib.iterdir()) if p.is_dir()]

patient_in_all_3 = list(
    set(patients_pet_labels) & set(patients_totalsegmentator) & set(patients_ribs)
)

# shuffle the list of patients
np.random.seed(42)
np.random.shuffle(patient_in_all_3)

selected = patient_in_all_3[:20]
path_destination_ribs = Path(
    "/home/iml/fryderyk.koegl/data/PET_CT_bone/segmentations_ribs/"
)

for patient in selected:
    source_path = dir_rib / patient
    destination_path = path_destination_ribs / patient
    shutil.copytree(source_path, destination_path)

x = 0
