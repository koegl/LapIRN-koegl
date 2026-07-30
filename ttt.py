from pathlib import Path

dir_rib = Path("/home/iml/fryderyk.koegl/data/PET_CT_bone/segmentations_ribs")

dir_pet_labels = Path("/home/iml/fryderyk.koegl/data/PET_CT_bone/pet_labels")
pet_label_patients = [p.name for p in sorted(dir_pet_labels.iterdir()) if p.is_dir()]
dir_total_labels = Path(
    "/home/iml/fryderyk.koegl/data/PET_CT_bone/segmentations_total_segmentator"
)
total_label_patients = [
    p.name for p in sorted(dir_total_labels.iterdir()) if p.is_dir()
]


not_in_pet = []
not_in_total = []

for patient in sorted(dir_rib.iterdir()):
    if patient.name not in pet_label_patients:
        not_in_pet.append(patient.name)
    if patient.name not in total_label_patients:
        not_in_total.append(patient.name)

x = 0

print("missing pet labels:")
for p in not_in_pet:
    print(p)

print("\n\nmissing total labels:")
for p in not_in_total:
    print(p)

x = 0
