from pathlib import Path

import nibabel as nib

dir_input = Path("/home/iml/fryderyk.koegl/data/PSMAReg/io_labels_pet")
dir_output = Path("/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/labelsTs_fast")

input_files = dir_input.glob("*.nii*")
for input_file in input_files:
    break
    seg = nib.load(input_file)

    input_name_without_ext = input_file.name.replace(".nii.gz", "").replace(".nii", "")
    input_split = input_name_without_ext.split("_")

    output_name = f"PSMARegPSMA_{input_split[1]}_0001_{input_split[2]}.nii.gz"

    output_file = dir_output / output_name

    nib.save(seg, output_file)

    x = 0
