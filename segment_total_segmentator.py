import subprocess
from pathlib import Path
from typing import List

from tqdm import tqdm


def find_ct_images(input_dir: Path) -> List[Path]:
    images = sorted(input_dir.glob("*.nii.gz"))
    # images = sorted(input_dir.glob("PSMARegPSMA_*_0000_*.nii.gz"))
    return images


def find_ct_images_klinikum(input_dir: Path) -> List[Path]:
    pass


def output_name(image: Path) -> str:
    parts = image.name.replace(".nii.gz", "").split("_")
    name = f"{parts[1]}_{parts[3]}.nii.gz"
    return name


def segment(image: Path, output: Path, fast: bool = True) -> None:
    cmd = ["TotalSegmentator", "-i", str(image), "-o", str(output), "--ml"]
    if fast:
        cmd.append("-f")

    # swallow TotalSegmentator's chatter so it does not break up the progress
    # bar, but surface it if the run actually failed
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        tqdm.write(result.stdout)
        tqdm.write(result.stderr)
        raise subprocess.CalledProcessError(
            result.returncode, cmd, result.stdout, result.stderr
        )


def main() -> None:
    input_dir = Path("/home/iml/fryderyk.koegl/data/PET_CT_bone/raw_data")
    output_dir = Path("/home/iml/fryderyk.koegl/data/PET_CT_bone/raw_labels")
    skip_existing = True
    fast = False

    output_dir.mkdir(parents=True, exist_ok=True)
    images = find_ct_images_klinikum(input_dir)

    pbar = tqdm(images, desc="TotalSegmentator", unit="case")
    for image in pbar:
        output = output_dir / image.name
        pbar.set_postfix_str(output.name)

        if skip_existing and output.exists():
            tqdm.write(f"skip {output.name}")
            continue

        segment(image, output, fast=fast)


if __name__ == "__main__":
    main()
