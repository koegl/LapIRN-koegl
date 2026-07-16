import subprocess
from pathlib import Path
from typing import List

from tqdm import tqdm


def find_ct_images(input_dir: Path) -> List[Path]:
    images = sorted(input_dir.glob("PSMARegPSMA_*_0000_*.nii.gz"))
    return images


def output_name(image: Path) -> str:
    parts = image.name.replace(".nii.gz", "").split("_")
    name = f"{parts[1]}_{parts[3]}.nii.gz"
    return name


def segment(image: Path, output: Path, fast: bool = True) -> None:
    cmd = ["TotalSegmentator", "-i", str(image), "-o", str(output), "--ml"]
    if fast:
        cmd.append("-f")
    subprocess.run(cmd, check=True)


def main() -> None:
    input_dir = Path("/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/imagesVal")
    output_dir = Path(
        "/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/segmentations_fast"
    )
    skip_existing = True

    output_dir.mkdir(parents=True, exist_ok=True)
    images = find_ct_images(input_dir)

    pbar = tqdm(images, desc="TotalSegmentator", unit="case")
    for image in pbar:
        output = output_dir / output_name(image)
        pbar.set_postfix_str(output.name)

        if skip_existing and output.exists():
            tqdm.write(f"skip {output.name}")
            continue

        segment(image, output)
        tqdm.write(f"done {output.name}")


if __name__ == "__main__":
    main()
