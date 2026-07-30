import argparse
import subprocess
from pathlib import Path
from typing import Any, List

from tqdm import tqdm

from data_klinikum_preprocess import automatically_find_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TotalSegmentator batch runner")
    parser.add_argument(
        "--order",
        choices=["forward", "reverse", "middle_out"],
        default="forward",
    )
    parser.add_argument("--start-index", type=int, default=None)
    args = parser.parse_args()
    return args


def reorder_middle_out(items: List[Any], start: int) -> List[Any]:
    idx_order = [start]
    offset = 1
    while len(idx_order) < len(items):
        left = start - offset
        right = start + offset
        if left >= 0:
            idx_order.append(left)
        if right < len(items):
            idx_order.append(right)
        offset += 1
    reordered = [items[i] for i in idx_order]
    return reordered


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
    args = parse_args()

    input_dir = Path("/home/iml/fryderyk.koegl/data/PET_CT_bone/raw_data")
    skip_existing = True
    fast = False

    image_pairs = automatically_find_pairs(input_dir)

    if args.order == "reverse":
        image_pairs.reverse()
    elif args.order == "middle_out":
        if args.start_index is not None:
            start_index = args.start_index
        else:
            start_index = len(image_pairs) // 2
        image_pairs = reorder_middle_out(image_pairs, start_index)

    pbar = tqdm(image_pairs, desc="TotalSegmentator", unit="case", ncols=150)
    for pair in pbar:
        for image in [pair["fixed"], pair["moving"]]:
            output_path = Path(
                image.as_posix().replace("raw_data", "segmentations_total_segmentator")
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            pbar.set_postfix_str(output_path.name)

            if skip_existing and output_path.exists():
                tqdm.write(f"skip {output_path.name}")
                continue

            segment(image, output_path, fast=fast)


if __name__ == "__main__":
    main()
