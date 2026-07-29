import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

EXCLUDE = [
    "01161aaa0b_FU_img_FU_img_00.nii.gz",
    "01161aaa0b_BL_img_BL_img_00.nii.gz",
    "013d407166_FU_img_FU_img_01.nii.gz",
    "0266e33d3f_FU_img_FU_img_01.nii.gz",
    "0266e33d3f_BL_img_BL_img_01.nii.gz",
    "03c6b06952_FU_img_FU_img_00.nii.gz",
    "13fe9d8431_FU_img_FU_img_00.nii.gz",
    "16a5cdae36_FU_img_FU_img_01.nii.gz",
    "16a5cdae36_BL_img_BL_img_01.nii.gz",
    "289dff0766_BL_img_BL_img_00.nii.gz",
    "289dff0766_FU_img_FU_img_00.nii.gz",
    "289dff0766_FU_img_FU_img_01.nii.gz",
    "2f1ee6251b_FU_img_FU_img_01.nii.gz",
    "31fefc0e57_BL_img_BL_img_01.nii.gz",
    "31fefc0e57_FU_img_FU_img_01.nii.gz",
    "37a749d808_BL_img_BL_img_00.nii.gz",
    "37a749d808_BL_img_BL_img_01.nii.gz",
    "37a749d808_BL_img_BL_img_02.nii.gz",
    "37a749d808_FU_img_FU_img_00.nii.gz",
    "3988c7f88e_BL_img_BL_img_00.nii.gz",
    "3988c7f88e_BL_img_BL_img_01.nii.gz",
    "3988c7f88e_FU_img_FU_img_00.nii.gz",
    "432aca3a1e_BL_img_BL_img_00.nii.gz",
    "432aca3a1e_BL_img_BL_img_01.nii.gz",
    "432aca3a1e_FU_img_FU_img_00.nii.gz",
    "432aca3a1e_FU_img_FU_img_01.nii.gz",
    "432aca3a1e_FU_img_FU_img_02.nii.gz",
    "46ba9f2a69_BL_img_BL_img_00.nii.gz",
    "46ba9f2a69_BL_img_BL_img_01.nii.gz",
    "46ba9f2a69_FU_img_FU_img_00.nii.gz",
    "46ba9f2a69_FU_img_FU_img_01.nii.gz",
    "46ba9f2a69_FU_img_FU_img_02.nii.gz",
    "48aedb8880_BL_img_BL_img_00.nii.gz",
    "48aedb8880_BL_img_BL_img_01.nii.gz",
    "48aedb8880_FU_img_FU_img_00.nii.gz",
    "555d6702c9_FU_img_FU_img_01.nii.gz",
    "577bcc914f_BL_img_BL_img_00.nii.gz",
    "577bcc914f_BL_img_BL_img_01.nii.gz",
    "577bcc914f_FU_img_FU_img_00.nii.gz",
    "577ef1154f_BL_img_BL_img_01.nii.gz",
    "58a2fc6ed3_FU_img_FU_img_01.nii.gz",
    "6cdd60ea00_FU_img_FU_img_00.nii.gz",
    "74db120f0a_FU_img_FU_img_01.nii.gz",
    "7658d0d211_BL_img_BL_img_01.nii.gz",
    "7f100b7b36_BL_img_BL_img_00.nii.gz",
    "7f100b7b36_BL_img_BL_img_01.nii.gz",
    "7f100b7b36_FU_img_FU_img_00.nii.gz",
    "7f1de29e6d_BL_img_BL_img_01.nii.gz",
    "7f1de29e6d_BL_img_BL_img_00.nii.gz",
    "7f1de29e6d_FU_img_FU_img_00.nii.gz",
    "839ab46820_BL_img_BL_img_00.nii.gz",
    "839ab46820_FU_img_FU_img_00.nii.gz",
    "839ab46820_FU_img_FU_img_01.nii.gz",
    "839ab46820_FU_img_FU_img_02.nii.gz",
    "84d9ee44e4_BL_img_BL_img_00.nii.gz",
    "84d9ee44e4_FU_img_FU_img_00.nii.gz",
    "85d8ce590a_BL_img_BL_img_00.nii.gz",
    "85d8ce590a_BL_img_BL_img_01.nii.gz",
    "85d8ce590a_BL_img_BL_img_02.nii.gz",
    "85d8ce590a_FU_img_FU_img_00.nii.gz",
    "85d8ce590a_FU_img_FU_img_01.nii.gz",
    "85d8ce590a_FU_img_FU_img_02.nii.gz",
    "8d3bba7425_FU_img_FU_img_01.nii.gz",
    "8d5e957f29_BL_img_BL_img_00.nii.gz",
    "8d5e957f29_FU_img_FU_img_00.nii.gz",
    "8d5e957f29_FU_img_FU_img_01.nii.gz",
    "8f85517967_BL_img_BL_img_00.nii.gz",
    "8f85517967_FU_img_FU_img_00.nii.gz",
    "8f85517967_FU_img_FU_img_01.nii.gz",
    "9188905e74_BL_img_BL_img_00.nii.gz",
    "9188905e74_FU_img_FU_img_00.nii.gz",
    "92c8c96e4c_BL_img_BL_img_00.nii.gz",
    "92c8c96e4c_BL_img_BL_img_02.nii.gz",
    "92c8c96e4c_FU_img_FU_img_00.nii.gz",
    "9c838d2e45_FU_img_FU_img_01.nii.gz",
    "9dfcd5e558_BL_img_BL_img_00.nii.gz",
    "9dfcd5e558_BL_img_BL_img_01.nii.gz",
    "9dfcd5e558_FU_img_FU_img_00.nii.gz",
    "a3458cbbcc_FU_img_FU_img_01.nii.gz",
    "a7228a7d88_BL_img_BL_img_01.nii.gz",
    "af032fbcb0_BL_img_BL_img_01.nii.gz",
    "cb70ab3756_BL_img_BL_img_01.nii.gz",
    "cb70ab3756_FU_img_FU_img_01.nii.gz",
    "e00da03b68_BL_img_BL_img_01.nii.gz",
    "e00da03b68_FU_img_FU_img_01.nii.gz",
    "e165421110_BL_img_BL_img_01.nii.gz",
    "e165421110_FU_img_FU_img_01.nii.gz",
    "e1eee5e2b4_BL_img_BL_img_01.nii.gz",
    "e1eee5e2b4_FU_img_FU_img_01.nii.gz",
    "eb160de1de_FU_img_FU_img_01.nii.gz",
    "eb160de1de_BL_img_BL_img_00.nii.gz",
    "f2fc990265_BL_img_BL_img_01.nii.gz",
    "a8f15eda80_BL_img_BL_img_00.nii.gz",
    "a8f15eda80_FU_img_FU_img_00.nii.gz",
    "a8f15eda80_FU_img_FU_img_01.nii.gz",
    "b34f33d0bc_BL_img_BL_img_00.nii.gz",
    "b34f33d0bc_BL_img_BL_img_01.nii.gz",
    "b34f33d0bc_FU_img_FU_img_00.nii.gz",
    "c6f057b865_BL_img_BL_img_00.nii.gz",
    "c6f057b865_FU_img_FU_img_00.nii.gz",
    "c6f057b865_FU_img_FU_img_01.nii.gz",
    "caf1a3dfb5_BL_img_BL_img_00.nii.gz",
    "caf1a3dfb5_FU_img_FU_img_00.nii.gz",
    "caf1a3dfb5_FU_img_FU_img_01.nii.gz",
    "cedebb6e87_BL_img_BL_img_00.nii.gz",
    "cedebb6e87_BL_img_BL_img_01.nii.gz",
    "cedebb6e87_FU_img_FU_img_00.nii.gz",
    "f899139df5_BL_img_BL_img_00.nii.gz",
    "f899139df5_BL_img_BL_img_01.nii.gz",
    "f899139df5_FU_img_FU_img_00.nii.gz",
]


def foreground_anchor(
    image: sitk.Image, threshold: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = sitk.GetArrayFromImage(image)  # (z, y, x)
    idx = np.array(np.nonzero(arr > threshold))
    lo_zyx = idx.min(axis=1).astype(float)
    hi_zyx = idx.max(axis=1).astype(float)
    mid_zyx = (lo_zyx + hi_zyx) / 2.0

    lo_xyz = lo_zyx[::-1]
    hi_xyz = hi_zyx[::-1]
    mid_xyz = mid_zyx[::-1]

    # center in x/y, top edge in z
    anchor_idx = np.array([mid_xyz[0], mid_xyz[1], hi_xyz[2]])
    anchor = np.array(
        image.TransformContinuousIndexToPhysicalPoint(anchor_idx.tolist())
    )
    return anchor, lo_xyz, hi_xyz


def preprocess_file(
    in_path: Path,
    out_path: Path,
    output_size: Tuple[int, int, int] = (192, 192, 288),
    output_spacing: Tuple[float, float, float] = (2.7344, 2.7344, 3.27),
    threshold: float = -500.0,
    top_margin_vox: int = 22,
) -> None:
    image = sitk.ReadImage(in_path)
    ref = sitk.Image(list(output_size), image.GetPixelID())
    ref.SetOrigin((0.0, 0.0, 0.0))
    ref.SetDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
    ref.SetSpacing(output_spacing)

    n = np.array(output_size, dtype=float)
    target_idx = np.array(
        [(n[0] - 1.0) / 2.0, (n[1] - 1.0) / 2.0, n[2] - 1.0 - top_margin_vox]
    )
    target = np.array(ref.TransformContinuousIndexToPhysicalPoint(target_idx.tolist()))

    anchor, lo_xyz, hi_xyz = foreground_anchor(image, threshold)
    offset = anchor - target

    transform = sitk.TranslationTransform(3)
    transform.SetOffset(offset.tolist())

    resampled = sitk.Resample(
        image, ref, transform, sitk.sitkLinear, -1024.0, image.GetPixelID()
    )

    sitk.WriteImage(resampled, out_path.as_posix())


def collect_pairs(input_dir: Path) -> Dict[str, Dict[str, List[Path]]]:
    files = sorted(input_dir.glob("*.nii.gz"))
    files = [f for f in files if "mask" not in f.name.lower()]

    pairs: Dict[str, Dict[str, List[Path]]] = {}
    for f in files:
        match = re.match(r"^(?P<pid>[^_]+)_(?P<phase>BL|FU)_", f.name)
        if match is None:
            continue
        pid = match.group("pid")
        key = "fix" if match.group("phase") == "BL" else "mov"

        if pid not in pairs:
            pairs[pid] = {"fix": [], "mov": []}
        pairs[pid][key].append(f)

    return pairs


def remove_excluded_files(
    pairs: Dict[str, Dict[str, List[Path]]],
) -> Dict[str, Dict[str, List[Path]]]:
    # remove excluded files from pairs - if fix or mov after remov has 0, remove the entire case
    for pid in list(pairs.keys()):
        for key in ["fix", "mov"]:
            pairs[pid][key] = [
                f for f in pairs[pid][key] if os.path.basename(f) not in EXCLUDE
            ]

        if len(pairs[pid]["fix"]) == 0 or len(pairs[pid]["mov"]) == 0:
            del pairs[pid]

    return pairs


def main() -> None:

    mapping = {}

    inp_dir = Path("/home/iml/fryderyk.koegl/data/Longitudinal-CT/inputsTr")
    out_dir = Path(
        "/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/imagesTr_tubingen"
    )

    mapping_path = Path(
        "/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/mapping_tubingen.json"
    )
    if mapping_path.exists():
        mapping_path.unlink()

    pairs = collect_pairs(inp_dir)

    pairs = remove_excluded_files(pairs)

    fix_0 = 0
    mov_0 = 0
    fix_more_than_1 = 0
    mov_more_than_1 = 0

    to_check = []
    for pid, d in pairs.items():
        if len(d["fix"]) == 0:
            fix_0 += 1
        elif len(d["fix"]) > 1:
            fix_more_than_1 += 1

            if pid not in to_check:
                to_check.append(pid)

        if len(d["mov"]) == 0:
            mov_0 += 1
        elif len(d["mov"]) > 1:
            mov_more_than_1 += 1

            if pid not in to_check:
                to_check.append(pid)

    # get all paths of images with pids inside to_check and store them in new list
    paths_to_check = {}
    for pid in to_check:
        for key in ["fix", "mov"]:
            paths_to_check[pid] = paths_to_check.get(pid, []) + pairs[pid][key]

    for p in paths_to_check:
        print(f"PID: {p}")
        for f in paths_to_check[p]:
            print(f"  {f.name}")
        print()

    x = 0

    # return
    for idx, (pid, d) in enumerate(tqdm(pairs.items())):
        if len(d["fix"]) > 1 or len(d["mov"]) > 1:
            print("warning: more than one fix or mov for pid:", pid)
            continue

        # if pid not in "1d7f7abc18_BL_img_BL_img_00":
        #     continue
        path_fix = d["fix"][0]
        path_mov = d["mov"][0]

        new_name_fix = f"PSMARegPSMA_1{idx:03d}_0000_00.nii.gz"
        new_name_mov = f"PSMARegPSMA_1{idx:03d}_0000_01.nii.gz"

        preprocess_file(path_fix, out_dir / new_name_fix)
        preprocess_file(path_mov, out_dir / new_name_mov)

        mapping[path_fix.name] = new_name_fix
        mapping[new_name_fix] = path_fix.name
        mapping[path_mov.name] = new_name_mov
        mapping[new_name_mov] = path_mov.name

        # if idx == 1:
        #     break

    # save the mapping to json with 4 idnentiaiotn
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=4)


if __name__ == "__main__":
    main()
