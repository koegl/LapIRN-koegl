from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np


def load_volume(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    img = nib.load(path.as_posix())
    return np.asarray(img.dataobj, dtype=np.float32), img.affine


def imgnorm(arr: np.ndarray, p_low: float = 1e-4, p_high: float = 1e-4) -> np.ndarray:
    flat = np.sort(arr.flatten())
    lo = flat[int(p_low * len(flat))]
    hi = flat[-int(p_high * len(flat))]
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def describe(name: str, arr: np.ndarray) -> None:
    print(
        f"{name}: shape={arr.shape} dtype={arr.dtype} "
        f"min={arr.min():.3f} max={arr.max():.3f} "
        f"mean={arr.mean():.3f} std={arr.std():.3f} "
        f"frac_zero={(arr == arr.min()).mean():.3f}"
    )


def global_corr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a.flatten(), b.flatten())[0, 1])


def main() -> None:
    moving_path = Path(
        "/home/iml/fryderyk.koegl/data/LungCT_preprocessed/imagesTr/LungCT_0003_0001.nii.gz"
    )
    fixed_path = Path(
        "/home/iml/fryderyk.koegl/data/LungCT_preprocessed/imagesTr/LungCT_0002_0001.nii.gz"
    )

    moving, aff_m = load_volume(moving_path)
    fixed, aff_f = load_volume(fixed_path)

    describe("moving (0000) raw", moving)
    describe("fixed  (0001) raw", fixed)
    print("\naffines equal:", np.allclose(aff_m, aff_f))
    print("global corr (raw)   :", f"{global_corr(moving, fixed):.4f}")

    moving_n = imgnorm(moving)
    fixed_n = imgnorm(fixed)
    describe("moving normed", moving_n)
    describe("fixed  normed", fixed_n)
    print("global corr (normed):", f"{global_corr(moving_n, fixed_n):.4f}")


if __name__ == "__main__":
    main()
