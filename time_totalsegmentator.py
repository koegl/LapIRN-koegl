import time

from totalsegmentator import python_api


def main() -> None:
    img: str = "/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/imagesVal/PSMARegPSMA_0001_0000_00.nii.gz"
    out_1: str = "seg_fixed.nii.gz"
    out_2: str = "seg_moving.nii.gz"

    t_0 = time.perf_counter()
    python_api.totalsegmentator(
        img, out_1, ml=True, task="total", fast=True, body_seg=True
    )
    t_1 = time.perf_counter()
    python_api.totalsegmentator(
        img, out_2, ml=True, task="total", fast=True, body_seg=True
    )
    t_2 = time.perf_counter()

    result = (
        f"first call:  {t_1 - t_0:.2f}s\n"
        f"second call: {t_2 - t_1:.2f}s\n"
        f"total:       {t_2 - t_0:.2f}s\n"
    )

    with open("timings.txt", "w") as f:
        f.write(result)

    print("\n" + result, flush=True)


if __name__ == "__main__":
    main()
