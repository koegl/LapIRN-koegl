import glob
import os
from typing import List

import matplotlib.pyplot as plt


def main() -> None:
    work_dir: str = "crop_sweep"
    plot_path: str = "crop_sweep.png"

    files = sorted(glob.glob(os.path.join(work_dir, "time_crop*.txt")))

    crops: List[int] = []
    z_sizes: List[int] = []
    times: List[float] = []

    for path in files:
        with open(path) as f:
            crop_str, z_str, time_str = f.read().strip().split("\t")
        crops.append(int(crop_str))
        z_sizes.append(int(z_str))
        times.append(float(time_str))

    for c, z, t in zip(crops, z_sizes, times):
        print(f"crop={c:3d}  z={z:3d}  time={t:6.2f}s")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(z_sizes, times, marker="o")
    ax.set_xlabel("axial size (z voxels)")
    ax.set_ylabel("time for 2 volumes (s)")
    ax.set_title("Runtime vs axial extent (fresh process, fast, body_seg)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)

    print(f"saved plot to {plot_path}", flush=True)


if __name__ == "__main__":
    main()
