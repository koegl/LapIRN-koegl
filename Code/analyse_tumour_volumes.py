from pathlib import Path
from typing import Dict, List, Tuple

import config
import matplotlib.pyplot as plt
import my_data
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm
from matplotlib.ticker import FuncFormatter


def plain_formatter() -> FuncFormatter:
    formatter = FuncFormatter(lambda v, _pos: f"{v:,.0f}")
    return formatter


def pet_tumour_volume_mm3(
    data_dir: Path, case_id: str, tp: str, tumour_label: int
) -> float:
    label_dir = data_dir / "labelsTr"
    path = label_dir / f"PSMARegPSMA_{case_id}_0001_{tp}.nii.gz"
    img = nib.load(path.as_posix())
    data = img.get_fdata().astype(np.uint8)
    voxel_count = int((data == tumour_label).sum())
    voxel_vol = float(np.abs(np.linalg.det(img.affine[:3, :3])))
    volume = voxel_count * voxel_vol
    return volume


def consecutive_pairs(tps: List[str]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for i in range(len(tps) - 1):
        pairs.append((tps[i], tps[i + 1]))
    return pairs


def build_volume_table(data_dir: Path, tumour_label: int) -> pd.DataFrame:
    case_timepoints = my_data.list_case_timepoints(data_dir)

    rows: List[Dict[str, object]] = []
    for case_id, tps in tqdm.tqdm(sorted(case_timepoints.items()), desc="cases"):
        if len(tps) < 2:
            continue
        for tp_a, tp_b in consecutive_pairs(tps):
            vol_first = pet_tumour_volume_mm3(data_dir, case_id, tp_a, tumour_label)
            vol_second = pet_tumour_volume_mm3(data_dir, case_id, tp_b, tumour_label)
            row = {
                "pair": f"{case_id}:{tp_a}-{tp_b}",
                "case_id": case_id,
                "vol_first_mm3": vol_first,
                "vol_second_mm3": vol_second,
                "vol_diff_mm3": vol_second - vol_first,
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    both_zero = (df["vol_first_mm3"] == 0) & (df["vol_second_mm3"] == 0)
    df = df[~both_zero].reset_index(drop=True)
    return df


def make_plots(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["vol_diff_mm3"], kde=True, ax=ax, bins=60)
    ax.axvline(0.0, color="red", linestyle="--", linewidth=1)
    ax.set_title("Tumour volume change (second - first)")
    ax.set_xlabel("Volume change [mm^3]")
    ax.xaxis.set_major_formatter(plain_formatter())
    ax.yaxis.set_major_formatter(plain_formatter())
    fig.tight_layout()
    fig.savefig(out_dir / "diff_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(data=df, x="vol_first_mm3", y="vol_second_mm3", ax=ax)
    lim = max(df["vol_first_mm3"].max(), df["vol_second_mm3"].max())
    ax.plot([0, lim], [0, lim], color="red", linestyle="--", linewidth=1)
    ax.set_title("First vs second tumour volume")
    ax.set_xlabel("First [mm^3]")
    ax.set_ylabel("Second [mm^3]")
    ax.xaxis.set_major_formatter(plain_formatter())
    ax.yaxis.set_major_formatter(plain_formatter())
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_dir / "first_vs_second.png", dpi=150)
    plt.close(fig)

    df_rel = df.copy()
    df_rel["rel_change"] = df_rel["vol_diff_mm3"] / (df_rel["vol_first_mm3"] + 1e-8)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df_rel["rel_change"], kde=True, ax=ax, bins=60)
    ax.axvline(0.0, color="red", linestyle="--", linewidth=1)
    ax.set_title("Relative tumour volume change")
    ax.set_xlabel("(second - first) / first")
    ax.xaxis.set_major_formatter(plain_formatter())
    ax.yaxis.set_major_formatter(plain_formatter())
    fig.tight_layout()
    fig.savefig(out_dir / "relative_change.png", dpi=150)
    plt.close(fig)


def print_summary(df: pd.DataFrame) -> None:
    stats = df[["vol_first_mm3", "vol_second_mm3", "vol_diff_mm3"]].describe()
    print(stats)


def main() -> None:
    pd.set_option("display.float_format", lambda v: f"{v:.2f}")
    np.set_printoptions(suppress=True)

    cfg = config.TrainingConfig()
    data_dir = cfg.data_dir
    tumour_label = 1
    out_dir = Path("/home/iml/fryderyk.koegl/code/LapIRN-koegl/tumour_volume_analysis")

    df = build_volume_table(data_dir, tumour_label)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "tumour_volumes.csv"
    df.to_csv(csv_path, index=False)
    print(f"wrote {csv_path}")

    print_summary(df)
    make_plots(df, out_dir)
    print(f"plots in {out_dir}")


if __name__ == "__main__":
    main()
