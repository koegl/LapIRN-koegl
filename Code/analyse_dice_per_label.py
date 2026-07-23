import os
from typing import Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm


def load_scores(csv_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load CSV, return per-label DataFrame (index=label) and the mean_dice row."""
    df = pd.read_csv(csv_path, index_col=0)
    mean_row = df.loc["mean_dice"].astype(float)
    labels = df.drop(index="mean_dice")
    labels.index = labels.index.astype(int)
    labels = labels.astype(float)
    return labels, mean_row


def compute_stats(labels: pd.DataFrame, present_eps: float) -> pd.DataFrame:
    """Per-label summary stats. 'present' cols exclude cases with dice <= eps."""
    present_mask = labels > present_eps
    n_present = present_mask.sum(axis=1)
    present_vals = labels.where(present_mask)

    stats = pd.DataFrame(index=labels.index)
    stats["mean_all"] = labels.mean(axis=1)
    stats["mean_present"] = present_vals.mean(axis=1)
    stats["median"] = labels.median(axis=1)
    stats["std"] = labels.std(axis=1)
    stats["min"] = labels.min(axis=1)
    stats["n_present"] = n_present
    stats["n_near_zero"] = (labels <= present_eps).sum(axis=1)
    stats = stats.sort_values("mean_all")
    return stats


def plot_heatmap(
    labels: pd.DataFrame,
    names: pd.Series,
    out_path: str,
    mask_below: float = 0.5,
    vmin: float = 0.5,
    volumes: Optional[pd.Series] = None,
    sort_by_volume: bool = False,
) -> None:
    """Heatmap of all labels x cases, with a mean column.

    Rows are ordered by label id, or by average label volume (largest first)
    when `sort_by_volume` is set. The volumes are shown in the row labels
    whenever `volumes` is given, independent of the ordering.

    Cells with Dice < mask_below are shown gray; vmin sets the color floor.
    The final 'mean' column (average over cases) is never masked.
    """
    sub = labels.loc[sort_labels(labels.index, volumes if sort_by_volume else None)]
    sub["mean"] = sub.mean(axis=1)
    sub.index = fmt_labels(sub.index, names, volumes)

    mask = sub < mask_below
    mask["mean"] = False  # keep the mean column always colored

    # the extra width leaves room for the colorbar and its tick labels
    fig, ax = plt.subplots(
        figsize=(max(8, sub.shape[1] * 0.5) + 2.5, max(4, sub.shape[0] * 0.18))
    )
    ax.set_facecolor("lightgray")
    sns.heatmap(
        sub,
        cmap="RdYlGn",
        vmin=vmin,
        vmax=1.0,
        mask=mask,
        ax=ax,
        cbar_kws={"label": "Dice"},
        linewidths=0.3,
        linecolor="white",
        yticklabels=True,
    )
    ax.axvline(sub.shape[1] - 1, color="black", linewidth=1.5)
    ax.set_xlabel("Case ID")
    ax.set_ylabel("Label" if not sort_by_volume else "Label (by volume)")
    title = "Dice Heatmap: All Labels x Cases"
    if sort_by_volume:
        title += " (sorted by mean volume)"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_label_names(csv_path: str) -> pd.Series:
    """Load id->name mapping (no header) as a Series indexed by label id."""
    df = pd.read_csv(csv_path, header=None, index_col=0)
    names = df[1]
    names.index = names.index.astype(int)
    return names


def load_label_volumes(csv_path: str) -> pd.Series:
    """Load the average per-label volumes written by calculate_label_volumes.py."""
    df = pd.read_csv(csv_path, index_col=0)
    volumes = df["volume_mm3"].astype(float)
    volumes.index = volumes.index.astype(int)
    return volumes


def sort_labels(ids: pd.Index, volumes: Optional[pd.Series]) -> pd.Index:
    """Order label ids by id, or by average volume (largest first) if given.

    Labels missing from the volume table are treated as volume 0 and end up
    last, ordered by id.
    """
    if volumes is None:
        return ids.sort_values()

    key = pd.Series([volumes.get(i, 0.0) for i in ids], index=ids)
    order = key.sort_values(ascending=False, kind="stable").index
    return order


def fmt_labels(
    ids: pd.Index, names: pd.Series, volumes: Optional[pd.Series] = None
) -> List[str]:
    """Format label ids as 'id: name' ('name; (volume ml); id=id' with volumes)."""
    if volumes is None:
        formatted = [f"{i}: {names.get(i, '?')}" for i in ids]
    else:
        formatted = [
            f"{names.get(i, '?')}; ({volumes.get(i, 0.0) / 1000.0:,.1f}ml); id={i}"
            for i in ids
        ]
    return formatted


def main() -> None:
    csv_path = "/home/iml/fryderyk.koegl/code/LapIRN-koegl/submission_results/csvs/results_official_val_dice_per_labelPSMAReg_LapIRN_resilient-shrike-38730428_stagelvl3_best_IO_lr2.0e-01_it60_wncc5.00_wJac2000.00_wSmooth2.00_wCT5.00_wPET0.00_wDiceCT5.00_wDicePET0.00_wTLG2.00_wMaskedJac2.00_wBoneRigid2.00 .csv"
    out_dir = "/home/iml/fryderyk.koegl/data/PSMAReg/dice_per_label_analysis_figs"
    names_path = (
        "/home/iml/fryderyk.koegl/code/LapIRN-koegl/total_segmentator_labels.csv"
    )
    volumes_path = "/home/iml/fryderyk.koegl/code/LapIRN-koegl/submission_results/csvs/label_volumes.csv"
    n_worst = 30
    present_eps = 1e-3
    min_present = 15

    # order the heatmap rows by average label volume instead of by label id
    sort_by_volume: bool = False

    model_name = (
        csv_path.split("/")[-1].split(".csv")[0].split("LapIRN_")[-1].split("_stage")[0]
    )

    os.makedirs(out_dir, exist_ok=True)
    labels, _ = load_scores(csv_path)
    names = load_label_names(names_path)
    volumes = load_label_volumes(volumes_path)
    stats = compute_stats(labels, present_eps)
    stats = stats[stats["n_present"] >= min_present]
    stats.insert(0, "name", [names.get(i, "?") for i in stats.index])

    print(f"\n=== {n_worst}ś worst labels (sorted by mean_all) ===")
    print(stats.head(n_worst).round(4).to_string())
    print("\nLabels with near-zero Dice in >=1 case:")
    flagged = stats[stats["n_near_zero"] > 0]
    print(
        flagged[["mean_all", "n_present", "n_near_zero"]].to_string()
        if not flagged.empty
        else "  none"
    )

    suffix = "_by_volume" if sort_by_volume else ""
    tasks: List[Tuple[str, Callable]] = [
        (
            f"worst_labels_heatmap_{model_name}{suffix}.png",
            lambda p: plot_heatmap(
                labels,
                names,
                p,
                mask_below=0.0,
                vmin=0.5,
                volumes=volumes,
                sort_by_volume=sort_by_volume,
            ),
        )
    ]

    for fname, fn in tqdm(tasks, desc="Rendering figures", ncols=80):
        out_path = os.path.join(out_dir, fname)
        tqdm.write(f"  writing {out_path}")
        fn(out_path)


if __name__ == "__main__":
    main()
