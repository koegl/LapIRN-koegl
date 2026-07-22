import os
from typing import Callable, List, Tuple

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
) -> None:
    """Heatmap of all labels x cases, ordered by label id, with a mean column.

    Cells with Dice < mask_below are shown gray; vmin sets the color floor.
    The final 'mean' column (average over cases) is never masked.
    """
    sub = labels.sort_index()
    sub["mean"] = sub.mean(axis=1)
    sub.index = fmt_labels(sub.index, names)

    mask = sub < mask_below
    mask["mean"] = False  # keep the mean column always colored

    fig, ax = plt.subplots(
        figsize=(max(8, sub.shape[1] * 0.5), max(4, sub.shape[0] * 0.18))
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
    ax.set_ylabel("Label ID")
    ax.set_title("Dice Heatmap: All Labels x Cases")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def load_label_names(csv_path: str) -> pd.Series:
    """Load id->name mapping (no header) as a Series indexed by label id."""
    df = pd.read_csv(csv_path, header=None, index_col=0)
    names = df[1]
    names.index = names.index.astype(int)
    return names


def fmt_labels(ids: pd.Index, names: pd.Series) -> List[str]:
    """Format label ids as 'id: name' for axes and tables."""
    formatted = [f"{i}: {names.get(i, '?')}" for i in ids]
    return formatted


def main() -> None:
    csv_path = "/home/iml/fryderyk.koegl/code/LapIRN-koegl/submission_results/csvs/results_official_val_dice_per_labelPSMAReg_LapIRN_resilient-shrike-38730428_stagelvl3_best.csv"
    out_dir = "/home/iml/fryderyk.koegl/data/PSMAReg/dice_per_label_analysis_figs"
    names_path = (
        "/home/iml/fryderyk.koegl/code/LapIRN-koegl/total_segmentator_labels.csv"
    )
    n_worst = 30
    present_eps = 1e-3
    min_present = 15

    model_name = (
        csv_path.split("/")[-1].split(".csv")[0].split("LapIRN_")[-1].split("_stage")[0]
    )

    os.makedirs(out_dir, exist_ok=True)
    labels, _ = load_scores(csv_path)
    names = load_label_names(names_path)
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

    tasks: List[Tuple[str, Callable]] = [
        (
            f"worst_labels_heatmap_{model_name}.png",
            lambda p: plot_heatmap(labels, names, p, mask_below=0.5, vmin=0.5),
        )
    ]

    for fname, fn in tqdm(tasks, desc="Rendering figures", ncols=80):
        out_path = os.path.join(out_dir, fname)
        tqdm.write(f"  writing {out_path}")
        fn(out_path)


if __name__ == "__main__":
    main()
