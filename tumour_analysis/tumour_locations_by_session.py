"""Compare the tumour/CT-label overlaps between the baseline and the follow-up sessions.

Case ids look like ``PSMARegPSMA_<patient>_0000_<session>``, where session ``00``
is the baseline scan and ``01``, ``02``, ... are the follow-ups.

Instead of only contrasting two independent violins (baseline vs. everything
else), the sessions are treated as what they are: repeated measurements nested
in a patient. Every metric therefore gets three views

* a paired violin, baseline vs. follow-up, with the scans of one patient joined
  by a line, so the pairing stays visible on top of the distribution,
* the distribution of the within-patient change (follow-up minus baseline),
  which removes the (large) between-patient variance, together with a Wilcoxon
  signed-rank test,
* the per-patient trajectory over the session index, which keeps the ordering of
  the follow-ups instead of collapsing them into one group.

Inputs are the CSVs written by ``tumour_locations.py``:

* ``tumour_bone_overlap_per_image.csv``            (one row per image)
* ``tumour_ct_label_overlap_per_image.csv``        (one row per image and CT label)

``tumour_ct_label_overlap.csv`` itself is already averaged over all images and
carries no session information, so the long per-image table is used for the CT
label part.
"""

import argparse
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

ANALYSIS_DIR = Path("/home/iml/fryderyk.koegl/code/LapIRN-koegl/tumour_analysis")
BONE_CSV = "tumour_bone_overlap_per_image.csv"
LABEL_LONG_CSV = "tumour_ct_label_overlap_per_image.csv"

BONE_METRICS = [
    "dice_tumour_bone",
    "frac_tumour_in_bone",
    "frac_bone_covered_by_tumour",
    "tumour_volume_ml",
]
LABEL_METRIC = "frac_label_covered_by_tumour"

BASELINE = "baseline"
FOLLOWUP = "follow-up"


def add_session_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Split ``case_id`` into patient id and session index."""
    parts = frame["case_id"].str.split("_")
    frame = frame.copy()
    frame["patient"] = parts.str[1]
    frame["session"] = parts.str[-1].astype(int)
    frame["group"] = np.where(frame["session"] == 0, BASELINE, FOLLOWUP)
    return frame


def paired_table(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per patient: baseline value, follow-up values and their difference.

    ``followup_mean`` averages all follow-ups of a patient, ``followup_last``
    takes the highest session index. Patients without a baseline or without any
    follow-up are dropped.
    """
    baseline = (
        frame[frame["session"] == 0]
        .groupby("patient")[metric]
        .mean()
        .rename("baseline")
    )
    followups = frame[frame["session"] > 0]
    if followups.empty:
        return pd.DataFrame()
    last_session = followups.sort_values("session").groupby("patient").tail(1)
    table = pd.concat(
        [
            baseline,
            followups.groupby("patient")[metric].mean().rename("followup_mean"),
            followups.groupby("patient")[metric].count().rename("n_followups"),
            last_session.set_index("patient")[metric].rename("followup_last"),
        ],
        axis=1,
    )  # .dropna(subset=["baseline", "followup_mean"])
    table["delta_mean"] = table["followup_mean"] - table["baseline"]
    table["delta_last"] = table["followup_last"] - table["baseline"]
    return table.reset_index()


def wilcoxon(table: pd.DataFrame, column: str = "delta_mean") -> dict:
    """Paired Wilcoxon signed-rank test on the within-patient differences."""
    deltas = table[column].dropna().to_numpy() if len(table) else np.array([])
    result = {
        "n_patients_paired": int(len(deltas)),
        "median_delta": float(np.median(deltas)) if len(deltas) else float("nan"),
        "mean_delta": float(np.mean(deltas)) if len(deltas) else float("nan"),
        "wilcoxon_stat": float("nan"),
        "wilcoxon_p": float("nan"),
    }
    if len(deltas) >= 5 and np.any(deltas != 0):
        statistic, p_value = stats.wilcoxon(deltas)
        result["wilcoxon_stat"] = float(statistic)
        result["wilcoxon_p"] = float(p_value)
    return result


def group_stats(frame: pd.DataFrame, metric: str, group: str) -> dict:
    values = frame.loc[frame["group"] == group, metric]  # .dropna()
    if values.empty:
        return {}
    return {
        f"n_{group}": int(len(values)),
        f"mean_{group}": float(values.mean()),
        f"std_{group}": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
        f"median_{group}": float(values.median()),
        f"q25_{group}": float(values.quantile(0.25)),
        f"q75_{group}": float(values.quantile(0.75)),
    }


def violin(axis: plt.Axes, groups: List[np.ndarray], positions: List[int], colour: str):
    """Violin with a light fill; empty groups are skipped."""
    usable = [(g, p) for g, p in zip(groups, positions) if len(g) > 1]
    if not usable:
        return
    parts = axis.violinplot(
        [g for g, _ in usable],
        positions=[p for _, p in usable],
        showextrema=False,
        widths=0.7,
    )
    for body in parts["bodies"]:
        body.set_facecolor(colour)
        body.set_alpha(0.35)


def plot_metric(frame: pd.DataFrame, metric: str, table: pd.DataFrame, path: Path):
    """Paired violin, within-patient change and per-patient trajectories."""
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    rng = np.random.default_rng(0)

    # (a) paired violin -----------------------------------------------------
    axis = axes[0]
    values = [
        frame.loc[frame["group"] == group, metric].dropna().to_numpy()
        for group in (BASELINE, FOLLOWUP)
    ]
    violin(axis, values, [0, 1], "tab:blue")
    for patient, rows in frame.groupby("patient"):
        base = rows.loc[rows["session"] == 0, metric]
        follow = rows.loc[rows["session"] > 0, metric].dropna()
        if base.empty or follow.empty:
            continue
        for value in follow:
            axis.plot(
                [0, 1],
                [base.iloc[0], value],
                color="grey",
                alpha=0.25,
                linewidth=0.6,
                zorder=1,
            )
    for position, group in enumerate((BASELINE, FOLLOWUP)):
        points = frame.loc[frame["group"] == group, metric].dropna().to_numpy()
        axis.scatter(
            position + rng.uniform(-0.06, 0.06, len(points)),
            points,
            s=8,
            color="tab:blue",
            alpha=0.6,
            zorder=2,
        )
    axis.set_xticks([0, 1])
    axis.set_xticklabels(
        [f"{BASELINE}\n(n={len(values[0])})", f"{FOLLOWUP}\n(n={len(values[1])})"]
    )
    axis.set_ylabel(metric)
    axis.set_title("baseline vs. follow-up (paired)")

    # (b) within-patient change --------------------------------------------
    axis = axes[1]
    deltas = table["delta_mean"].dropna().to_numpy() if len(table) else np.array([])
    violin(axis, [deltas], [0], "tab:orange")
    if len(deltas):
        axis.scatter(
            rng.uniform(-0.06, 0.06, len(deltas)),
            deltas,
            s=10,
            color="tab:orange",
            alpha=0.7,
        )
    axis.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    test = wilcoxon(table)
    axis.set_xticks([0])
    axis.set_xticklabels([f"n={test['n_patients_paired']} patients"])
    axis.set_ylabel(f"mean follow-up - baseline\n({metric})")
    axis.set_title(
        f"within-patient change\nmedian={test['median_delta']:.4g}, "
        f"Wilcoxon p={test['wilcoxon_p']:.3g}"
    )

    # (c) trajectories ------------------------------------------------------
    axis = axes[2]
    for patient, rows in frame.groupby("patient"):
        rows = rows.sort_values("session")
        if len(rows) < 2:
            continue
        axis.plot(
            rows["session"],
            rows[metric],
            color="grey",
            alpha=0.3,
            linewidth=0.7,
            marker="o",
            markersize=2,
        )
    per_session = frame.groupby("session")[metric].median()
    axis.plot(
        per_session.index,
        per_session.to_numpy(),
        color="tab:red",
        linewidth=2,
        marker="o",
        label="median",
    )
    axis.set_xlabel("session index (0 = baseline)")
    axis.set_ylabel(metric)
    axis.set_xticks(sorted(frame["session"].unique()))
    axis.set_title("per-patient trajectory")
    axis.legend()

    figure.suptitle(metric)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"wrote {path}")


def print_bone_fraction(frame: pd.DataFrame) -> None:
    """Fraction of the tumour that sits inside bone, baseline vs. follow-ups.

    One patient is one sample: a patient's follow-ups are averaged first, so the
    few heavily-imaged patients do not dominate. Reported as median [IQR]
    because the fraction is bounded in [0, 1] and skewed -- a mean +- std would
    describe a symmetric spread that is not there.

    The two group lines are NOT directly comparable: every patient has a
    baseline, but only those with a follow-up appear in the second line, and
    that subset is not a random one. The third line is the honest comparison --
    the within-patient change, over the patients who have both.
    """
    metric = "frac_tumour_in_bone"
    if metric not in frame.columns:
        print(f"cannot print bone fractions: no '{metric}' column")
        return

    def median_iqr(values: pd.Series, label: str) -> str:
        return (
            f"{values.median():.4f} "
            f"[{values.quantile(0.25):.4f}-{values.quantile(0.75):.4f}] "
            f"(n={len(values)} {label})"
        )

    per_patient = (
        frame.groupby(["patient", "group"])[metric].mean()
        # frame.dropna(subset=[metric]).groupby(["patient", "group"])[metric].mean()
    ).reset_index()
    baseline = per_patient.loc[per_patient["group"] == BASELINE, metric]
    followup = per_patient.loc[per_patient["group"] == FOLLOWUP, metric]

    print(
        "tumour fraction inside bone, baseline:   " + median_iqr(baseline, "patients")
    )
    print(
        "tumour fraction inside bone, follow-ups: " + median_iqr(followup, "patients")
    )

    table = paired_table(frame, metric)
    if len(table):
        test = wilcoxon(table)
        deltas = table["delta_mean"]  # .dropna()
        print(
            "within-patient change (follow-up - baseline): "
            + median_iqr(deltas, "paired patients")
            + f", Wilcoxon p={test['wilcoxon_p']:.3g}"
        )


def analyse_bone(frame: pd.DataFrame, output_dir: Path) -> None:
    print_bone_fraction(frame)

    summary_rows = []
    paired_rows = []
    for metric in BONE_METRICS:
        if metric not in frame.columns:
            print(f"skipping missing column '{metric}'")
            continue
        table = paired_table(frame, metric)
        row = {"metric": metric}
        row.update(group_stats(frame, metric, BASELINE))
        row.update(group_stats(frame, metric, FOLLOWUP))
        row.update(wilcoxon(table))
        # unpaired comparison as well, for reference
        base_values = frame.loc[frame["group"] == BASELINE, metric].dropna()
        follow_values = frame.loc[frame["group"] == FOLLOWUP, metric].dropna()
        if len(base_values) and len(follow_values):
            statistic, p_value = stats.mannwhitneyu(
                base_values, follow_values, alternative="two-sided"
            )
            row["mannwhitney_stat"] = float(statistic)
            row["mannwhitney_p"] = float(p_value)
        summary_rows.append(row)

        if len(table):
            annotated = table.copy()
            annotated.insert(0, "metric", metric)
            paired_rows.append(annotated)
            plot_metric(frame, metric, table, output_dir / f"session_{metric}.png")

    summary = pd.DataFrame(summary_rows)
    summary_path = output_dir / "session_bone_summary.csv"
    summary.to_csv(summary_path, index=False, float_format="%.6g")
    print(f"wrote {summary_path}")

    if paired_rows:
        paired = pd.concat(paired_rows, ignore_index=True)
        paired_path = output_dir / "session_bone_paired_per_patient.csv"
        paired.to_csv(paired_path, index=False, float_format="%.6g")
        print(f"wrote {paired_path}")


def dumbbell(labels: pd.DataFrame, path: Path, top_n: int) -> None:
    """Baseline vs. follow-up coverage per CT label, as a dumbbell plot."""
    subset = labels.reindex(
        labels[[f"mean_{BASELINE}", f"mean_{FOLLOWUP}"]].max(axis=1).sort_values().index
    ).tail(top_n)
    figure, axis = plt.subplots(figsize=(8, max(4.0, 0.28 * len(subset))))
    positions = np.arange(len(subset))
    axis.hlines(
        positions,
        subset[f"mean_{BASELINE}"],
        subset[f"mean_{FOLLOWUP}"],
        color="lightgrey",
        linewidth=2,
        zorder=1,
    )
    axis.scatter(
        subset[f"mean_{BASELINE}"], positions, s=26, color="tab:blue", label=BASELINE
    )
    axis.scatter(
        subset[f"mean_{FOLLOWUP}"], positions, s=26, color="tab:red", label=FOLLOWUP
    )
    axis.set_yticks(positions)
    axis.set_yticklabels(subset["label_name"], fontsize=7)
    axis.set_xlabel(f"mean {LABEL_METRIC}")
    axis.set_title(f"top {len(subset)} CT labels, baseline vs. follow-up")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"wrote {path}")


def analyse_labels(frame: pd.DataFrame, output_dir: Path, top_n: int) -> None:
    rows = []
    for (label_id, label_name), label_frame in frame.groupby(
        ["label_id", "label_name"], sort=True
    ):
        table = paired_table(label_frame, LABEL_METRIC)
        row = {
            "label_id": label_id,
            "label_name": label_name,
            "is_bone_label": int(label_frame["is_bone_label"].max()),
        }
        row.update(group_stats(label_frame, LABEL_METRIC, BASELINE))
        row.update(group_stats(label_frame, LABEL_METRIC, FOLLOWUP))
        row.update(wilcoxon(table))
        rows.append(row)

    labels = pd.DataFrame(rows)
    if f"mean_{BASELINE}" in labels and f"mean_{FOLLOWUP}" in labels:
        labels["mean_difference"] = (
            labels[f"mean_{FOLLOWUP}"] - labels[f"mean_{BASELINE}"]
        )
        labels = labels.sort_values("mean_difference", ascending=False)
    path = output_dir / "session_ct_label_summary.csv"
    labels.to_csv(path, index=False, float_format="%.6g")
    print(f"wrote {path}")

    if len(labels):
        dumbbell(labels, output_dir / "session_ct_labels_dumbbell.png", top_n)
        print("\nlargest increases from baseline to follow-up (mean label coverage):")
        for _, row in labels.head(10).iterrows():
            print(
                f"  {row['label_name']:<25} {row.get(f'mean_{BASELINE}', float('nan')):.5f}"
                f" -> {row.get(f'mean_{FOLLOWUP}', float('nan')):.5f}"
                f"  (delta={row['mean_difference']:+.5f}, p={row['wilcoxon_p']:.3g})"
            )


def keep_registration_cohort(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only patients with a baseline and at least one follow-up.

    That is the cohort the registration is trained and evaluated on, and it also
    removes the confound in the group summaries: with every patient contributing
    to both groups, a baseline/follow-up difference can no longer be explained
    by which patients happened to be re-scanned.
    """
    sessions = frame.groupby("patient")["session"]
    paired = sessions.min().eq(0) & sessions.max().gt(0)
    keep = set(paired[paired].index)
    dropped = frame["patient"].nunique() - len(keep)
    if dropped:
        print(
            f"restricted to {len(keep)} patients with baseline + follow-up ({dropped} dropped)"
        )
    return frame[frame["patient"].isin(keep)].copy()


def read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.is_file():
        print(f"missing input {path} - skipping")
        return None
    return keep_registration_cohort(add_session_columns(pd.read_csv(path)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=ANALYSIS_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--top-n", type=int, default=30, help="CT labels shown in the dumbbell plot"
    )
    args = parser.parse_args()

    output_dir = args.output_dir or args.analysis_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    bone = read_csv(args.analysis_dir / BONE_CSV)
    if bone is not None:
        print(
            f"{len(bone)} images, {bone['patient'].nunique()} patients, "
            f"sessions {sorted(bone['session'].unique())}"
        )
        analyse_bone(bone, output_dir)

    labels = read_csv(args.analysis_dir / LABEL_LONG_CSV)
    if labels is not None:
        analyse_labels(labels, output_dir, args.top_n)


if __name__ == "__main__":
    main()
