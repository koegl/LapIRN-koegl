"""Experiment 2: rigidity weight sweep on the validation set -> PDF + LaTeX.

T2 + PET IO with one rigidity penalty (Staring, Staring with erosion, per-bone
rigid fit) at x0.1 ... x100 of its calibrated reference weight
(rigidity_sweep/curve.csv, patient means, written by sweep_rigidity.py).
    (a) CT Dice vs. short-distance (5-15 mm) distortion within bones
    (b) bone Dice vs. long-distance (>= 15 mm) distortion within bones
One line per method through its weights; R1 (no rigidity) as a separate point;
the weight chosen for the test set outlined; (a) also shows the selection
threshold (1 Dice point below R1).

    ../../.venv/bin/python figure_rigidity.py  ->  out/figures/rigidity.{pdf,tex}
"""

import json

import matplotlib

matplotlib.use("Agg")
import common  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SWEEP_DIR = common.REPO / "journal_results" / "rigidity_sweep"
CALIBRATION = (
    common.REPO / "journal_results" / "rigidity_calibration" / "reference.json"
)
# method -> (label, colour); categorical slots 1-3 of the reference palette
METHODS = {
    "staring": ("Staring", "#eb6834"),
    "staring_eroded": ("Staring + erosion", "#1baf7a"),
    "kabsch": ("Per-bone rigid fit", "#2a78d6"),
}
PANELS = (
    (
        "dist_distortion_short",
        "dice_ct",
        "Short-distance distortion (5--15 mm)",
        r"CT Dice (\%) $\uparrow$",
    ),
    (
        "dist_distortion_long",
        "bone_dice",
        r"Long-distance distortion ($\geq$15 mm)",
        r"Bone Dice (\%) $\uparrow$",
    ),
)
REFERENCE_COLOR = "#0b0b0b"
LABEL_NUDGE = {"staring": -4, "staring_eroded": 4, "kabsch": 0}  # pt, smallest weight
# largest weight: left of the end for Staring (bottom of the plot), below it
# erosion, above the rigid fit (clear of the lines, threshold and axis)
LARGEST_OFFSET = {"staring": (-6, 0), "staring_eroded": (0, -9), "kabsch": (4, 9)}
LARGEST_HA = {"staring": "right", "staring_eroded": "center", "kabsch": "center"}
TEXT = "#0b0b0b"
MUTED = "#52514e"


def main() -> None:
    curve = pd.read_csv(SWEEP_DIR / "curve.csv")
    chosen = json.loads((SWEEP_DIR / "chosen.json").read_text())
    reference_weight = json.loads(CALIBRATION.read_text())["reference_weight"]
    chosen_weight = {c["method"]: c["weight"] for c in chosen["chosen"].values()}
    for column in ("dice_ct", "bone_dice"):
        curve[column] *= 100
    r1 = curve[curve["method"] == "none"].iloc[0]

    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.size": 8,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": TEXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.6))
    for ax, (x_col, y_col, x_label, y_label) in zip(axes, PANELS):
        for method, (_, color) in METHODS.items():
            d = curve[curve["method"] == method].sort_values("weight")
            ax.plot(d[x_col], d[y_col], color=color, linewidth=1.5, zorder=3)
            ax.scatter(
                d[x_col],
                d[y_col],
                s=14,
                color=color,
                edgecolors="white",
                linewidths=0.6,
                zorder=4,
            )
            best = d[(d["weight"] - chosen_weight[method]).abs() < 1e-9].iloc[0]
            ax.scatter(
                best[x_col],
                best[y_col],
                s=40,
                color=color,
                edgecolors=TEXT,
                linewidths=1.0,
                zorder=5,
            )
            # weight factors at the two ends of each line
            # smallest weight right of its end, largest left of it; vertical
            # nudge keeps the nearby ends of Staring and erosion apart
            nudge = LABEL_NUDGE[method]
            ends = (
                (d.iloc[0], (5, nudge), "left"),
                (d.iloc[-1], LARGEST_OFFSET[method], LARGEST_HA[method]),
            )
            for row, offset, ha in ends:
                factor = row["weight"] / reference_weight[method]
                ax.annotate(
                    rf"$\times${factor:g}",
                    (row[x_col], row[y_col]),
                    textcoords="offset points",
                    xytext=offset,
                    ha=ha,
                    va="center",
                    fontsize=6,
                    color=color,
                )
        ax.scatter(
            r1[x_col], r1[y_col], marker="*", s=70, color=REFERENCE_COLOR, zorder=6
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(color="#e4e3df", linewidth=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    threshold = 100 * (chosen["reference_dice"] - chosen["dice_tolerance"])
    axes[0].axhline(
        threshold, color=MUTED, linewidth=0.8, linestyle=(0, (4, 2)), zorder=1
    )
    axes[0].annotate(
        "selection threshold",
        (axes[0].get_xlim()[1], threshold),
        textcoords="offset points",
        xytext=(-2, -8),
        ha="right",
        fontsize=6,
        color=MUTED,
    )
    axes[0].set_title("(a)", fontsize=8, loc="left")
    axes[1].set_title("(b)", fontsize=8, loc="left")

    handles = [
        Line2D([], [], color=c, marker="o", markersize=3, linewidth=1.5, label=label)
        for label, c in METHODS.values()
    ]
    handles += [
        Line2D(
            [],
            [],
            color=REFERENCE_COLOR,
            marker="*",
            markersize=7,
            linestyle="",
            label="no rigidity",
        ),
        Line2D(
            [],
            [],
            color="white",
            marker="o",
            markersize=6,
            markeredgecolor=TEXT,
            linestyle="",
            label="chosen weight",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 1.07),
        handlelength=1.8,
    )
    fig.tight_layout(w_pad=2.0)

    out_dir = common.OUT_DIR / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "rigidity.pdf", bbox_inches="tight", pad_inches=0)
    tex = r"""\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{figures/rigidity.pdf}
\caption{Rigidity weight sweep on the validation set: network trained with the PET losses, IO with the PET terms and one rigidity penalty at $0.1$ to $100$ times its calibrated reference weight (factors at the ends of each line). (a)~CT Dice against the distortion of short (5--15~mm) within-bone distances; dashed: selection threshold, one Dice point below the variant without rigidity (star). (b)~Bone Dice against the distortion of long ($\geq$15~mm) within-bone distances. Outlined: weight chosen for the test set (the largest within the threshold, at most $\times100$). Points: patient means.}
\label{fig:rigidity}
\end{figure}
"""
    (out_dir / "rigidity.tex").write_text(tex)
    print(curve[["method", "weight", x_col, y_col]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
