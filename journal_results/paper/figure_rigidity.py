"""Experiment 2: rigidity weight sweep on the validation set -> PDF + LaTeX.

T2 + PET IO with one rigidity penalty (Staring, Staring with erosion, per-bone
rigid fit) at x0.1 ... x100 of its calibrated reference weight
(evaluation_official_scorer/rigidity_sweep/curve.csv, patient means, written
by official_scorer/select_stage.py rigidity).
    columns: short-distance (5-15 mm) and long-distance (>= 15 mm) distortion
    within bones; rows: CT Dice and bone Dice
One line per method through its weights; R1 (no rigidity) as a separate point;
the weight chosen for the test set outlined; the CT Dice row also shows the
selection threshold (2 Dice points below R1).

    ../../.venv/bin/python figure_rigidity.py  ->  out/figures/rigidity.{pdf,tex}
"""

import json

import matplotlib

matplotlib.use("Agg")
import common  # noqa: E402

PAPER_FIGURE_DIR = common.REPO.parent / "psmareg-extension" / "figures"
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SWEEP_DIR = common.REPO / "evaluation_official_scorer" / "rigidity_sweep"
CALIBRATION = (
    common.REPO / "journal_results" / "rigidity_calibration" / "reference.json"
)
# method -> (label, color); categorical slots 1-3 of the reference palette
METHODS = {
    "staring": ("Staring", "#eb6834"),
    "staring_eroded": ("Staring + erosion", "#1baf7a"),
    "kabsch": ("Per-bone rigid fit", "#2a78d6"),
}
# columns: distortion; rows: accuracy (grid of all combinations)
X_AXES = (
    ("dist_distortion_short", r"Short-distance distortion (\%) $\downarrow$"),
    ("dist_distortion_long", r"Long-distance distortion (\%) $\downarrow$"),
)
Y_AXES = (
    ("dice_ct", r"CT Dice (\%) $\uparrow$"),
    ("bone_dice", r"Bone Dice (\%) $\uparrow$"),
)
REFERENCE_COLOR = "#0b0b0b"
# weight-factor labels per panel: method -> ((x, y) offset of the largest
# weight's label, (x, y) offset of the smallest weight's label), in points
OFFSETS_A = {
    "staring": ((10, -2), (10, -1)),
    "staring_eroded": ((-13, -0.5), (10, 0)),
    "kabsch": ((-13, -0.4), (10, 0)),
}
OFFSETS_B = {
    "staring": ((-11, 0), (10, -1)),
    "staring_eroded": ((-13, -4), (10, 0)),
    "kabsch": ((5.5, -7), (10, 0)),
}
OFFSETS_C = {
    "staring": ((10, -2), (10, 0)),
    "staring_eroded": ((5.5, -9), (7, -4)),
    "kabsch": ((-13, -0.5), (10, 0)),
}
OFFSETS_D = {
    "staring": ((-11, 0), (10, 0)),
    "staring_eroded": ((-13, -0.5), (10, 0)),
    "kabsch": ((5.5, -7), (10, 0)),
}
TEXT = "#0b0b0b"
MUTED = "#52514e"


def main() -> None:
    curve = pd.read_csv(SWEEP_DIR / "curve.csv")
    chosen = json.loads((SWEEP_DIR / "chosen.json").read_text())
    reference_weight = json.loads(CALIBRATION.read_text())["reference_weight"]
    chosen_weight = {c["method"]: c["weight"] for c in chosen["chosen"].values()}
    for column in ("dice_ct", "bone_dice") + tuple(c for c, _ in X_AXES):
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
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.6), sharex="col", sharey="row")
    panels = [
        (axes[i, j], x_col, y_col)
        for i, (y_col, _) in enumerate(Y_AXES)
        for j, (x_col, _) in enumerate(X_AXES)
    ]
    offsets = (OFFSETS_A, OFFSETS_B, OFFSETS_C, OFFSETS_D)  # row-major, as panels
    for (ax, x_col, y_col), panel_offsets in zip(panels, offsets):
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
            largest_offset, smallest_offset = panel_offsets[method]
            for row, offset in (
                (d.iloc[-1], largest_offset),
                (d.iloc[0], smallest_offset),
            ):
                factor = row["weight"] / reference_weight[method]
                ax.annotate(
                    rf"$\times${factor:g}",
                    (row[x_col], row[y_col]),
                    textcoords="offset points",
                    xytext=offset,
                    ha="center",
                    va="center",
                    fontsize=6,
                    color=color,
                )
        ax.scatter(
            r1[x_col], r1[y_col], marker="*", s=70, color=REFERENCE_COLOR, zorder=6
        )
        ax.grid(color="#e4e3df", linewidth=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for j, (_, x_label) in enumerate(X_AXES):
        axes[1, j].set_xlabel(x_label)
    for i, (_, y_label) in enumerate(Y_AXES):
        axes[i, 0].set_ylabel(y_label)
    # selection threshold on CT Dice (top row)
    threshold = 100 * (chosen["reference_dice"] - chosen["dice_tolerance"])
    for ax in axes[0]:
        ax.axhline(
            threshold, color=MUTED, linewidth=0.8, linestyle=(0, (4, 2)), zorder=1
        )
    axes[0, 1].annotate(
        "selection threshold",
        (axes[0, 1].get_xlim()[1], threshold),
        textcoords="offset points",
        xytext=(-2, -8),
        ha="right",
        fontsize=6,
        color=MUTED,
    )
    for ax, panel_label in zip(axes.flat, ("(a)", "(b)", "(c)", "(d)")):
        if panel_label in ("(a)", "(c)"):
            ax.text(
                -0.175,
                1.065,
                panel_label,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontweight="bold",
                color=TEXT,
            )
        else:
            ax.text(
                -0.12,
                1.065,
                panel_label,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontweight="bold",
                color=TEXT,
            )

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
            label="no bone rigidity",
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
        bbox_to_anchor=(0.5, 1.04),
        handlelength=1.8,
    )
    fig.tight_layout(w_pad=1.5, h_pad=1.0)

    out_dir = PAPER_FIGURE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "rigidity.pdf", bbox_inches="tight", pad_inches=0)
    tex = r"""\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{figures/rigidity.pdf}
\caption{Sweep of bone-rigidity weights on the validation set (26 patients, 38 pairs): network trained with the PET-preservation losses, followed by instance optimization (IO) with the PET-preservation terms and one bone-rigidity penalty at $0.1$ to $100$ times its calibrated reference weight (factors at the ends of each line).
Top: CT Dice; bottom: bone Dice; left: distortion of short (5--15~mm) and right: long ($\geq15$~mm) within-bone distances, defined as the mean absolute relative change in distance between point pairs in the same bone.
Color identifies the bone-rigidity penalty.
Star: IO without bone rigidity; dashed line in (a,~b): selection threshold two Dice points below the reference; outlined marker: weight chosen for the test set, the largest within the threshold up to $\times100$.
Points are means over patients after averaging each patient's pairs.}
\label{fig:rigidity}
\end{figure}
"""
    (out_dir / "rigidity.tex").write_text(tex)
    columns = ["method", "weight"] + [c for c, _ in X_AXES] + [c for c, _ in Y_AXES]
    print(curve[columns].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
