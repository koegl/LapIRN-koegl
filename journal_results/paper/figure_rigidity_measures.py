"""Experiment 2: rigidity penalties evaluated with all rigidity measures -> PDF + LaTeX.

R2-R4 (rigidity.py) relative to R1 (no rigidity), per patient:
    (a) each rigidity measure as a percentage of its value without rigidity
        (residual of a per-bone rigid fit, |det J - 1|, orthonormality
        deviation, short and long distance distortion; lower = more rigid).
        A penalty that is consistently rigid sits low in every group; each
        penalty is expected to win on the measure it optimizes itself.
    (b) the accuracy cost: CT, bone, and soft-tissue Dice minus R1 (points)
Points: patient means; bars: 95 % patient-bootstrap CIs.

    ../../.venv/bin/python figure_rigidity_measures.py
        ->  out/figures/rigidity_measures.{pdf,tex}
"""

from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import common  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

PAPER_FIGURE_DIR = common.REPO.parent / "psmareg-extension" / "figures"
import rigidity  # noqa: E402
from rigidity import COLORS, RUNS  # noqa: E402

RIGIDITY = {
    "bre_mm": "Rigid-fit\nresidual",
    "staring_energy_bone": "Staring\nenergy",
    "dist_distortion_short": "Distortion\nshort",
    "dist_distortion_long": "Distortion\nlong",
}
ACCURACY = {"dice_ct": "CT", "bone_dice": "Bone", "soft_dice": "Soft tissue"}
N_BOOTSTRAP = 2000
SEED = 0
DODGE = 0.22
PENALTY_MARKERS = ("o", "s", "^")
TEXT = "#0b0b0b"
MUTED = "#52514e"


def mean_ci(values: pd.Series, rng) -> Tuple[float, float, float]:
    v = values.dropna().to_numpy()
    means = rng.choice(v, size=(N_BOOTSTRAP, len(v)), replace=True).mean(axis=1)
    return v.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def main() -> None:
    patients = rigidity.patients()
    reference = patients["R1"]
    rng = np.random.default_rng(SEED)
    penalties = list(RUNS)[1:]

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
    fig, axes = plt.subplots(
        1, 2, figsize=(6.5, 2.5), gridspec_kw={"width_ratios": [5, 3]}
    )

    def dots(ax, groups, value):
        for g, column in enumerate(groups):
            for k, run in enumerate(penalties):
                x = g + (k - 1) * DODGE
                values = value(run, column)
                if values.isna().all():  # measure missing (draft data)
                    continue
                mean, lo, hi = mean_ci(values, rng)
                ax.errorbar(
                    x,
                    mean,
                    yerr=[[mean - lo], [hi - mean]],
                    fmt=PENALTY_MARKERS[k],
                    markersize=4,
                    color=COLORS[run],
                    elinewidth=0.9,
                    capsize=0,
                    zorder=3,
                )
        ax.set_xticks(range(len(groups)), list(groups.values()), fontsize=7)
        ax.set_xlim(-0.6, len(groups) - 0.4)
        ax.grid(axis="y", color="#e4e3df", linewidth=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    # (a) rigidity measures relative to no rigidity, per patient
    dots(
        axes[0],
        RIGIDITY,
        lambda run, c: 100 * patients[run][c] / reference[c],
    )
    axes[0].axhline(100, color=MUTED, linewidth=0.8, linestyle=(0, (4, 2)), zorder=1)
    axes[0].set_ylabel(r"\% of no bone rigidity $\downarrow$")
    axes[0].set_ylim(bottom=0)
    axes[0].set_title("Bone rigidity", fontsize=8, loc="center")

    # (b) accuracy cost in Dice points
    dots(
        axes[1],
        ACCURACY,
        lambda run, c: 100 * (patients[run][c] - reference[c]),
    )
    axes[1].axhline(0, color=MUTED, linewidth=0.8, linestyle=(0, (4, 2)), zorder=1)
    axes[1].set_ylabel(r"Dice difference (points) $\uparrow$")
    axes[1].set_title("Alignment", fontsize=8, loc="center")

    for ax, panel_label in zip(axes, ("(a)", "(b)")):
        if panel_label == "(a)":
            ax.text(
                -0.135,
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
                -0.22,
                1.065,
                panel_label,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontweight="bold",
                color=TEXT,
            )

    handles = [
        Line2D(
            [],
            [],
            color=COLORS[r],
            marker=PENALTY_MARKERS[k],
            markersize=4,
            linestyle="",
        )
        for k, r in enumerate(penalties)
    ]
    handles.append(Line2D([], [], color=MUTED, linewidth=0.8, linestyle=(0, (4, 2))))
    fig.legend(
        handles,
        [RUNS[r] for r in penalties] + ["no bone rigidity (a: 100\\,\\%, b: 0)"],
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.07),
    )
    fig.tight_layout(w_pad=2.0)

    out_dir = PAPER_FIGURE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "rigidity_measures.pdf", bbox_inches="tight", pad_inches=0)
    n = len(reference)
    tex = rf"""\begin{{figure}}[t]
\centering
\includegraphics[width=\textwidth]{{figures/rigidity_measures.pdf}}
\caption{{Bone-rigidity penalties evaluated with all bone-rigidity measures on the test set ({n} patients, 38 pairs): network trained with the PET-preservation losses, followed by instance optimization (IO) with the PET-preservation terms and one bone-rigidity penalty at the weight chosen on the validation set.
(a)~Bone rigidity relative to IO without bone rigidity (dashed, 100\,\%): residual of a rigid fit to each bone, Staring energy (properness, orthonormality, and affinity) on the whole bone, and distortion of short (5--15~mm) and long ($\geq15$~mm) within-bone distances.
(b)~Change of CT, bone, and soft-tissue Dice relative to IO without bone rigidity.
Color and marker shape identify the bone-rigidity penalty.
Points are means over patients after averaging each patient's pairs; bars are 95\,\% patient-bootstrap confidence intervals.}}
\label{{fig:rigidity_measures}}
\end{{figure}}
"""
    (out_dir / "rigidity_measures.tex").write_text(tex)
    for run in penalties:
        rel = {c: (100 * patients[run][c] / reference[c]).mean() for c in RIGIDITY}
        print(run, {k: round(v, 1) for k, v in rel.items()})


if __name__ == "__main__":
    main()
