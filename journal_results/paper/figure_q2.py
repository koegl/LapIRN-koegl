"""Q2 of experiment 1: losses in training + IO, or only in IO? -> PDF + LaTeX.

For each loss set (PET / rigidity / both) two runs on test, both with the same
IO losses: the network trained with the losses (I2-I4, filled) and the network
trained without them (I5-I7, hollow). Reference:
neither loss (I1, grey). Two panels: |MTV change| and |TLG change| against CT
Dice; points are patient means with 95 % patient-bootstrap CIs.

The paired differences for the text (IO only minus trained, Holm over the 3
pairs of a metric) are printed and written to out/figures/q2_effects.txt.

    ../../.venv/bin/python figure_q2.py  ->  out/figures/q2.{pdf,tex}
"""

from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import common  # noqa: E402
from common import METRIC_COLUMNS, two_sided  # noqa: E402

# loss set -> (trained + IO, IO only, colour); categorical slots 1-3 of the
# reference palette (valid for all pairs)
PAIRS: Dict[str, Tuple[str, str, str]] = {
    "PET": ("I2", "I5", "#2a78d6"),
    "Rigidity": ("I3", "I6", "#eb6834"),
    "PET + rigidity": ("I4", "I7", "#1baf7a"),
}
REFERENCE = ("I1", "Neither", "#8c8b86")
X_METRIC = "dice_ct"
Y_METRICS = ("mtv_bias", "tlg_bias")
AXIS_LABELS = {
    "dice_ct": r"CT Dice (\%) $\uparrow$",
    "mtv_bias": r"$|\Delta\mathrm{MTV}|$ (\%) $\downarrow$",
    "tlg_bias": r"$|\Delta\mathrm{TLG}|$ (\%) $\downarrow$",
}
N_BOOTSTRAP = 2000
SEED = 0
ALPHA = 0.05
TEXT = "#0b0b0b"
MUTED = "#52514e"


def mean_ci(values: pd.Series, rng) -> Tuple[float, float, float]:
    v = values.dropna().to_numpy()
    means = rng.choice(v, size=(N_BOOTSTRAP, len(v)), replace=True).mean(axis=1)
    return v.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def effects(patients) -> str:
    """IO only minus trained, per loss set and metric, for the text."""
    rng = np.random.default_rng(SEED)
    lines = []
    for metric in (X_METRIC,) + Y_METRICS:
        _, higher, scale, decimals, _ = METRIC_COLUMNS[metric]
        p, rows = {}, {}
        for name, (trained, io_only, _) in PAIRS.items():
            p[name], _ = two_sided(patients[io_only][metric], patients[trained][metric], higher)
            both = pd.concat(
                [patients[io_only][metric], patients[trained][metric]], axis=1, keys=["a", "b"]
            ).dropna()
            diff = (both["a"] - both["b"]) * scale
            _, lo, hi = mean_ci(diff, rng)
            rows[name] = f"{diff.mean():+.{decimals}f} [{lo:+.{decimals}f}, {hi:+.{decimals}f}]"
        adjusted = common.holm(p)
        for name in PAIRS:
            lines.append(f"{metric:9s} {name:15s} IO only - trained = {rows[name]}  p_holm = {adjusted[name]:.3g}")
    return "\n".join(lines)


def main() -> None:
    runs = [REFERENCE[0]] + [r for t, o, _ in PAIRS.values() for r in (t, o)]
    metrics = [X_METRIC, *Y_METRICS]
    patients = {r: common.patient_means(common.load_run(r), metrics) for r in runs}
    rng = np.random.default_rng(SEED)
    stats = {
        r: {m: mean_ci(patients[r][m] * METRIC_COLUMNS[m][2], rng) for m in metrics}
        for r in runs
    }

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
    fig, axes = plt.subplots(1, len(Y_METRICS), figsize=(6.0, 2.6))

    def point(ax, run, color, filled):
        x, xl, xh = stats[run][X_METRIC]
        y, yl, yh = stats[run][ax.y_metric]
        ax.errorbar(
            x, y, xerr=[[x - xl], [xh - x]], yerr=[[y - yl], [yh - y]],
            fmt="none", ecolor=color, elinewidth=0.8, alpha=0.6, zorder=2,
        )
        ax.scatter(
            x, y, s=36, zorder=4, linewidths=1.5, edgecolors=color,
            facecolors=color if filled else "white",
        )

    for ax, y_metric in zip(axes, Y_METRICS):
        ax.y_metric = y_metric
        point(ax, REFERENCE[0], REFERENCE[2], True)
        for trained, io_only, color in PAIRS.values():
            point(ax, trained, color, True)
            point(ax, io_only, color, False)
        ax.set_xlabel(AXIS_LABELS[X_METRIC])
        ax.set_ylabel(AXIS_LABELS[y_metric])
        ax.grid(color="#e4e3df", linewidth=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    handles = [
        Line2D([], [], marker="o", ls="", color=c, markersize=6, label=name)
        for name, (_, _, c) in PAIRS.items()
    ]
    handles += [
        Line2D([], [], marker="o", ls="", color=REFERENCE[2], markersize=6, label=f"{REFERENCE[1]} (I1)"),
        Line2D([], [], marker="o", ls="", markerfacecolor=MUTED, markeredgecolor=MUTED, markersize=6, label="trained + IO"),
        Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=MUTED, markersize=6, label="IO only"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 1.06), handletextpad=0.2, columnspacing=1.0)
    fig.tight_layout()

    out_dir = common.OUT_DIR / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "q2.pdf", bbox_inches="tight")
    text = effects(patients)
    (out_dir / "q2_effects.txt").write_text(text + "\n")
    n = len(patients[REFERENCE[0]])
    tex = rf"""\begin{{figure}}[t]
\centering
\includegraphics[width=\textwidth]{{figures/q2.pdf}}
\caption{{Losses in training and IO versus only in IO (test set, {n} patients). For each loss set, filled: network trained with the losses, then IO with them; hollow: network trained without them (T1), same IO. Grey: neither loss (I1). Points: patient means with 95\,\% patient-bootstrap confidence intervals.}}
\label{{fig:q2}}
\end{{figure}}
"""
    (out_dir / "q2.tex").write_text(tex)
    print(text)


if __name__ == "__main__":
    main()
