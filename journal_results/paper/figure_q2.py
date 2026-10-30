"""Experiment 1 (Q1 + Q2): which losses, and where they act -> PDF + LaTeX.

All eleven variants on test in two panels (|MTV change| and |TLG change|
against CT Dice; patient means): color = losses used (PET / rigidity / both),
marker = where they act (square: only in training, no IO, T2-T4; filled
circle: in training and IO, I2-I4; filled triangle: only in IO, I5-I7). No
losses: black X (no IO, T1) and star (with IO, I1). A thin dashed line connects
the variants of a loss group in this order. The significance tests are in the
appendix table (table_ablations.py) and the text.

The paired differences for the text (IO only minus training and IO, Holm over
the 3 pairs of a metric) are printed and written to out/figures/q2_effects.txt.

    ../../.venv/bin/python figure_q2.py  ->  out/figures/q2.{pdf,tex}
"""

from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import common  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import METRIC_COLUMNS, two_sided  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

PAPER_FIGURE_DIR = common.REPO.parent / "psmareg-extension" / "figures"

# losses used -> color (PET blue, rigidity red, both purple; none black)
GROUPS = {
    "None": "#0b0b0b",
    "PET": "#2a78d6",
    "Rigidity": "#d1342f",
    "PET + rigidity": "#8b3fbf",
}
LOSS_LABELS = {
    "PET": "PET preservation",
    "Rigidity": "Bone rigidity",
    "PET + rigidity": "PET preservation + bone rigidity",
}
# (losses used, where they act) -> run; "net" = only in training (no IO),
# "both" = in training and IO, "io" = only in IO; for None: without / with IO
RUNS: Dict[Tuple[str, str], str] = {
    ("None", "net"): "T1",
    ("PET", "net"): "T2",
    ("Rigidity", "net"): "T3",
    ("PET + rigidity", "net"): "T4",
    ("None", "both"): "I1",
    ("PET", "both"): "I2",
    ("Rigidity", "both"): "I3",
    ("PET + rigidity", "both"): "I4",
    ("PET", "io"): "I5",
    ("Rigidity", "io"): "I6",
    ("PET + rigidity", "io"): "I7",
}
PLACEMENT_ORDER = ("net", "both", "io")
MARKER = {"net": "s", "both": "o", "io": "^"}
NONE_MARKER = {"net": "x", "both": "*"}
# PET and PET + rigidity without IO nearly coincide: larger square below
SIZE = {("PET", "net"): 80, ("PET + rigidity", "net"): 24}
# Q2 pairs for the text: loss set -> (training and IO, only in IO)
PAIRS = {"PET": ("I2", "I5"), "Rigidity": ("I3", "I6"), "PET + rigidity": ("I4", "I7")}
X_METRIC = "dice_ct"
Y_METRICS = ("mtv_bias", "tlg_bias")
AXIS_LABELS = {
    "dice_ct": r"CT-organ Dice (\%) $\uparrow$",
    "mtv_bias": r"$|\Delta\mathrm{MTV}|$ (\%) $\downarrow$",
    "tlg_bias": r"$|\Delta\mathrm{TLG}|$ (\%) $\downarrow$",
}
N_BOOTSTRAP = 2000
SEED = 0
TEXT = "#0b0b0b"
MUTED = "#52514e"


def mean_ci(values: pd.Series, rng) -> Tuple[float, float, float]:
    v = values.dropna().to_numpy()
    means = rng.choice(v, size=(N_BOOTSTRAP, len(v)), replace=True).mean(axis=1)
    return v.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def effects(patients) -> str:
    """Only in IO minus training and IO, per loss set and metric, for the text."""
    rng = np.random.default_rng(SEED)
    lines = []
    for metric in (X_METRIC,) + Y_METRICS:
        _, higher, scale, decimals, _ = METRIC_COLUMNS[metric]
        p, rows = {}, {}
        for name, (trained, io_only) in PAIRS.items():
            p[name], _ = two_sided(
                patients[io_only][metric], patients[trained][metric], higher
            )
            both = pd.concat(
                [patients[io_only][metric], patients[trained][metric]],
                axis=1,
                keys=["a", "b"],
            ).dropna()
            diff = (both["a"] - both["b"]) * scale
            _, lo, hi = mean_ci(diff, rng)
            rows[name] = (
                f"{diff.mean():+.{decimals}f} [{lo:+.{decimals}f}, {hi:+.{decimals}f}]"
            )
        adjusted = common.holm(p)
        for name in PAIRS:
            lines.append(
                f"{metric:9s} {name:15s} IO only - trained = {rows[name]}"
                f"  p_holm = {adjusted[name]:.3g}"
            )
    return "\n".join(lines)


def main() -> None:
    metrics = [X_METRIC, *Y_METRICS]
    patients = {
        r: common.patient_means(common.load_run(r), metrics) for r in RUNS.values()
    }
    mean = {
        r: {m: patients[r][m].mean() * METRIC_COLUMNS[m][2] for m in metrics}
        for r in patients
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
    fig, axes = plt.subplots(1, len(Y_METRICS), figsize=(6.5, 2.8))
    for ax, y_metric in zip(axes, Y_METRICS):
        for group, color in GROUPS.items():
            path = [RUNS[(group, p)] for p in PLACEMENT_ORDER if (group, p) in RUNS]
            ax.plot(
                [mean[r][X_METRIC] for r in path],
                [mean[r][y_metric] for r in path],
                color=color,
                linewidth=0.8,
                linestyle=(0, (3, 2)),
                alpha=0.6,
                zorder=1,
            )
        for (group, placement), run in RUNS.items():
            color = GROUPS[group]
            x, y = mean[run][X_METRIC], mean[run][y_metric]
            if group == "None":
                ax.scatter(
                    x,
                    y,
                    s=80 if placement == "both" else 50,
                    marker=NONE_MARKER[placement],
                    color=color,
                    linewidths=1.2,
                    zorder=4,
                )
                continue
            ax.scatter(
                x,
                y,
                s=SIZE.get((group, placement), 40),
                marker=MARKER[placement],
                facecolors=color,
                edgecolors=color,
                linewidths=1.4,
                zorder=4,
            )
        ax.set_xlabel(AXIS_LABELS[X_METRIC])
        ax.set_ylabel(AXIS_LABELS[y_metric])
        ax.grid(color="#e4e3df", linewidth=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax, panel_label in zip(axes, ("(a)", "(b)")):
        ax.text(
            -0.18,
            1.0555,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
            color=TEXT,
        )

    loss_labels = {
        "None": "None",
        "PET": "PET preservation",
        "Rigidity": "Bone rigidity",
        "PET + rigidity": "PET preservation + bone rigidity",
    }
    loss_handles = [
        Line2D(
            [],
            [],
            color=color,
            linewidth=3.0,
            label=loss_labels[group],
        )
        for group, color in GROUPS.items()
    ]
    placement_handles = [
        axes[0].scatter(
            [],
            [],
            s=40,
            marker="s",
            facecolors=MUTED,
            edgecolors=MUTED,
            linewidths=1.4,
            label="Training only",
        ),
        axes[0].scatter(
            [],
            [],
            s=40,
            marker="o",
            facecolors=MUTED,
            edgecolors=MUTED,
            linewidths=1.4,
            label="Training and IO",
        ),
        axes[0].scatter(
            [],
            [],
            s=40,
            marker="^",
            facecolors=MUTED,
            edgecolors=MUTED,
            linewidths=1.4,
            label="IO only",
        ),
        Line2D(
            [],
            [],
            marker="x",
            ls="",
            color=MUTED,
            markersize=6,
            markeredgewidth=1.0,
            label="No losses, network output",
        ),
        Line2D(
            [],
            [],
            marker="*",
            ls="",
            color=MUTED,
            markersize=7,
            markeredgewidth=1.2,
            label="No losses, after IO",
        ),
    ]
    style = dict(
        frameon=True,
        edgecolor="#c9c8c3",
        fontsize=7,
        title_fontsize=7,
        handletextpad=0.5,
        columnspacing=1.0,
    )
    fig.legend(
        handles=loss_handles,
        title="PET-preservation and bone-rigidity losses used",
        ncol=2,
        loc="lower left",
        bbox_to_anchor=(0.018, 1.0135),
        **style,
    )
    fig.legend(
        handles=placement_handles,
        title="PET-preservation and bone-rigidity loss placement",
        ncol=2,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.965),
        **style,
    )
    fig.tight_layout()

    PAPER_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    tight_bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    top_extended_bbox = Bbox.from_extents(
        tight_bbox.x0,
        tight_bbox.y0,
        tight_bbox.x1,
        tight_bbox.y1 + 0.01,
    )
    fig.savefig(
        PAPER_FIGURE_DIR / "q2.pdf", bbox_inches=top_extended_bbox, pad_inches=0
    )
    text = effects(patients)
    analysis_out_dir = common.OUT_DIR / "figures"
    analysis_out_dir.mkdir(parents=True, exist_ok=True)
    (analysis_out_dir / "q2_effects.txt").write_text(text + "\n")
    n = len(patients["T1"])
    n_pet = patients["T1"]["mtv_bias"].notna().sum()
    pair_rows = common.load_run("T1")
    n_pairs = len(pair_rows)
    n_pet_pairs = pair_rows["mtv_bias"].notna().sum()
    tex = rf"""\begin{{figure}}[t]
\centering
\includegraphics[width=\textwidth]{{figures/q2.pdf}}
\caption{{Effect of the PET-preservation and bone-rigidity losses and of where they act on the test set: (a) metabolic tumor volume (MTV) change and (b) total lesion glycolysis (TLG) change against CT-organ Dice.
CT-organ Dice includes {n} patients and {n_pairs} pairs, while MTV and TLG include {n_pet} patients and {n_pet_pairs} pairs.
Colors identify which PET-preservation and bone-rigidity losses are used, including the configuration without either loss group.
Markers identify where the losses act: a square denotes the network output after training with the losses, a filled circle denotes the result after subsequent instance optimization (IO) with the same losses, and a filled triangle denotes IO using a network trained without these losses.
For the configuration without either loss group, $\times$ denotes the network output and $\star$ denotes the result after IO.
Lines connect configurations with the same loss configuration.
Each point is the mean over patients after first averaging over image pairs within each patient.
Uncertainty and significance tests are reported in Table~\ref{{tab:ablations}}.}}
\label{{fig:q2}}
\end{{figure}}
"""
    (PAPER_FIGURE_DIR / "q2.tex").write_text(tex)
    print(text)


if __name__ == "__main__":
    main()
