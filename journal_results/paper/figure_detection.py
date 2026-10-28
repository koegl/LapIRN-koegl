"""Experiment 3: is PET IO limited by missed lesions? -> PDF + LaTeX.

Figure version of table_detection_groups.py and table_detection_contributions.py
(same numbers, detection.py). Ground-truth lesions grouped by how much of each
the predicted lesion mask (the one IO sees) covers. Rows: |MTV change|,
|TLG change|. Columns:
- change within each group (evaluation metric over the group's lesions only),
  one line per group across the runs
- contribution of each group to the evaluation metric (only the group's
  lesions change), stacked per run; black tick: the evaluation metric itself
  (below the bar top because changes of different lesions partly cancel)

    ../../.venv/bin/python figure_detection.py  ->  out/figures/detection.{pdf,tex}
"""

import matplotlib

matplotlib.use("Agg")
import common  # noqa: E402
import detection  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
from detection import RUNS  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# group -> (legend label, color); categorical slots 1-3 of the reference palette
GROUP_STYLE = {
    "missed": ("missed", "#eb6834"),
    "partial": (r"partly covered ($<$50\,\%)", "#1baf7a"),
    "covered": (r"covered ($\geq$50\,\%)", "#2a78d6"),
}
RUN_LABELS = {
    "T1": "Network\nonly",
    "I1": "+ IO\nPET $\\times$0",
    "I5": "+ IO\nPET $\\times$0.1",
    "W1": "+ IO\nPET $\\times$1",
}
METRICS = (
    ("mtv_change", "lesion_voxels", r"$|\Delta\mathrm{MTV}|$ (\%)"),
    ("tlg_change", "uptake", r"$|\Delta\mathrm{TLG}|$ (\%)"),
)
TEXT = "#0b0b0b"
MUTED = "#52514e"
ALL_COLOR = "#0b0b0b"  # all lesions (evaluation metric)
PAPER_FIGURE_DIR = common.REPO.parent / "psmareg-extension" / "figures"


def main() -> None:
    runs = list(RUNS)
    within = {}  # (metric index, group) -> values per run
    contribution = {}
    total = {}
    for run in runs:
        les = detection.lesions(run)
        for m, (change, weight, _) in enumerate(METRICS):
            total[m, run] = detection.scan_metric(les, change, weight)
            for g in GROUP_STYLE:
                within.setdefault((m, g), []).append(
                    detection.scan_metric(les[les["group"] == g], change, weight)
                )
                contribution.setdefault((m, g), []).append(
                    detection.scan_metric(les, change, weight, les["group"] == g)
                )

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
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.2), sharex=True)
    x = np.arange(len(runs))
    for m, (_, _, ylabel) in enumerate(METRICS):
        ax_w, ax_c = axes[m]
        for g, (_, color) in GROUP_STYLE.items():
            ax_w.plot(x, within[m, g], color=color, linewidth=1.5, zorder=3)
            ax_w.scatter(
                x,
                within[m, g],
                s=18,
                color=color,
                edgecolors="white",
                linewidths=0.8,
                zorder=4,
            )
        all_lesions = [total[m, r] for r in runs]
        ax_w.plot(
            x,
            all_lesions,
            color=ALL_COLOR,
            linewidth=0.8,
            linestyle=(0, (3, 2)),
            zorder=3,
        )
        ax_w.scatter(
            x,
            all_lesions,
            s=18,
            color=ALL_COLOR,
            edgecolors="white",
            linewidths=0.8,
            zorder=4,
        )
        ax_w.set_ylim(bottom=0)
        ax_w.set_ylabel(ylabel)

        bottom = np.zeros(len(runs))
        for i, (g, (_, color)) in enumerate(GROUP_STYLE.items()):
            values = np.array(contribution[m, g])
            ax_c.bar(
                x,
                values,
                bottom=bottom,
                width=0.6,
                color=color,
                linewidth=0,
                zorder=3,
            )
            if i > 0:
                # white gap between stacked segments only (not at the baseline)
                ax_c.hlines(
                    bottom, x - 0.3, x + 0.3, color="white", linewidth=1.0, zorder=4
                )
            bottom += values
        ax_c.scatter(
            x,
            [total[m, r] for r in runs],
            marker="_",
            s=260,
            color=ALL_COLOR,
            linewidths=1.6,
            zorder=5,
        )
        for xi, r in zip(x, runs):
            ax_c.text(
                xi + 0.34,
                total[m, r],
                f"{total[m, r]:.1f}",
                va="center",
                ha="left",
                fontsize=6.5,
                color=TEXT,
            )
        ax_c.set_ylabel(ylabel)

        for ax in (ax_w, ax_c):
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=True))
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
            ax.grid(axis="y", color="#e4e3df", linewidth=0.5)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
    axes[0, 0].set_title("Change within each group", fontsize=8, color=TEXT)
    axes[0, 1].set_title(
        "Contribution to the evaluation metric", fontsize=8, color=TEXT
    )
    for ax, panel_label in zip(axes.flat, ("(a)", "(b)", "(c)", "(d)")):
        ax.text(
            -0.145,
            1.072,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
            color=TEXT,
        )
    for ax in axes[1]:
        ax.set_xticks(x, [RUN_LABELS[r] for r in runs], fontsize=7)

    handles = [
        Line2D([], [], color=c, linewidth=3.0, label=label)
        for label, c in GROUP_STYLE.values()
    ]
    handles.append(
        Line2D(
            [],
            [],
            marker="o",
            color=ALL_COLOR,
            linestyle=(0, (3, 2)),
            markerfacecolor=ALL_COLOR,
            markeredgecolor="white",
            markeredgewidth=0.8,
            markersize=4,
            linewidth=0.8,
            label="all lesions (evaluation metric)",
        )
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.04),
        handlelength=2.0,
    )
    fig.tight_layout(h_pad=1.0, w_pad=2.0)

    PAPER_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIGURE_DIR / "detection.pdf", bbox_inches="tight", pad_inches=0)
    shares = detection.shares()
    lesion_rows = detection.lesions("T1")
    n_patients = lesion_rows["case_id"].nunique()
    n_pairs = lesion_rows[["case_id", "tp_moving"]].drop_duplicates().shape[0]
    tex = rf"""\begin{{figure}}[t]
\centering
\includegraphics[width=\textwidth]{{figures/detection.pdf}}
\caption{{Effect of predicted lesion coverage on registration-induced metabolic tumor volume (MTV) and total lesion glycolysis (TLG) change in {n_patients} test patients ({shares["all"]} lesions, {n_pairs} image pairs).
The predicted mask used during instance optimization (IO) defines three groups: missed (0\,\% coverage), partly covered ($<50\,\%$), and covered ($\geq50\,\%$).
The groups contain {shares["missed"]}, {shares["partial"]}, and {shares["covered"]} lesions and account for 6\,\%, 6\,\%, and 88\,\% of total lesion volume, respectively.
The network is trained without PET-preservation losses.
For IO, PET $\times0$ denotes no PET-preservation losses, PET $\times0.1$ the validation-selected weights, and PET $\times1$ the training weights.
Panels (a) and (c) show the absolute change within each coverage group, with the all-lesion evaluation metric shown by the black dashed line.
Panels (b) and (d) show each group's contribution to the image-level change.
The stacked contributions can exceed the black evaluation value because opposing signed changes cancel in the image total.}}
\label{{fig:detection}}
\end{{figure}}
"""
    (PAPER_FIGURE_DIR / "detection.tex").write_text(tex)


if __name__ == "__main__":
    main()
