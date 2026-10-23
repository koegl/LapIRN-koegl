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
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import common  # noqa: E402
import detection  # noqa: E402
from detection import RUNS  # noqa: E402

# group -> (legend label, colour); categorical slots 1-3 of the reference palette
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
            ax_w.scatter(x, within[m, g], s=18, color=color, edgecolors="white", linewidths=0.8, zorder=4)
        ax_w.set_ylim(bottom=0)
        ax_w.set_ylabel(ylabel)

        bottom = np.zeros(len(runs))
        for i, (g, (_, color)) in enumerate(GROUP_STYLE.items()):
            values = np.array(contribution[m, g])
            ax_c.bar(x, values, bottom=bottom, width=0.6, color=color, linewidth=0, zorder=3)
            if i > 0:
                # white gap between stacked segments only (not at the baseline)
                ax_c.hlines(bottom, x - 0.3, x + 0.3, color="white", linewidth=1.0, zorder=4)
            bottom += values
        ax_c.scatter(
            x, [total[m, r] for r in runs], marker="_", s=260, color=TEXT, linewidths=1.6, zorder=5
        )
        for xi, r in zip(x, runs):
            ax_c.text(xi + 0.34, total[m, r], f"{total[m, r]:.1f}", va="center", ha="left", fontsize=6.5, color=TEXT)
        ax_c.set_ylabel(ylabel)

        for ax in (ax_w, ax_c):
            ax.grid(axis="y", color="#e4e3df", linewidth=0.5)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
    axes[0, 0].set_title("Change within each group", fontsize=8, color=TEXT)
    axes[0, 1].set_title("Contribution to the evaluation metric", fontsize=8, color=TEXT)
    for ax in axes[1]:
        ax.set_xticks(x, [RUN_LABELS[r] for r in runs], fontsize=7)

    handles = [Patch(facecolor=c, label=label) for label, c in GROUP_STYLE.values()]
    handles.append(Line2D([], [], marker="_", ls="", color=TEXT, markersize=12, markeredgewidth=1.6, label="evaluation metric (all lesions)"))
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.04), handlelength=1.2)
    fig.tight_layout(h_pad=1.0, w_pad=2.0)

    out_dir = common.OUT_DIR / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "detection.pdf", bbox_inches="tight", pad_inches=0)
    shares = detection.shares()
    lesion_rows = detection.lesions("T1")
    n_patients = lesion_rows["case_id"].nunique()
    n_pairs = lesion_rows[["case_id", "tp_moving"]].drop_duplicates().shape[0]
    tex = rf"""\begin{{figure}}[t]
\centering
\includegraphics[width=\textwidth]{{figures/detection.pdf}}
\caption{{Lesion volume and uptake change of all {shares['all']} in {n_patients} test patients ({n_pairs} pairs), including components smaller than five voxels, grouped by how much of each lesion the predicted lesion mask seen by IO covers: missed ({shares['missed']}), partly covered ({shares['partial']}), covered ({shares['covered']}). Network trained without PET losses; IO with the PET weights scaled by the given factor. Left: change within each group (per scan, the change of the summed volume or uptake of the group's lesions; mean over the patients with lesions in the group). Right: contribution of each group to the evaluation metric (the change per scan if only the group's lesions changed; mean over patients); black: the evaluation metric, below the stacked total because changes of different lesions in a scan partly cancel.}}
\label{{fig:detection}}
\end{{figure}}
"""
    (out_dir / "detection.tex").write_text(tex)


if __name__ == "__main__":
    main()
