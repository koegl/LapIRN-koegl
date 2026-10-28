"""Per-lesion volume change by lesion size and location (experiment 1, todo
step 6) on test -> PDF + LaTeX.

Lesions: 26-connected components of the moving scan's ground-truth lesion mask
(evaluate.py *_lesions.csv); "tiny" ones (< 5 voxels, interpolation noise) are
left out. Strata: soft tissue / bone-associated (>= 50 % of the voxels on a
bone label) x small (< 0.5 ml) / medium (< 5 ml) / large.

Median per-lesion |MTV change| (main text) and |TLG change| (appendix, same
layout); upper tail in lesions_summary.txt.
Columns: network output (T1-T4) and network + matched IO (I1-I4); one marker
per loss set. Error bars: 95 % CI from a patient-cluster bootstrap (patients
resampled with all their lesions). Signed changes and the lesion counts are
printed and written to out/figures/lesions_summary.txt.

    ../../.venv/bin/python figure_lesions.py
        ->  out/figures/lesions.{pdf,tex}, out/figures/lesions_tlg.{pdf,tex}
"""

from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import common  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

# loss set -> (network run, network + IO run, color); colors as in figure_q2
CONDITIONS: Dict[str, Tuple[str, str, str]] = {
    "None": ("T1", "I1", "#0b0b0b"),
    "PET": ("T2", "I2", "#2a78d6"),
    "Rigidity": ("T3", "I3", "#d1342f"),
    "PET + rigidity": ("T4", "I4", "#8b3fbf"),
}
DISPLAY_NAMES = {
    "None": "None",
    "PET": "PET preservation",
    "Rigidity": "Bone rigidity",
    "PET + rigidity": "PET preservation + bone rigidity",
}
# markers as in figure_q2: block -> (with losses, without losses)
MARKERS = (("s", "x"), ("o", "*"))
BLOCKS = ("Network", "Network + IO")
STRATA = [
    (False, "small"),
    (False, "medium"),
    (False, "large"),
    (True, "small"),
    (True, "medium"),
    (True, "large"),
]
STATS = {"median": 50}
N_BOOTSTRAP = 1000
SEED = 0
TEXT = "#0b0b0b"
MUTED = "#52514e"
PAPER_FIGURE_DIR = common.REPO.parent / "psmareg-extension" / "figures"


def load_lesions(run_id: str) -> pd.DataFrame:
    df = pd.read_csv(common.EVAL_DIR / f"{run_id}_lesions.csv", dtype={"case_id": str})
    return df[df["size_bin"] != "tiny"]


def stat_ci(df: pd.DataFrame, q: float, rng, column: str) -> Tuple[float, float, float]:
    """Percentile q of |change| of a column (in %), with a patient-cluster
    bootstrap CI."""
    values = df[column].to_numpy() * 100
    groups = [g[column].to_numpy() * 100 for _, g in df.groupby("case_id")]
    samples = []
    for _ in range(N_BOOTSTRAP):
        picks = rng.integers(0, len(groups), len(groups))
        samples.append(np.percentile(np.concatenate([groups[i] for i in picks]), q))
    return (
        np.percentile(values, q),
        np.percentile(samples, 2.5),
        np.percentile(samples, 97.5),
    )


# figure -> (column, label, output name)
METRICS = {
    "MTV": ("mtv_bias", "lesions"),
    "TLG": ("tlg_bias", "lesions_tlg"),
}


def main(metric: str) -> None:
    column, name_out = METRICS[metric]
    lesions = {r: load_lesions(r) for runs in CONDITIONS.values() for r in runs[:2]}
    rng = np.random.default_rng(SEED)

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
        len(STATS),
        len(BLOCKS),
        figsize=(7.0, 2.6),
        sharex=True,
        sharey="row",
        squeeze=False,
    )
    offsets = np.linspace(-0.27, 0.27, len(CONDITIONS))
    x = np.arange(len(STRATA))
    for col, block in enumerate(BLOCKS):
        for row, (stat_name, q) in enumerate(STATS.items()):
            ax = axes[row, col]
            for offset, (name, runs) in zip(offsets, CONDITIONS.items()):
                df = lesions[runs[col]]
                est = [
                    stat_ci(
                        df[(df["bone_associated"] == bone) & (df["size_bin"] == size)],
                        q,
                        rng,
                        column,
                    )
                    for bone, size in STRATA
                ]
                mid = np.array([e[0] for e in est])
                err = np.array(
                    [
                        [m - e[1] for m, e in zip(mid, est)],
                        [e[2] - m for m, e in zip(mid, est)],
                    ]
                )
                marker = MARKERS[col][name == "None"]
                ax.errorbar(
                    x + offset,
                    mid,
                    yerr=err,
                    fmt=marker,
                    color=runs[2],
                    markersize=(7 if col == 1 else 5) if name == "None" else 3.5,
                    markeredgewidth=1.0,
                    elinewidth=0.8,
                    capsize=0,
                )
            ax.axvline(2.5, color=MUTED, linewidth=0.5)
            ax.grid(axis="y", color="#e4e3df", linewidth=0.5)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if row == 0:
                ax.set_title(block, fontsize=9)
            if col == 0:
                ax.set_ylabel(rf"{stat_name} $|\Delta\mathrm{{{metric}}}|$ (\%)")
    for ax in axes[-1]:
        ax.set_xticks(x)
        ax.set_xticklabels([s for _, s in STRATA])
        for xpos, label in ((1, "soft tissue"), (4, "bone")):
            ax.text(
                xpos,
                -0.13,
                label,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
            )
    for ax, panel_label in zip(axes.flat, ("(a)", "(b)")):
        ax.text(
            -0.128,
            1.064,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
            color=TEXT,
        )
    loss_handles = [
        Line2D(
            [],
            [],
            ls="-",
            color=c,
            linewidth=3.0,
            label=DISPLAY_NAMES[n],
        )
        for n, (_, _, c) in CONDITIONS.items()
    ]
    placement_handles = [
        axes[0, 0].scatter(
            [],
            [],
            s=40,
            marker="s",
            facecolors=MUTED,
            edgecolors=MUTED,
            linewidths=1.4,
            label="Training only",
        ),
        axes[0, 0].scatter(
            [],
            [],
            s=40,
            marker="o",
            facecolors=MUTED,
            edgecolors=MUTED,
            linewidths=1.4,
            label="Training and IO",
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
            markersize=9,
            markeredgewidth=1.0,
            label="No losses, after IO",
        ),
    ]
    fig.legend(
        handles=loss_handles,
        title="PET-preservation and bone-rigidity losses used",
        loc="lower left",
        ncol=2,
        frameon=True,
        bbox_to_anchor=(0.04, 0.98),
        fontsize=7,
        title_fontsize=7,
        handletextpad=0.5,
        columnspacing=1.0,
    )
    fig.legend(
        handles=placement_handles,
        title="PET-preservation and bone-rigidity loss placement",
        loc="lower right",
        ncol=2,
        frameon=True,
        bbox_to_anchor=(0.97, 0.98),
        fontsize=7,
        title_fontsize=7,
        handletextpad=0.5,
        columnspacing=1.0,
    )
    fig.tight_layout()

    # numbers for the text: counts and median signed / absolute change
    lines = []
    for name, runs in CONDITIONS.items():
        for block, run in zip(BLOCKS, runs[:2]):
            df = lesions[run]
            for bone, size in STRATA:
                s = df[(df["bone_associated"] == bone) & (df["size_bin"] == size)]
                lines.append(
                    f"{run:3s} {name:15s} {'bone' if bone else 'soft':4s} {size:6s} n={len(s):4d} "
                    f"median signed MTV {100 * s['mtv_change'].median():+.1f} %  "
                    f"|MTV| {100 * s['mtv_bias'].median():.1f} / P90 {100 * s['mtv_bias'].quantile(0.9):.1f} %  "
                    f"|TLG| {100 * s['tlg_bias'].median():.1f} %"
                )

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
        PAPER_FIGURE_DIR / f"{name_out}.pdf",
        bbox_inches=top_extended_bbox,
        pad_inches=0,
    )
    (common.OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
    (common.OUT_DIR / "figures" / "lesions_summary.txt").write_text(
        "\n".join(lines) + "\n"
    )
    n_lesions = len(lesions["T1"])
    n_excluded = len(pd.read_csv(common.EVAL_DIR / "T1_lesions.csv")) - n_lesions
    n_patients = lesions["T1"]["case_id"].nunique()
    n_pairs = lesions["T1"][["case_id", "tp_moving"]].drop_duplicates().shape[0]
    if metric == "MTV":
        caption = rf"""Per-lesion absolute volume change on the test set ({n_lesions} lesions from {n_patients} patients and {n_pairs} pairs after excluding {n_excluded} components smaller than five voxels), stratified by location (soft tissue or bone-associated, with $\geq$50\,\% of the lesion on bone) and size (small $<0.5$~ml, medium $0.5$ to $<5$~ml, large $\geq5$~ml).
Points show the median per-lesion $|\Delta\mathrm{{MTV}}|$.
Panel (a) shows the network output, and panel (b) shows the result after instance optimization (IO) with the same PET-preservation and bone-rigidity loss configuration.
Colors identify which PET-preservation and bone-rigidity losses are applied.
Squares denote the network output after training with the losses, and circles denote the result after subsequent IO with the same losses.
For the configuration without either loss group, $\times$ denotes the network output and $\star$ denotes the result after IO.
Error bars show 95\,\% confidence intervals from a patient-cluster bootstrap."""
    else:
        caption = rf"""Per-lesion absolute total lesion glycolysis (TLG) change on the test set ({n_lesions} lesions from {n_patients} patients and {n_pairs} pairs after excluding {n_excluded} components smaller than five voxels), stratified by location (soft tissue or bone-associated, with $\geq$50\,\% of the lesion on bone) and size (small $<0.5$~ml, medium $0.5$ to $<5$~ml, large $\geq5$~ml).
Points show the median per-lesion $|\Delta\mathrm{{TLG}}|$.
Panel (a) shows the network output, and panel (b) shows the result after instance optimization (IO) with the same PET-preservation and bone-rigidity loss configuration.
Colors identify which PET-preservation and bone-rigidity losses are applied.
Squares denote the network output after training with the losses, and circles denote the result after subsequent IO with the same losses.
For the configuration without either loss group, $\times$ denotes the network output and $\star$ denotes the result after IO.
Error bars show 95\,\% confidence intervals from a patient-cluster bootstrap."""
    tex = rf"""\begin{{figure}}[t]
\centering
\includegraphics[width=\textwidth]{{figures/{name_out}.pdf}}
\caption{{{caption}}}
\label{{fig:{name_out}}}
\end{{figure}}
"""
    (PAPER_FIGURE_DIR / f"{name_out}.tex").write_text(tex)
    print("\n".join(lines))


if __name__ == "__main__":
    for metric in METRICS:
        main(metric)
