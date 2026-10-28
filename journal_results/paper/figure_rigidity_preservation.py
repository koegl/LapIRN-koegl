"""Experiment 2: effect of the rigidity weight on lesion preservation -> PDF + LaTeX.

Rigidity weight sweep on the validation set (T2 + PET IO with one rigidity
penalty at x0.1 ... x100 of its calibrated reference weight): |MTV change| and
|TLG change| against the weight factor, one line per penalty; dashed: no
rigidity (R1 = I2 on val); outlined: weight chosen for the test set. Patient
means from the per-pair CSVs of the sweep.

SWEEP_DIR: the sweep with the official scorer (curve.csv, chosen.json).

    ../../.venv/bin/python figure_rigidity_preservation.py
        ->  out/figures/rigidity_preservation.{pdf,tex}
"""

import json

import matplotlib

matplotlib.use("Agg")
import common  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

PAPER_FIGURE_DIR = common.REPO.parent / "psmareg-extension" / "figures"
SWEEP_DIR = common.REPO / "evaluation_official_scorer" / "rigidity_sweep"
CALIBRATION = (
    common.REPO / "journal_results" / "rigidity_calibration" / "reference.json"
)
# method -> (label, color); as figure_rigidity.py
METHODS = {
    "staring": ("Staring", "#eb6834"),
    "staring_eroded": ("Staring + erosion", "#1baf7a"),
    "kabsch": ("Per-bone rigid fit", "#2a78d6"),
}
METRICS = (
    ("mtv_bias", r"$|\Delta\mathrm{MTV}|$ (\%) $\downarrow$"),
    ("tlg_bias", r"$|\Delta\mathrm{TLG}|$ (\%) $\downarrow$"),
)
TEXT = "#0b0b0b"
ZERO_X = -2.0  # x position of weight 0 (log10 axis, set apart from x0.1)
MUTED = "#52514e"


def main() -> None:
    reference_weight = json.loads(CALIBRATION.read_text())["reference_weight"]
    chosen = json.loads((SWEEP_DIR / "chosen.json").read_text())["chosen"]
    chosen_factor = {
        c["method"]: round(c["weight"] / reference_weight[c["method"]], 6)
        for c in chosen.values()
    }
    all_rows = pd.read_csv(SWEEP_DIR / "curve.csv")
    for column, _ in METRICS:
        all_rows[column] *= 100
    reference = all_rows[all_rows["method"] == "none"].iloc[0]
    curve = all_rows[all_rows["method"] != "none"].copy()
    curve["factor"] = [
        round(w / reference_weight[m], 6)
        for m, w in zip(curve["method"], curve["weight"])
    ]

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
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.4), sharex=True)
    for ax, (metric, label) in zip(axes, METRICS):
        ax.axhline(reference[metric], color=MUTED, linewidth=0.8, linestyle=(0, (4, 2)))
        ax.scatter(ZERO_X, reference[metric], marker="*", s=90, color=TEXT, zorder=6)
        for method, (_, color) in METHODS.items():
            d = curve[curve["method"] == method].sort_values("factor")
            x = np.log10(d["factor"].to_numpy())
            # thin dotted link from no rigidity (weight 0) to the smallest weight
            ax.plot(
                [ZERO_X, x[0]],
                [reference[metric], d[metric].iloc[0]],
                color=color,
                linewidth=0.8,
                linestyle=(0, (1, 1.5)),
                zorder=2,
            )
            ax.plot(x, d[metric], color=color, linewidth=1.5, zorder=3)
            ax.scatter(
                x,
                d[metric],
                s=14,
                color=color,
                edgecolors="white",
                linewidths=0.6,
                zorder=4,
            )
            best = d[d["factor"] == chosen_factor[method]].iloc[0]
            ax.scatter(
                np.log10(best["factor"]),
                best[metric],
                s=40,
                color=color,
                edgecolors=TEXT,
                linewidths=1.0,
                zorder=5,
            )
        ax.set_xticks([ZERO_X, -1, 0, 1, 2], ["0", "0.1", "1", "10", "100"])
        ax.set_xlabel(r"Bone-rigidity weight (factor of the reference weight)")
        ax.set_ylabel(label)
        ax.grid(color="#e4e3df", linewidth=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax, panel_label in zip(axes, ("(a)", "(b)")):
        ax.text(
            -0.16,
            1.07,
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
            color=TEXT,
            marker="*",
            markersize=8,
            linestyle="",
            label="no bone rigidity (weight 0)",
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
        bbox_to_anchor=(0.5, 1.08),
        handlelength=1.8,
    )
    fig.tight_layout(w_pad=2.0)

    out_dir = PAPER_FIGURE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_dir / "rigidity_preservation.pdf", bbox_inches="tight", pad_inches=0
    )
    tex = r"""\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{figures/rigidity_preservation.pdf}
\caption{Effect of bone-rigidity weight on lesion preservation on the validation set (26 patients, 32 lesion-positive pairs): network trained with the PET-preservation losses, followed by instance optimization (IO) with the PET-preservation terms and one bone-rigidity penalty at $0.1$ to $100$ times its calibrated reference weight.
(a)~Absolute metabolic tumor volume (MTV) change; (b)~absolute total lesion glycolysis (TLG) change.
Color identifies the bone-rigidity penalty.
Star and dashed line: IO without bone rigidity (weight 0); dotted segment: connection to the smallest nonzero weight; outlined marker: weight chosen for the test set.
Points are means over patients after averaging each patient's lesion-positive pairs.}
\label{fig:rigidity_preservation}
\end{figure}
"""
    (out_dir / "rigidity_preservation.tex").write_text(tex)
    print(curve.round(2).to_string(index=False))
    print(
        "no bone rigidity:",
        reference[[m for m, _ in METRICS]].astype(float).round(2).to_dict(),
    )


if __name__ == "__main__":
    main()
