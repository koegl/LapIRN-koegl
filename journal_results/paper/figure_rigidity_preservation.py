"""Experiment 2: effect of the rigidity weight on lesion preservation -> PDF + LaTeX.

Rigidity weight sweep on the validation set (T2 + PET IO with one rigidity
penalty at x0.1 ... x100 of its calibrated reference weight): |MTV change| and
|TLG change| against the weight factor, one line per penalty; dashed: no
rigidity (R1 = I2 on val); outlined: weight chosen for the test set. Patient
means from the per-pair CSVs of the sweep.

SWEEP_DIR: the sweep folder; MTV / TLG do not depend on the bone metrics, so
the first sweep (rigidity_sweep_interior_metric) can be used until the rerun
with the whole-bone distortion is synced to rigidity_sweep.

    ../../.venv/bin/python figure_rigidity_preservation.py
        ->  out/figures/rigidity_preservation.{pdf,tex}
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import common  # noqa: E402

SWEEP_DIR = common.REPO / "journal_results" / "rigidity_sweep_interior_metric"
REFERENCE_CSV = common.REPO / "evaluation" / "val" / "I2_interior_metric.csv"
CALIBRATION = (
    common.REPO / "journal_results" / "rigidity_calibration" / "reference.json"
)
# chosen weights (evaluate.RIGIDITY_WEIGHTS, as factors of the reference weight)
CHOSEN_FACTOR = {"staring": 1, "staring_eroded": 100, "kabsch": 100}
# method -> (label, colour); as figure_rigidity.py
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
MUTED = "#52514e"


def patient_mean(csv) -> pd.Series:
    df = pd.read_csv(csv, dtype={"case_id": str})
    return df.groupby("case_id")[[m for m, _ in METRICS]].mean().mean() * 100


def main() -> None:
    sweep = json.loads(CALIBRATION.read_text())["sweep"]
    reference_weight = json.loads(CALIBRATION.read_text())["reference_weight"]
    reference = patient_mean(REFERENCE_CSV)
    rows = []
    for method in METHODS:
        for weight in sweep[method]:
            csv = SWEEP_DIR / method / "val" / f"{method}_w{weight:.3g}.csv"
            factor = round(weight / reference_weight[method], 6)
            rows.append({"method": method, "factor": factor, **patient_mean(csv)})
    curve = pd.DataFrame(rows)

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
        for method, (_, color) in METHODS.items():
            d = curve[curve["method"] == method].sort_values("factor")
            ax.plot(d["factor"], d[metric], color=color, linewidth=1.5, zorder=3)
            ax.scatter(
                d["factor"],
                d[metric],
                s=14,
                color=color,
                edgecolors="white",
                linewidths=0.6,
                zorder=4,
            )
            best = d[d["factor"] == CHOSEN_FACTOR[method]].iloc[0]
            ax.scatter(
                best["factor"],
                best[metric],
                s=40,
                color=color,
                edgecolors=TEXT,
                linewidths=1.0,
                zorder=5,
            )
        ax.set_xscale("log")
        ax.set_xticks([0.1, 1, 10, 100], ["0.1", "1", "10", "100"])
        ax.set_xlabel(r"Rigidity weight (factor of the reference weight)")
        ax.set_ylabel(label)
        ax.grid(color="#e4e3df", linewidth=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
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
            color=MUTED,
            linewidth=0.8,
            linestyle=(0, (4, 2)),
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
        bbox_to_anchor=(0.5, 1.08),
        handlelength=1.8,
    )
    fig.tight_layout(w_pad=2.0)

    out_dir = common.OUT_DIR / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_dir / "rigidity_preservation.pdf", bbox_inches="tight", pad_inches=0
    )
    tex = r"""\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{figures/rigidity_preservation.pdf}
\caption{Effect of the rigidity weight on lesion preservation (validation set): network trained with the PET losses, IO with the PET terms and one rigidity penalty at $0.1$ to $100$ times its calibrated reference weight. (a)~Lesion volume change, (b)~uptake change. Dashed: IO without rigidity; outlined: weight chosen for the test set. Points: patient means.}
\label{fig:rigidity_preservation}
\end{figure}
"""
    (out_dir / "rigidity_preservation.tex").write_text(tex)
    print(curve.round(2).to_string(index=False))
    print("no rigidity:", reference.round(2).to_dict())


if __name__ == "__main__":
    main()
